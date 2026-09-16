"""LLM-based particle finder using Claude or OpenAI vision capabilities.

Uses Claude or OpenAI to directly analyze TIFF images and identify particles based on
natural language queries, returning coordinates for high-resolution scanning.
"""

import os
import io
import json
import base64
import numpy as np
import tifffile as tiff
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple


class LLMVisionParticleFinder:
    """Use LLM vision (Claude or OpenAI) to find particles in images based on natural language."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 provider: Optional[str] = None):
        """
        Initialize the LLM vision particle finder.

        Args:
            api_key: API key (tries AIFAPIM_API_KEY, then provider-specific env vars)
            base_url: Custom API endpoint (e.g., BNL Hermes gateway)
            model: Model name to use
            provider: LLM provider ("claude" or "openai"). Auto-detected if not specified.
        """
        # Auto-detect provider from environment or use explicit parameter
        self.provider = provider or os.environ.get('LLM_PROVIDER', 'claude').lower()

        if self.provider not in ['claude', 'openai']:
            raise ValueError(f"Provider must be 'claude' or 'openai', got: {self.provider}")

        # Set up API key
        if self.provider == 'claude':
            self.api_key = api_key or os.environ.get('AIFAPIM_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
            self.base_url = base_url or os.environ.get('ANTHROPIC_BASE_URL')
            self.model = model or os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
        else:  # openai
            self.api_key = api_key or os.environ.get('AIFAPIM_API_KEY') or os.environ.get('OPENAI_API_KEY')
            self.base_url = base_url or os.environ.get('OPENAI_BASE_URL', 'https://hermes.nsls2.bnl.gov/openai/v1')
            self.model = model or os.environ.get('OPENAI_MODEL', 'gpt-5.6-luna')

        self.loaded_images = {}  # element -> numpy array
        self.image_shape = None
        self.microns_per_pixel_x = 1.0
        self.microns_per_pixel_y = 1.0

    def load_tiff_images(self, tiff_paths: List[str], element_names: List[str],
                        microns_per_pixel_x: float = 1.0,
                        microns_per_pixel_y: float = 1.0):
        """
        Load TIFF images for analysis.

        Args:
            tiff_paths: List of paths to TIFF files
            element_names: List of element names corresponding to each TIFF
            microns_per_pixel_x: Calibration for x-axis
            microns_per_pixel_y: Calibration for y-axis
        """
        self.loaded_images = {}
        self.microns_per_pixel_x = microns_per_pixel_x
        self.microns_per_pixel_y = microns_per_pixel_y

        for path, element in zip(tiff_paths, element_names):
            img = tiff.imread(path).astype(np.float32)
            self.loaded_images[element] = img

            if self.image_shape is None:
                self.image_shape = img.shape

    def _tiff_to_base64_png(self, img_array: np.ndarray, normalize: bool = True) -> str:
        """
        Convert TIFF numpy array to base64-encoded PNG for vision APIs.

        Args:
            img_array: 2D numpy array
            normalize: Whether to normalize to 0-255 range

        Returns:
            Base64-encoded PNG string
        """
        # Normalize to 0-255
        if normalize:
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(img_array, dtype=np.uint8)
        else:
            img_normalized = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert to PIL Image
        pil_img = Image.fromarray(img_normalized)

        # Convert to PNG bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()

        # Encode to base64
        return base64.b64encode(png_bytes).decode('utf-8')

    def _create_composite_image(self, elements: List[str]) -> str:
        """
        Create matplotlib figure with RGB composite + colored element panels.
        Matches the format from /tmp/automap_z2oajwr__llm_vision_composite.png.

        Args:
            elements: List of element names to composite (max 3)

        Returns:
            Base64-encoded PNG of the composite figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Get images for specified elements
        images = [self.loaded_images.get(elem) for elem in elements[:3]]
        valid_images = [(elem, img) for elem, img in zip(elements[:3], images) if img is not None]

        if not valid_images:
            raise ValueError("No valid images to composite")

        # Normalize each channel for RGB composite
        channels = []
        color_names = ["RED", "GREEN", "BLUE"]
        color_cmaps = ['Reds', 'Greens', 'Blues']  # Matplotlib colormaps

        for elem, img in valid_images:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(img, dtype=np.uint8)
            channels.append(normalized)

        # Pad to 3 channels if needed
        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0], dtype=np.uint8))

        # Create RGB composite
        rgb_composite = np.stack(channels[:3], axis=-1)

        # Create matplotlib figure
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(len(valid_images), 2, width_ratios=[3, 1],
                               hspace=0.05, wspace=0.05)

        # Main RGB composite (left, spans all rows)
        ax_main = fig.add_subplot(gs[:, 0])
        ax_main.imshow(rgb_composite, aspect='auto', origin='upper')
        ax_main.set_xlabel('Pixels (X)', fontsize=10)
        ax_main.set_ylabel('Pixels (Y)', fontsize=10)
        ax_main.grid(False)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = []
        for i, (elem, _) in enumerate(valid_images):
            color = ['red', 'green', 'blue'][i]
            legend_elements.append(Patch(facecolor=color, label=elem))
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Individual element panels (right, in their channel colors)
        for i, (elem, img) in enumerate(valid_images):
            ax_panel = fig.add_subplot(gs[i, 1])

            # Normalize for display
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(img, dtype=np.float32)

            # Display with colored colormap (Reds for Ca, Greens for Ce, Blues for P)
            ax_panel.imshow(normalized, cmap=color_cmaps[i], aspect='auto', origin='upper')
            ax_panel.set_title(elem, fontsize=10, pad=3)
            ax_panel.set_xticks([])
            ax_panel.set_yticks([])

        plt.tight_layout()

        # Store figure and axis for later drawing of circles
        self._fig = fig
        self._ax_main = ax_main
        self._dpi = 100

        # Save to buffer for LLM
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=self._dpi, bbox_inches='tight')

        # Save to temp file for GUI (without circles initially)
        import tempfile
        self._last_composite_path = tempfile.mktemp(suffix='_composite.png', prefix='automap_gui_')
        plt.savefig(self._last_composite_path, format='PNG', dpi=self._dpi, bbox_inches='tight')

        # Don't close figure - we'll reuse it to draw circles
        # plt.close(fig)

        png_bytes = buffer.getvalue()
        return base64.b64encode(png_bytes).decode('utf-8')

    def redraw_composite_with_circles(self, particles):
        """Redraw the composite figure with circles overlaid on particles.

        Args:
            particles: List of particle dicts with 'center_px' in data coordinates
        """
        if not hasattr(self, '_fig') or not hasattr(self, '_ax_main'):
            return  # No figure to draw on

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Remove any existing circles
        if hasattr(self, '_circle_patches'):
            for patch in self._circle_patches:
                patch.remove()
        self._circle_patches = []

        # Draw circles at data coordinates
        for particle in particles:
            x_data, y_data = particle['center_px']
            size_px = particle.get('size_px', 20)
            radius = size_px / 2 + 5

            # Create circle in data coordinates
            circle = Circle((x_data, y_data), radius,
                          fill=False, edgecolor='cyan', linewidth=2, zorder=100)
            self._ax_main.add_patch(circle)
            self._circle_patches.append(circle)

        # Redraw and save
        self._fig.canvas.draw()
        plt.savefig(self._last_composite_path, format='PNG', dpi=self._dpi, bbox_inches='tight')

        # Also save for return
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=self._dpi, bbox_inches='tight')
        return buffer.getvalue()

    def highlight_particles(self, particles_to_highlight):
        """Add yellow highlight circles to selected particles.

        Args:
            particles_to_highlight: List of particle dicts to highlight
        """
        if not hasattr(self, '_fig') or not hasattr(self, '_ax_main'):
            return

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        # Remove existing highlights
        if hasattr(self, '_highlight_patches'):
            for patch in self._highlight_patches:
                patch.remove()
        self._highlight_patches = []

        # Add yellow highlight circles
        for particle in particles_to_highlight:
            x_data, y_data = particle['center_px']
            size_px = particle.get('size_px', 20)
            radius = size_px / 2 + 10  # Larger for highlight

            circle = Circle((x_data, y_data), radius,
                          fill=False, edgecolor='yellow', linewidth=3, zorder=150)
            self._ax_main.add_patch(circle)
            self._highlight_patches.append(circle)

        # Redraw and save
        self._fig.canvas.draw()
        plt.savefig(self._last_composite_path, format='PNG', dpi=self._dpi, bbox_inches='tight')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=self._dpi, bbox_inches='tight')
        return buffer.getvalue()

    def _build_system_prompt(self, element_names: List[str], max_results: int) -> str:
        """Build system prompt for particle detection."""
        elem1 = element_names[0] if len(element_names) > 0 else '?'
        elem2 = element_names[1] if len(element_names) > 1 else '?'
        elem3 = element_names[2] if len(element_names) > 2 else '?'

        return f"""This is an X-ray fluorescence (XRF) microscopy image showing:
- LEFT: RGB composite where Red={elem1}, Green={elem2}, Blue={elem3}
- RIGHT: Individual element intensity maps in their respective colors

The main plot (LEFT) has axis labels showing pixel coordinates.

Color interpretation:
- Pure red = {elem1} only
- Pure green = {elem2} only
- Pure blue = {elem3} only
- Yellow = {elem1}+{elem2} overlap
- Magenta = {elem1}+{elem3} overlap
- Cyan = {elem2}+{elem3} overlap
- White = all three overlap

**Coordinates:**
Read the axis labels on the main plot. Return coordinates in data space (0-{self.image_shape[1]-1} for X, 0-{self.image_shape[0]-1} for Y).

Use BOTH the RGB composite AND the individual colored element maps (right side) to identify particles accurately.

Return results as JSON array with up to {max_results} particles, sorted by relevance/confidence.

JSON format:
[
  {{
    "id": 1,
    "center_px": [x, y],
    "center_um": [x_microns, y_microns],
    "size_px": estimated_diameter,
    "confidence": 0.95,
    "description": "bright spot with strong signal"
  }},
  ...
]

IMPORTANT:
- Return ONLY the JSON array, no other text
- Coordinates must be within image bounds
- Focus on the most relevant particles matching the query
- If no particles match, return empty array []
"""

    def _build_user_message(self, query: str, element_names: List[str], max_results: int) -> str:
        """Build user message for particle detection."""
        return f"""{query}

Return pixel coordinates (x, y) for the center of each matching particle/region.

Coordinate constraints:
- X: 0 to {self.image_shape[1]-1}
- Y: 0 to {self.image_shape[0]-1}
- Origin (0,0) is top-left corner

Return up to {max_results} results as JSON array:
[
  {{"id": 1, "center_px": [x, y], "size_px": diameter, "confidence": 0.95, "description": "brief description"}},
  ...
]"""

    def _call_claude_vision(self, system_prompt: str, user_message: str,
                           composite_b64: str, max_tokens: int = 4096) -> str:
        """Call Claude API for vision analysis."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # Create client with custom endpoint if provided
        if self.base_url:
            client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        else:
            client = anthropic.Anthropic(api_key=self.api_key)

        # Call Claude with vision
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": composite_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ],
                }
            ],
        )

        # Extract text from response (skip thinking blocks)
        response_text = next(
            (block.text for block in message.content if block.type == "text"),
            None
        )
        if not response_text:
            raise ValueError("No text response from Claude")

        return response_text

    def _call_openai_vision(self, system_prompt: str, user_message: str,
                           composite_b64: str, max_tokens: int = 4096) -> str:
        """Call OpenAI API for vision analysis (supports both standard and BNL Hermes format)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Create client with custom endpoint and headers (BNL Hermes requires x-api-key header)
        client_kwargs = {"api_key": self.api_key}

        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            # BNL Hermes requires explicit x-api-key header
            if "hermes.nsls2.bnl.gov" in self.base_url:
                client_kwargs["default_headers"] = {"x-api-key": self.api_key}

        client = OpenAI(**client_kwargs)

        # Check if this is BNL Hermes endpoint (uses custom API format)
        is_bnl_hermes = self.base_url and "hermes.nsls2.bnl.gov" in self.base_url

        if is_bnl_hermes:
            # BNL Hermes uses custom format: client.responses.create()
            # Image format: type="input_image" + image_url field (not url, not data, not source)
            input_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{composite_b64}"
                        },
                        {
                            "type": "input_text",
                            "text": user_message
                        }
                    ]
                }
            ]

            # BNL Hermes custom API format
            response = client.responses.create(
                model=self.model,
                input=input_messages
            )

            # Extract text from BNL Hermes response format (with error handling)
            # BNL may return reasoning blocks first, then text blocks - iterate to find text
            response_text = None
            try:
                if hasattr(response, 'output') and response.output:
                    if isinstance(response.output, list) and len(response.output) > 0:
                        # Iterate through output items to find text content (skip reasoning blocks)
                        for output_item in response.output:
                            # Skip reasoning/thinking blocks (they have encrypted_content, not text)
                            if hasattr(output_item, 'type') and output_item.type == 'reasoning':
                                continue

                            # Look for text in content
                            if hasattr(output_item, 'content') and output_item.content:
                                if isinstance(output_item.content, list) and len(output_item.content) > 0:
                                    for content_item in output_item.content:
                                        if hasattr(content_item, 'text') and content_item.text:
                                            response_text = content_item.text
                                            break
                                elif isinstance(output_item.content, str):
                                    # Sometimes content is directly a string
                                    response_text = output_item.content
                                    break

                            if response_text:
                                break

                        if not response_text:
                            # Debug: show what we got
                            output_types = [getattr(item, 'type', 'unknown') for item in response.output]
                            raise ValueError(f"No text content found in response. Output types: {output_types}. First item: {response.output[0]}")
                    else:
                        raise ValueError(f"Empty or invalid output array. Response: {response}")
                else:
                    raise ValueError(f"No 'output' field in response. Response: {response}")

                if not response_text:
                    raise ValueError("Empty text response from OpenAI (BNL Hermes)")
            except (IndexError, AttributeError, TypeError) as e:
                raise ValueError(f"Failed to parse BNL Hermes response: {e}. Response: {response}")

        else:
            # Standard OpenAI format
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{composite_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_message
                        }
                    ]
                }
            ]

            # Standard OpenAI API
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens
            )

            # Extract text from standard OpenAI response
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("No text response from OpenAI")

        return response_text

    def find_particles(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Use LLM vision to find particles based on natural language query.

        Args:
            query: Natural language description of particles to find
            max_results: Maximum number of particles to return

        Returns:
            Dict with particle locations and metadata
        """
        if not self.api_key:
            raise ValueError(f"No API key provided for {self.provider}. Set appropriate environment variable.")

        if not self.loaded_images:
            raise ValueError("No images loaded. Call load_tiff_images() first.")

        # Create composite image
        element_names = list(self.loaded_images.keys())
        composite_b64 = self._create_composite_image(element_names)

        # Save composite image for debugging (so user can see what LLM sees)
        import tempfile
        debug_path = tempfile.mktemp(suffix='_llm_vision_composite.png', prefix='automap_')
        try:
            composite_png = base64.b64decode(composite_b64)
            with open(debug_path, 'wb') as f:
                f.write(composite_png)
            print(f"DEBUG: Saved composite image (what LLM sees) to: {debug_path}")
        except Exception as e:
            print(f"Warning: Could not save debug image: {e}")

        # Build prompts
        system_prompt = self._build_system_prompt(element_names, max_results)
        user_message = self._build_user_message(query, element_names, max_results)

        # Call appropriate API
        if self.provider == 'claude':
            response_text = self._call_claude_vision(system_prompt, user_message, composite_b64)
        else:  # openai
            response_text = self._call_openai_vision(system_prompt, user_message, composite_b64)

        # Extract JSON from response (handles code blocks)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        # Parse JSON
        particles = json.loads(response_text)

        # Calculate micron coordinates if not already present
        for particle in particles:
            if 'center_um' not in particle and 'center_px' in particle:
                x_px, y_px = particle['center_px']
                particle['center_um'] = [
                    x_px * self.microns_per_pixel_x,
                    y_px * self.microns_per_pixel_y
                ]

        return {
            'query': query,
            'particles': particles,
            'count': len(particles),
            'image_shape': self.image_shape,
            'elements': element_names,
            'provider': self.provider,
            'model': self.model,
            'calibration': {
                'microns_per_pixel_x': self.microns_per_pixel_x,
                'microns_per_pixel_y': self.microns_per_pixel_y
            }
        }

    def get_scan_coordinates(self, particles: List[Dict]) -> List[Dict[str, Any]]:
        """
        Convert particle locations to high-resolution scan coordinates.

        Args:
            particles: List of particle dicts with 'center_um' field

        Returns:
            List of scan coordinate dicts ready for queue submission
        """
        scan_coords = []

        for i, particle in enumerate(particles):
            if 'center_um' not in particle:
                continue

            x_um, y_um = particle['center_um']

            scan_coords.append({
                'id': particle.get('id', i),
                'x_center_um': x_um,
                'y_center_um': y_um,
                'description': particle.get('description', 'LLM-identified particle'),
                'confidence': particle.get('confidence', 1.0),
                'size_px': particle.get('size_px', 0),
            })

        return scan_coords

    def explain_query(self, query: str) -> str:
        """
        Get an explanation of what the vision system will look for.

        Args:
            query: Natural language query

        Returns:
            Human-readable explanation
        """
        if not self.api_key:
            raise ValueError(f"No API key provided for {self.provider}.")

        element_names = list(self.loaded_images.keys())

        prompt = f"""You are analyzing X-ray fluorescence microscopy images with elements: {', '.join(element_names)}.

Given the query: "{query}"

Explain in 2-3 sentences what visual features you will look for in the image to identify matching particles.
Focus on: brightness, color (element), spatial patterns, size, and overlap characteristics.

Be specific and technical."""

        # Call appropriate API (text-only, no image)
        if self.provider == 'claude':
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")

            if self.base_url:
                client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
            else:
                client = anthropic.Anthropic(api_key=self.api_key)

            message = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract text from response (skip thinking blocks)
            response_text = next(
                (block.text for block in message.content if block.type == "text"),
                None
            )
            if not response_text:
                raise ValueError("No text response from Claude")

            return response_text

        else:  # openai
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")

            if self.base_url:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )

            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("No text response from OpenAI")

            return response_text
