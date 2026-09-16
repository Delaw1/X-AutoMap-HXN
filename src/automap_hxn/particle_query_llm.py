"""LLM-powered natural language particle query system.

Allows users to query detected particles using natural language:
- "top 5 spots with Fe and S overlapping but no Ca"
- "all particles with high calcium intensity"
- "particles in the upper right quadrant larger than 100 pixels"
- "brightest 10 iron particles"
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class ParticleQueryLLM:
    """Natural language query interface for particle data."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the particle query system.

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.particles_data = None
        self.image_shape = None
        self.element_colors = {}  # Map elements to colors (e.g., {'Ca': 'red', 'Fe': 'green'})

    def load_detection_results(self, blobs_by_color: Dict, union_blobs: Dict,
                              image_shape: tuple, element_mapping: Dict):
        """
        Load detection results for querying.

        Args:
            blobs_by_color: Dict of {color: [blob1, blob2, ...]}
            union_blobs: Dict of union detection results
            image_shape: (height, width) of images
            element_mapping: Dict of {element: color}, e.g., {'Ca': 'red'}
        """
        self.image_shape = image_shape
        self.element_colors = element_mapping

        # Build unified particle database
        self.particles_data = {
            'individual_particles': [],
            'union_particles': [],
            'metadata': {
                'image_shape': image_shape,
                'element_mapping': element_mapping,
                'colors_to_elements': {v: k for k, v in element_mapping.items()}
            }
        }

        # Add individual particles
        for color, blob_list in blobs_by_color.items():
            element = self.particles_data['metadata']['colors_to_elements'].get(color, color)
            for blob in blob_list:
                particle = {
                    'id': f"{element}_{len(self.particles_data['individual_particles'])}",
                    'type': 'individual',
                    'element': element,
                    'color': color,
                    'center_px': blob['center'],
                    'radius_px': blob['radius'],
                    'area_px': np.pi * blob['radius']**2,
                    'max_intensity': blob.get('max_intensity', 0),
                    'mean_intensity': blob.get('mean_intensity', 0),
                    'box_x': blob.get('box_x', blob['center'][0] - blob['radius']),
                    'box_y': blob.get('box_y', blob['center'][1] - blob['radius']),
                    'box_size': blob.get('box_size', blob['radius'] * 2),
                }
                self.particles_data['individual_particles'].append(particle)

        # Add union particles
        for union_id, union_data in union_blobs.items():
            # Determine which elements are present (from the union formation)
            elements_present = []
            if 'elements' in union_data:
                elements_present = list(union_data['elements'].keys())
            elif 'group_name' in union_data:
                # Parse from group name like "CaFeS"
                elements_present = self._parse_elements_from_group(union_data['group_name'])

            particle = {
                'id': f"union_{union_id}",
                'type': 'union',
                'elements': elements_present,
                'center_px': union_data.get('image_center', union_data.get('center', [0, 0])),
                'area_px': union_data.get('image_area_px²', union_data.get('area', 0)),
                'size_px': union_data.get('image_length', 0),
                'real_center_um': union_data.get('real_center_um', [0, 0]),
                'real_size_um': union_data.get('real_size_um', [0, 0]),
            }
            self.particles_data['union_particles'].append(particle)

    def query(self, natural_language_query: str, return_format: str = 'summary') -> Dict:
        """
        Query particles using natural language.

        Args:
            natural_language_query: User's query in plain English
            return_format: 'summary', 'full', 'coordinates', or 'ids'

        Returns:
            Dict with query results and metadata
        """
        if not self.api_key:
            raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")

        if not self.particles_data:
            raise ValueError("No particle data loaded. Call load_detection_results() first.")

        # Convert query to structured filter using LLM
        structured_query = self._llm_parse_query(natural_language_query)

        # Execute the structured query
        results = self._execute_query(structured_query)

        # Format results based on requested format
        formatted = self._format_results(results, return_format)

        return {
            'query': natural_language_query,
            'structured_query': structured_query,
            'results': formatted,
            'count': len(results)
        }

    def _llm_parse_query(self, query: str) -> Dict:
        """
        Use LLM to convert natural language query to structured filter.

        Args:
            query: Natural language query

        Returns:
            Structured query dict with filters and operations
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)

        # Build prompt with current data context
        system_prompt = f"""You are a particle query parser for X-ray fluorescence microscopy data.

Convert natural language queries into structured JSON filters.

Available data:
- Elements detected: {list(self.element_colors.keys())}
- Image dimensions: {self.image_shape} pixels
- Particle types: individual (single element) and union (multi-element overlap)

Individual particles have:
- element: which element (e.g., "Ca", "Fe", "S")
- center_px: [x, y] coordinates in pixels
- radius_px: particle radius
- area_px: particle area
- max_intensity: maximum pixel intensity (0-255)
- mean_intensity: average pixel intensity
- box coordinates: box_x, box_y, box_size

Union particles have:
- elements: list of overlapping elements
- center_px: [x, y] coordinates
- area_px: total area
- real_center_um: [x, y] in microns

Output a JSON object with these fields:
{{
  "type": "individual" | "union" | "both",
  "element_filters": {{
    "required": ["element1", "element2"],  // Must have these
    "forbidden": ["element3"],              // Must NOT have these
    "any_of": ["element4", "element5"]      // Has at least one
  }},
  "property_filters": [
    {{"field": "max_intensity", "operator": ">", "value": 100}},
    {{"field": "area_px", "operator": ">=", "value": 200}}
  ],
  "spatial_filters": [
    {{"region": "upper_right"}},            // quadrant
    {{"x_range": [0, 256], "y_range": [0, 256]}}  // explicit range
  ],
  "sort": {{"field": "max_intensity", "order": "desc"}},
  "limit": 5
}}

Examples:

Query: "top 5 spots with Fe and S overlapping but no Ca"
Output: {{
  "type": "union",
  "element_filters": {{"required": ["Fe", "S"], "forbidden": ["Ca"]}},
  "sort": {{"field": "area_px", "order": "desc"}},
  "limit": 5
}}

Query: "all calcium particles brighter than 150"
Output: {{
  "type": "individual",
  "element_filters": {{"required": ["Ca"]}},
  "property_filters": [{{"field": "max_intensity", "operator": ">", "value": 150}}]
}}

Query: "10 largest iron particles in the upper right"
Output: {{
  "type": "individual",
  "element_filters": {{"required": ["Fe"]}},
  "spatial_filters": [{{"region": "upper_right"}}],
  "sort": {{"field": "area_px", "order": "desc"}},
  "limit": 10
}}

Query: "particles with iron or sulfur, area > 500 pixels"
Output: {{
  "type": "individual",
  "element_filters": {{"any_of": ["Fe", "S"]}},
  "property_filters": [{{"field": "area_px", "operator": ">", "value": 500}}]
}}

Return ONLY the JSON object, no explanations."""

        user_message = f'Query: "{query}"\n\nGenerate the structured filter JSON:'

        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        response_text = message.content[0].text

        # Extract JSON from response
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        structured_query = json.loads(response_text)
        return structured_query

    def _execute_query(self, structured_query: Dict) -> List[Dict]:
        """
        Execute a structured query on particle data.

        Args:
            structured_query: Parsed query from LLM

        Returns:
            List of matching particles
        """
        # Determine which particle pool to search
        query_type = structured_query.get('type', 'both')

        candidates = []
        if query_type in ('individual', 'both'):
            candidates.extend(self.particles_data['individual_particles'])
        if query_type in ('union', 'both'):
            candidates.extend(self.particles_data['union_particles'])

        # Apply element filters
        element_filters = structured_query.get('element_filters', {})
        if element_filters:
            candidates = self._filter_by_elements(candidates, element_filters)

        # Apply property filters
        property_filters = structured_query.get('property_filters', [])
        for prop_filter in property_filters:
            candidates = self._filter_by_property(candidates, prop_filter)

        # Apply spatial filters
        spatial_filters = structured_query.get('spatial_filters', [])
        for spatial_filter in spatial_filters:
            candidates = self._filter_by_spatial(candidates, spatial_filter)

        # Sort results
        sort_spec = structured_query.get('sort')
        if sort_spec:
            field = sort_spec['field']
            reverse = (sort_spec.get('order', 'asc') == 'desc')
            candidates = sorted(candidates,
                              key=lambda p: p.get(field, 0),
                              reverse=reverse)

        # Limit results
        limit = structured_query.get('limit')
        if limit:
            candidates = candidates[:limit]

        return candidates

    def _filter_by_elements(self, particles: List[Dict], filters: Dict) -> List[Dict]:
        """Filter particles by element presence/absence."""
        required = filters.get('required', [])
        forbidden = filters.get('forbidden', [])
        any_of = filters.get('any_of', [])

        filtered = []
        for particle in particles:
            # Get particle elements
            if particle['type'] == 'individual':
                particle_elements = [particle['element']]
            else:
                particle_elements = particle.get('elements', [])

            # Check required elements
            if required and not all(e in particle_elements for e in required):
                continue

            # Check forbidden elements
            if forbidden and any(e in particle_elements for e in forbidden):
                continue

            # Check any_of
            if any_of and not any(e in particle_elements for e in any_of):
                continue

            filtered.append(particle)

        return filtered

    def _filter_by_property(self, particles: List[Dict], prop_filter: Dict) -> List[Dict]:
        """Filter particles by property comparison."""
        field = prop_filter['field']
        operator = prop_filter['operator']
        value = prop_filter['value']

        filtered = []
        for particle in particles:
            particle_value = particle.get(field)
            if particle_value is None:
                continue

            # Apply operator
            if operator == '>' and particle_value > value:
                filtered.append(particle)
            elif operator == '>=' and particle_value >= value:
                filtered.append(particle)
            elif operator == '<' and particle_value < value:
                filtered.append(particle)
            elif operator == '<=' and particle_value <= value:
                filtered.append(particle)
            elif operator == '==' and particle_value == value:
                filtered.append(particle)
            elif operator == '!=' and particle_value != value:
                filtered.append(particle)

        return filtered

    def _filter_by_spatial(self, particles: List[Dict], spatial_filter: Dict) -> List[Dict]:
        """Filter particles by spatial location."""
        filtered = []

        # Handle region-based filters (quadrants)
        if 'region' in spatial_filter:
            region = spatial_filter['region']
            h, w = self.image_shape

            for particle in particles:
                x, y = particle['center_px']

                if region == 'upper_left' and x < w/2 and y < h/2:
                    filtered.append(particle)
                elif region == 'upper_right' and x >= w/2 and y < h/2:
                    filtered.append(particle)
                elif region == 'lower_left' and x < w/2 and y >= h/2:
                    filtered.append(particle)
                elif region == 'lower_right' and x >= w/2 and y >= h/2:
                    filtered.append(particle)
                elif region == 'center':
                    # Center 50%
                    if w/4 <= x <= 3*w/4 and h/4 <= y <= 3*h/4:
                        filtered.append(particle)

        # Handle explicit range filters
        elif 'x_range' in spatial_filter or 'y_range' in spatial_filter:
            x_min, x_max = spatial_filter.get('x_range', [0, self.image_shape[1]])
            y_min, y_max = spatial_filter.get('y_range', [0, self.image_shape[0]])

            for particle in particles:
                x, y = particle['center_px']
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    filtered.append(particle)

        return filtered if filtered else particles

    def _format_results(self, results: List[Dict], format_type: str) -> Any:
        """Format query results based on requested format."""
        if format_type == 'summary':
            return [{
                'id': p['id'],
                'type': p['type'],
                'element': p.get('element') or p.get('elements'),
                'center_px': p['center_px'],
                'area_px': p.get('area_px', 0),
                'max_intensity': p.get('max_intensity', 'N/A')
            } for p in results]

        elif format_type == 'coordinates':
            return [p['center_px'] for p in results]

        elif format_type == 'ids':
            return [p['id'] for p in results]

        elif format_type == 'full':
            return results

        else:
            raise ValueError(f"Unknown format type: {format_type}")

    def _parse_elements_from_group(self, group_name: str) -> List[str]:
        """Parse element symbols from group name like 'CaFeS'."""
        # Simple parsing: assumes 1-2 char element symbols
        elements = []
        i = 0
        while i < len(group_name):
            if i + 1 < len(group_name) and group_name[i:i+2] in self.element_colors:
                elements.append(group_name[i:i+2])
                i += 2
            elif group_name[i] in self.element_colors:
                elements.append(group_name[i])
                i += 1
            else:
                i += 1
        return elements

    def explain_query(self, natural_language_query: str) -> str:
        """
        Explain what a query will do without executing it.

        Args:
            natural_language_query: User's query

        Returns:
            Human-readable explanation
        """
        structured = self._llm_parse_query(natural_language_query)

        explanation = f"This query will:\n"

        # Type
        query_type = structured.get('type', 'both')
        if query_type == 'individual':
            explanation += "- Search individual (single-element) particles\n"
        elif query_type == 'union':
            explanation += "- Search union (multi-element overlap) particles\n"
        else:
            explanation += "- Search all particle types\n"

        # Elements
        elem_filters = structured.get('element_filters', {})
        if elem_filters.get('required'):
            explanation += f"- Require elements: {', '.join(elem_filters['required'])}\n"
        if elem_filters.get('forbidden'):
            explanation += f"- Exclude elements: {', '.join(elem_filters['forbidden'])}\n"
        if elem_filters.get('any_of'):
            explanation += f"- Has at least one of: {', '.join(elem_filters['any_of'])}\n"

        # Properties
        for prop in structured.get('property_filters', []):
            explanation += f"- Where {prop['field']} {prop['operator']} {prop['value']}\n"

        # Spatial
        for spatial in structured.get('spatial_filters', []):
            if 'region' in spatial:
                explanation += f"- Located in {spatial['region']} region\n"
            elif 'x_range' in spatial:
                explanation += f"- In region x={spatial['x_range']}, y={spatial['y_range']}\n"

        # Sort
        if 'sort' in structured:
            sort_field = structured['sort']['field']
            sort_order = structured['sort'].get('order', 'asc')
            explanation += f"- Sort by {sort_field} ({'descending' if sort_order == 'desc' else 'ascending'})\n"

        # Limit
        if 'limit' in structured:
            explanation += f"- Return top {structured['limit']} results\n"

        return explanation
