"""AI-powered JSON Maker widget for building AutoMap configs using natural language.

Accepts user requests in plain English and uses an LLM to generate or modify
JSON configurations. Complements the manual JSON Maker with a conversational interface.
"""

import os
import json
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QMessageBox, QFileDialog, QComboBox,
    QSplitter, QLineEdit, QCheckBox,
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont


# ---------------------------------------------------------------------------
# Background thread for LLM API calls
# ---------------------------------------------------------------------------

class LLMThread(QThread):
    """Calls the LLM API in a background thread to avoid blocking the UI."""

    finished = Signal(str)  # Generated JSON text
    error = Signal(str)  # Error message

    def __init__(self, prompt, context_json, api_key, model, parent=None):
        super().__init__(parent)
        self._prompt = prompt
        self._context_json = context_json
        self._api_key = api_key
        self._model = model

    def run(self):
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)

            # Build the system prompt
            system_prompt = """You are an expert at generating JSON configuration files for X-ray fluorescence (XRF) microscopy experiments at the HXN beamline.

Your task is to generate or modify JSON configurations based on user requests. The JSON files control:
- Scan parameters (motor positions, step sizes, exposure times)
- Detection methods (simple OpenCV, cellpose, watershed, stardist, yolo, etc.)
- Analysis parameters (thresholds, morphology operations)
- Export settings (data paths, element lists)

When given a user request:
1. If context JSON is provided, start with that configuration and modify it
2. Otherwise, create a new configuration from defaults
3. Output ONLY valid JSON, no explanations or markdown
4. Ensure all required fields are present
5. Use sensible defaults for any unspecified parameters

Key parameter types:
- Detection methods: simple, cellpose, watershed, connected_components, yolo, stardist, hough, contours
- Execution modes: real, simulation, offline, analysis-only
- Coordinates are in microns (µm) or pixels
- Thresholds are typically 0-255 for intensity, pixels² for area"""

            # Build the user message
            if self._context_json:
                user_message = f"""Current configuration:
```json
{self._context_json}
```

User request: {self._prompt}

Generate the updated JSON configuration."""
            else:
                user_message = f"""User request: {self._prompt}

Generate a complete JSON configuration for this request."""

            # Call Claude API
            message = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # Extract the response
            response_text = message.content[0].text

            # Try to extract JSON if wrapped in markdown
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()

            # Validate it's proper JSON
            json.loads(response_text)

            self.finished.emit(response_text)

        except ImportError:
            self.error.emit(
                "The 'anthropic' package is not installed.\n\n"
                "Install it with: pip install anthropic"
            )
        except json.JSONDecodeError as err:
            self.error.emit(
                f"LLM did not return valid JSON:\n{err}\n\n"
                f"Response was:\n{response_text[:500]}"
            )
        except Exception as err:
            self.error.emit(f"Error calling LLM API:\n{str(err)}")


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class AIJSONMakerWidget(QWidget):
    """AI-powered JSON configuration builder using natural language requests.

    Allows users to describe what they want in plain English rather than
    manually filling out form fields. Can start from scratch or modify
    existing configurations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._llm_thread = None
        self._setup_ui()
        self._load_example_prompts()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === HEADER ===
        header = QLabel("<b>AI JSON Configuration Builder</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        info = QLabel(
            "Describe your scan configuration in plain English. "
            "The AI will generate or modify the JSON for you."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; padding: 5px; font-size: 11px;")
        layout.addWidget(info)

        # === API CONFIGURATION ===
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(8)

        # API Key row
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter your Anthropic API key or set ANTHROPIC_API_KEY env var")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_key_input.setText(api_key)
        key_row.addWidget(self.api_key_input, 1)

        self.show_key_checkbox = QCheckBox("Show")
        self.show_key_checkbox.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.show_key_checkbox)
        api_layout.addLayout(key_row)

        # Model selection row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "claude-sonnet-4-5",
            "claude-opus-5",
            "claude-haiku-4-5-20251001",
        ])
        self.model_combo.setCurrentText("claude-sonnet-4-5")
        model_row.addWidget(self.model_combo, 1)
        model_row.addStretch()
        api_layout.addLayout(model_row)

        layout.addWidget(api_group)

        # === CONTEXT SELECTION ===
        context_group = QGroupBox("Starting Configuration (Optional)")
        context_layout = QVBoxLayout(context_group)

        context_row = QHBoxLayout()
        self.context_label = QLabel("No file loaded - will create from scratch")
        self.context_label.setStyleSheet("font-style: italic; color: #666;")
        context_row.addWidget(self.context_label, 1)

        load_btn = QPushButton("Load JSON")
        load_btn.clicked.connect(self._load_context_json)
        context_row.addWidget(load_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_context)
        context_row.addWidget(clear_btn)

        context_layout.addLayout(context_row)
        layout.addWidget(context_group)

        # === EXAMPLE PROMPTS ===
        examples_row = QHBoxLayout()
        examples_row.addWidget(QLabel("Examples:"))
        self.examples_combo = QComboBox()
        self.examples_combo.addItem("-- Select an example prompt --", "")
        self.examples_combo.currentIndexChanged.connect(self._on_example_selected)
        examples_row.addWidget(self.examples_combo, 1)
        layout.addLayout(examples_row)

        # === INPUT/OUTPUT SPLIT ===
        splitter = QSplitter(Qt.Horizontal)

        # Left side: User input
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)

        input_label = QLabel("<b>Your Request:</b>")
        input_layout.addWidget(input_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText(
            "Describe what you want:\n\n"
            "Examples:\n"
            "- Create a simple blob detection scan with 0.25 µm step size\n"
            "- Change the detection method to cellpose with diameter 30\n"
            "- Set up a calcium and iron scan from -10 to 10 µm\n"
            "- Use stardist detection with 0.5 probability threshold"
        )
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.prompt_input.setFont(font)
        input_layout.addWidget(self.prompt_input, 1)

        # Generate button
        self.generate_btn = QPushButton("🤖 Generate Configuration")
        self.generate_btn.setStyleSheet("padding: 10px; font-weight: bold; font-size: 12px;")
        self.generate_btn.clicked.connect(self._generate_config)
        input_layout.addWidget(self.generate_btn)

        splitter.addWidget(input_widget)

        # Right side: Generated output
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("<b>Generated Configuration:</b>")
        output_layout.addWidget(output_label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(font)
        self.output_text.setPlaceholderText("Generated JSON will appear here...")
        output_layout.addWidget(self.output_text, 1)

        # Output buttons
        output_buttons = QHBoxLayout()

        self.validate_btn = QPushButton("✓ Validate JSON")
        self.validate_btn.clicked.connect(self._validate_json)
        self.validate_btn.setEnabled(False)
        output_buttons.addWidget(self.validate_btn)

        self.save_btn = QPushButton("💾 Save Configuration")
        self.save_btn.clicked.connect(self._save_config)
        self.save_btn.setEnabled(False)
        output_buttons.addWidget(self.save_btn)

        output_buttons.addStretch()
        output_layout.addLayout(output_buttons)

        splitter.addWidget(output_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # === STATUS BAR ===
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"
        )
        layout.addWidget(self.status_label)

        # Store context
        self._context_json = None
        self._context_path = None

    def _load_example_prompts(self):
        """Populate the examples dropdown with common use cases."""
        examples = [
            ("Simple blob detection scan",
             "Create a simple blob detection configuration with step size 0.25 µm, "
             "scanning from -12.5 to 12.5 µm in both X and Y, with 0.005s exposure time."),

            ("Cellpose cell detection",
             "Set up a cellpose-based detection with diameter 30 pixels, model type 'cyto3', "
             "and scan the sample with 0.1 µm steps."),

            ("Multi-element XRF scan",
             "Create a scan for detecting calcium, iron, and sulfur with simple blob detection. "
             "Use step size 0.2 µm and scan a 20x20 µm area."),

            ("Change to stardist detection",
             "Switch the detection method to stardist with model '2D_versatile_fluo', "
             "probability threshold 0.5, and NMS threshold 0.4."),

            ("Adjust thresholds",
             "Set intensity threshold to 150 and minimum area to 300 pixels. "
             "Add sweep values [100, 150, 200] for threshold testing."),

            ("Mosaic scan setup",
             "Configure a mosaic scan with 100 µm coverage in X and Y, "
             "10% overlap, and 250 nm step size with 0.01s dwell time."),
        ]

        for title, prompt in examples:
            self.examples_combo.addItem(title, prompt)

    def _on_example_selected(self, index):
        """Load selected example prompt into the input field."""
        prompt = self.examples_combo.itemData(index)
        if prompt:
            self.prompt_input.setPlainText(prompt)

    def _toggle_key_visibility(self, checked):
        """Toggle API key visibility."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)

    def _load_context_json(self):
        """Load an existing JSON file as starting context."""
        configs_dir = Path(__file__).resolve().parents[2] / "configs"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load JSON Configuration",
            str(configs_dir),
            "JSON files (*.json)",
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                self._context_json = f.read()
                json.loads(self._context_json)  # Validate it's proper JSON

            self._context_path = file_path
            filename = Path(file_path).name
            self.context_label.setText(f"Loaded: {filename}")
            self.context_label.setStyleSheet("color: #0a0; font-weight: bold;")
            self._set_status(f"Loaded context from {filename}")

        except Exception as err:
            QMessageBox.critical(self, "Error Loading JSON", str(err))

    def _clear_context(self):
        """Clear the loaded context."""
        self._context_json = None
        self._context_path = None
        self.context_label.setText("No file loaded - will create from scratch")
        self.context_label.setStyleSheet("font-style: italic; color: #666;")
        self._set_status("Context cleared")

    def _generate_config(self):
        """Generate JSON config using the LLM."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "No Input", "Please enter a request describing your configuration.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(
                self,
                "No API Key",
                "Please enter your Anthropic API key or set the ANTHROPIC_API_KEY environment variable."
            )
            return

        model = self.model_combo.currentText()

        # Disable UI during generation
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ Generating...")
        self.output_text.clear()
        self._set_status("Calling LLM API...")

        # Start background thread
        self._llm_thread = LLMThread(prompt, self._context_json, api_key, model, self)
        self._llm_thread.finished.connect(self._on_generation_finished)
        self._llm_thread.error.connect(self._on_generation_error)
        self._llm_thread.start()

    def _on_generation_finished(self, json_text):
        """Handle successful JSON generation."""
        self.output_text.setPlainText(json_text)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🤖 Generate Configuration")
        self.validate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._set_status("Configuration generated successfully", success=True)

        # Auto-validate
        self._validate_json()

    def _on_generation_error(self, error_msg):
        """Handle generation errors."""
        self.output_text.setPlainText(f"ERROR:\n\n{error_msg}")
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🤖 Generate Configuration")
        self._set_status("Generation failed", success=False)
        QMessageBox.critical(self, "Generation Error", error_msg)

    def _validate_json(self):
        """Validate the generated JSON."""
        json_text = self.output_text.toPlainText().strip()
        if not json_text:
            return

        try:
            config = json.loads(json_text)

            # Check for required top-level sections
            required_sections = [
                "execution_params", "scan_params", "detection_methods",
                "segmentation_params", "export_params"
            ]
            missing = [s for s in required_sections if s not in config]

            if missing:
                msg = f"⚠️ Missing sections: {', '.join(missing)}"
                self._set_status(msg, success=False)
                QMessageBox.warning(self, "Incomplete Configuration", msg)
            else:
                self._set_status("✓ JSON is valid and complete", success=True)
                QMessageBox.information(
                    self,
                    "Valid JSON",
                    "The generated configuration is valid and contains all required sections."
                )

        except json.JSONDecodeError as err:
            self._set_status("Invalid JSON", success=False)
            QMessageBox.critical(self, "Invalid JSON", f"JSON validation failed:\n{err}")

    def _save_config(self):
        """Save the generated configuration to a file."""
        json_text = self.output_text.toPlainText().strip()
        if not json_text:
            return

        try:
            # Validate before saving
            json.loads(json_text)
        except json.JSONDecodeError as err:
            QMessageBox.critical(
                self,
                "Invalid JSON",
                f"Cannot save invalid JSON:\n{err}"
            )
            return

        configs_dir = Path(__file__).resolve().parents[2] / "configs"
        default_name = "ai_generated_config.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON Configuration",
            str(configs_dir / default_name),
            "JSON files (*.json)",
        )

        if not file_path:
            return

        if not file_path.lower().endswith(".json"):
            file_path += ".json"

        try:
            with open(file_path, "w") as f:
                f.write(json_text)
                f.write("\n")

            filename = Path(file_path).name
            self._set_status(f"Saved to {filename}", success=True)
            QMessageBox.information(
                self,
                "Configuration Saved",
                f"Saved configuration to:\n{file_path}"
            )

        except OSError as err:
            QMessageBox.critical(self, "Error Saving File", str(err))

    def _set_status(self, message, success=None):
        """Update the status bar with a message."""
        self.status_label.setText(message)

        if success is True:
            self.status_label.setStyleSheet(
                "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
                "border-radius: 3px; color: #155724;"
            )
        elif success is False:
            self.status_label.setStyleSheet(
                "padding: 5px; background: #f8d7da; border: 1px solid #f5c6cb; "
                "border-radius: 3px; color: #721c24;"
            )
        else:
            self.status_label.setStyleSheet(
                "padding: 5px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"
            )
