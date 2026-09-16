"""GUI widget for LLM vision-based particle finding - integrated with image view."""

import os
import json
import cv2
import numpy as np
import tifffile as tiff
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QGroupBox,
    QCheckBox, QLineEdit, QMessageBox, QFileDialog, QSpinBox,
    QDoubleSpinBox, QFormLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QSplitter, QComboBox
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor

from .llm_vision_particle_finder import LLMVisionParticleFinder
from .element_parser import extract_element_from_filename, validate_element_symbol


class LLMGraphicsView(QGraphicsView):
    """Graphics view for displaying images and detected particles."""

    def __init__(self, scene, parent=None):
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.particle_items = []
        self.highlight_items = []

    def wheelEvent(self, event):
        """Zoom with mouse wheel."""
        cursor_pos = event.pos()
        scene_pos = self.mapToScene(cursor_pos)
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
        mouse_centered = self.mapFromScene(scene_pos)
        delta = cursor_pos - mouse_centered
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mousePressEvent(self, event):
        """Pan with left mouse button."""
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def clear_particles(self):
        """Remove all particle circles from the view."""
        for item in self.particle_items:
            self.scene().removeItem(item)
        self.particle_items.clear()

    def draw_particles(self, particles, axis_bounds=None):
        """Draw circles around detected particles.

        Args:
            particles: List of particle dicts with 'center_px' in data coordinates
            axis_bounds: Dict with transformation info to convert data->figure pixels
        """
        self.clear_particles()

        for particle in particles:
            x_data, y_data = particle['center_px']
            size = particle.get('size_px', 20)
            radius = size / 2 + 5  # Add padding

            # Transform data coordinates to figure pixel coordinates
            if axis_bounds:
                x_px = axis_bounds['left'] + (x_data / axis_bounds['data_width']) * axis_bounds['width']
                y_px = axis_bounds['bottom'] + (y_data / axis_bounds['data_height']) * axis_bounds['height']
            else:
                # Fallback: assume data coords = pixel coords
                x_px, y_px = x_data, y_data

            # Draw circle
            circle = QGraphicsEllipseItem(
                x_px - radius, y_px - radius,
                radius * 2, radius * 2
            )
            circle.setPen(QPen(QColor("cyan"), 2, Qt.SolidLine))
            circle.setZValue(100)
            self.scene().addItem(circle)
            self.particle_items.append(circle)

    def highlight_selected_particles(self, selected_items, axis_bounds=None):
        """Highlight selected particles with yellow circles.

        Args:
            selected_items: List of selected QListWidgetItems
            axis_bounds: Dict with transformation info to convert data->figure pixels
        """
        for item in self.highlight_items:
            self.scene().removeItem(item)
        self.highlight_items.clear()

        for list_item in selected_items:
            particle = list_item.data(Qt.UserRole)
            if particle:
                x_data, y_data = particle['center_px']
                size = particle.get('size_px', 20)
                radius = size / 2 + 10  # Larger radius for highlight

                # Transform data coordinates to figure pixel coordinates
                if axis_bounds:
                    x_px = axis_bounds['left'] + (x_data / axis_bounds['data_width']) * axis_bounds['width']
                    y_px = axis_bounds['bottom'] + (y_data / axis_bounds['data_height']) * axis_bounds['height']
                else:
                    # Fallback: assume data coords = pixel coords
                    x_px, y_px = x_data, y_data

                highlight = QGraphicsEllipseItem(
                    x_px - radius, y_px - radius,
                    radius * 2, radius * 2
                )
                highlight.setPen(QPen(QColor("yellow"), 3, Qt.SolidLine))
                highlight.setZValue(150)
                self.scene().addItem(highlight)
                self.highlight_items.append(highlight)


class VisionQueryThread(QThread):
    """Background thread for LLM vision query execution."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, finder, query_text, max_results, parent=None):
        super().__init__(parent)
        self.finder = finder
        self.query_text = query_text
        self.max_results = max_results

    def run(self):
        try:
            results = self.finder.find_particles(self.query_text, self.max_results)
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class LLMVisionWidget(QWidget):
    """Widget for LLM vision-based particle querying with integrated image view."""

    # Signal emitted when particles are selected for scanning
    particles_selected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.finder = LLMVisionParticleFinder()
        self.current_results = None
        self._query_thread = None
        self._tiff_paths = []
        self._element_names = []
        self.graphics_scene = None
        self.graphics_view = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === HEADER ===
        header = QLabel("<b>LLM Vision Particle Finder</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        # === MAIN CONTENT: Split between image view and controls ===
        splitter = QSplitter(Qt.Horizontal)

        # Left side: Image view
        image_panel = self._create_image_panel()
        splitter.addWidget(image_panel)

        # Right side: Controls
        controls_panel = self._create_controls_panel()
        splitter.addWidget(controls_panel)

        splitter.setStretchFactor(0, 2)  # Image takes 2/3
        splitter.setStretchFactor(1, 1)  # Controls takes 1/3

        layout.addWidget(splitter, 1)

        # === STATUS ===
        self.status_label = QLabel("Ready - Load TIFF files to begin")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"
        )
        layout.addWidget(self.status_label)

    def _create_image_panel(self):
        """Create the left panel with image display."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label and element mapping display
        info = QLabel("Load TIFF images below to see them here")
        info.setStyleSheet("color: #666; padding: 5px; font-size: 11px;")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # Element-to-color mapping label (hidden until images loaded)
        self.element_mapping_label = QLabel("")
        self.element_mapping_label.setStyleSheet(
            "background: #f8f9fa; padding: 8px; border: 1px solid #dee2e6; "
            "border-radius: 4px; font-family: monospace; font-size: 11px;"
        )
        self.element_mapping_label.setAlignment(Qt.AlignCenter)
        self.element_mapping_label.hide()
        layout.addWidget(self.element_mapping_label)

        # Graphics view
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = LLMGraphicsView(self.graphics_scene)
        layout.addWidget(self.graphics_view, 1)

        # View controls
        view_controls = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(lambda: self.graphics_view.resetTransform())
        view_controls.addWidget(reset_btn)

        clear_circles_btn = QPushButton("Clear Circles")
        clear_circles_btn.clicked.connect(lambda: self.graphics_view.clear_particles())
        view_controls.addWidget(clear_circles_btn)

        view_controls.addStretch()
        layout.addLayout(view_controls)

        return panel

    def _create_controls_panel(self):
        """Create the right panel with all controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        # === TIFF LOADING ===
        loading_group = QGroupBox("1. Load TIFF Images")
        loading_layout = QVBoxLayout(loading_group)

        load_btn_row = QHBoxLayout()
        self.load_tiffs_btn = QPushButton("📁 Select TIFF Files (1-3)")
        self.load_tiffs_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.load_tiffs_btn.clicked.connect(self._load_tiffs)
        load_btn_row.addWidget(self.load_tiffs_btn)
        loading_layout.addLayout(load_btn_row)

        self.tiff_list_label = QLabel("No files loaded")
        self.tiff_list_label.setStyleSheet("color: #888; font-style: italic; margin-left: 10px;")
        loading_layout.addWidget(self.tiff_list_label)

        # Calibration
        calib_form = QFormLayout()
        self.microns_per_px_x = QDoubleSpinBox()
        self.microns_per_px_x.setRange(0.001, 100.0)
        self.microns_per_px_x.setDecimals(4)
        self.microns_per_px_x.setValue(1.0)
        self.microns_per_px_x.setSuffix(" µm/px")
        calib_form.addRow("X Calibration:", self.microns_per_px_x)

        self.microns_per_px_y = QDoubleSpinBox()
        self.microns_per_px_y.setRange(0.001, 100.0)
        self.microns_per_px_y.setDecimals(4)
        self.microns_per_px_y.setValue(1.0)
        self.microns_per_px_y.setSuffix(" µm/px")
        calib_form.addRow("Y Calibration:", self.microns_per_px_y)

        loading_layout.addLayout(calib_form)
        layout.addWidget(loading_group)

        # === API CONFIGURATION ===
        api_group = QGroupBox("2. API Configuration")
        api_layout = QVBoxLayout(api_group)

        # Provider selection row
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Claude (Anthropic)", "OpenAI (ChatGPT)"])
        self.provider_combo.currentTextChanged.connect(self._update_api_placeholders)
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        api_layout.addLayout(provider_row)

        # API Key row
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Set AIFAPIM_API_KEY or ANTHROPIC_API_KEY env var")
        # Try institutional key first, then provider-specific key
        api_key = os.environ.get("AIFAPIM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_key_input.setText(api_key)
        key_row.addWidget(self.api_key_input, 1)

        self.show_key_checkbox = QCheckBox("Show")
        self.show_key_checkbox.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.show_key_checkbox)
        api_layout.addLayout(key_row)

        # Endpoint row
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(QLabel("Endpoint:"))
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("Default: api.anthropic.com (or set ANTHROPIC_BASE_URL)")
        endpoint = os.environ.get("ANTHROPIC_BASE_URL", "")
        self.endpoint_input.setText(endpoint)
        endpoint_row.addWidget(self.endpoint_input, 1)
        api_layout.addLayout(endpoint_row)

        # Model row
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Default: claude-sonnet-4-6")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.model_input.setText(model)
        model_row.addWidget(self.model_input, 1)
        api_layout.addLayout(model_row)

        # Info label (will be updated based on provider)
        self.info_label = QLabel("💡 For BNL: endpoint=https://hermes.nsls2.bnl.gov/anthropic, model=claude-sonnet-4-6")
        self.info_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        self.info_label.setWordWrap(True)
        api_layout.addWidget(self.info_label)

        layout.addWidget(api_group)

        # === QUERY INPUT ===
        query_group = QGroupBox("3. Query Particles")
        query_layout = QVBoxLayout(query_group)

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText(
            "Examples:\n"
            "- Find the 5 brightest calcium spots\n"
            "- Show me iron and sulfur overlapping\n"
            "- Locate small bright particles\n"
            "- 10 most prominent particles"
        )
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.query_input.setFont(font)
        self.query_input.setMaximumHeight(80)
        query_layout.addWidget(self.query_input)

        # Max results
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max results:"))
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 100)
        self.max_results_spin.setValue(10)
        max_row.addWidget(self.max_results_spin)
        max_row.addStretch()
        query_layout.addLayout(max_row)

        # Query button
        self.execute_btn = QPushButton("🔍 Find Particles")
        self.execute_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.execute_btn.setEnabled(False)
        self.execute_btn.clicked.connect(self._execute_query)
        query_layout.addWidget(self.execute_btn)

        layout.addWidget(query_group)

        # === RESULTS ===
        results_group = QGroupBox("4. Results")
        results_layout = QVBoxLayout(results_group)

        self.results_summary = QLabel("No query executed yet")
        self.results_summary.setStyleSheet("font-style: italic; color: #666;")
        results_layout.addWidget(self.results_summary)

        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.results_list.itemSelectionChanged.connect(self._on_selection_changed)
        results_layout.addWidget(self.results_list, 1)

        # Results actions
        results_actions = QVBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.results_list.selectAll)
        self.select_all_btn.setEnabled(False)
        results_actions.addWidget(self.select_all_btn)

        self.export_coords_btn = QPushButton("📋 Export Coordinates")
        self.export_coords_btn.clicked.connect(self._export_coordinates)
        self.export_coords_btn.setEnabled(False)
        results_actions.addWidget(self.export_coords_btn)

        self.send_to_queue_btn = QPushButton("📤 Send Selected to Queue")
        self.send_to_queue_btn.setStyleSheet("font-weight: bold;")
        self.send_to_queue_btn.clicked.connect(self._send_to_queue)
        self.send_to_queue_btn.setEnabled(False)
        results_actions.addWidget(self.send_to_queue_btn)

        results_layout.addLayout(results_actions)

        layout.addWidget(results_group, 1)

        return panel

    def _toggle_key_visibility(self, checked):
        """Toggle API key visibility."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)

    def _update_api_placeholders(self, provider_text):
        """Update placeholders and info based on selected provider."""
        if "Claude" in provider_text:
            # Claude/Anthropic configuration
            self.api_key_input.setPlaceholderText("Set AIFAPIM_API_KEY or ANTHROPIC_API_KEY env var")
            self.endpoint_input.setPlaceholderText("Default: api.anthropic.com (or set ANTHROPIC_BASE_URL)")
            self.model_input.setPlaceholderText("Default: claude-sonnet-4-6")
            self.info_label.setText("💡 For BNL: endpoint=https://hermes.nsls2.bnl.gov/anthropic, model=claude-sonnet-4-6")

            # Load Claude-specific env vars
            api_key = os.environ.get("AIFAPIM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
            endpoint = os.environ.get("ANTHROPIC_BASE_URL", "")
            model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        else:
            # OpenAI configuration
            self.api_key_input.setPlaceholderText("Set AIFAPIM_API_KEY or OPENAI_API_KEY env var")
            self.endpoint_input.setPlaceholderText("Default: https://hermes.nsls2.bnl.gov/openai/v1")
            self.model_input.setPlaceholderText("Default: gpt-5.6-luna (BNL) or gpt-4-vision-preview (standard)")
            self.info_label.setText("💡 For BNL: endpoint=https://hermes.nsls2.bnl.gov/openai/v1, models=gpt-5.6-luna, gpt-4.1-mini, gpt-5.4")

            # Load OpenAI-specific env vars
            api_key = os.environ.get("AIFAPIM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
            endpoint = os.environ.get("OPENAI_BASE_URL", "https://hermes.nsls2.bnl.gov/openai/v1")
            model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

        # Update inputs only if they're currently empty or match the old defaults
        if not self.api_key_input.text():
            self.api_key_input.setText(api_key)
        if not self.endpoint_input.text() or self.endpoint_input.text() in ["", "https://hermes.nsls2.bnl.gov/anthropic", "https://hermes.nsls2.bnl.gov/openai/v1", "https://hermes.nsls2.bnl.gov/openai/v1/"]:
            self.endpoint_input.setText(endpoint)
        if not self.model_input.text() or self.model_input.text() in ["claude-sonnet-4-6", "gpt-4-vision-preview", "gpt-4.1-mini", "gpt-5.6-luna"]:
            self.model_input.setText(model)

    def _load_tiffs(self):
        """Load TIFF files for analysis and display them."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select TIFF Files (1-3)", "",
            "TIFF files (*.tif *.tiff)"
        )

        if not file_paths:
            return

        if len(file_paths) > 3:
            QMessageBox.warning(
                self, "Too Many Files",
                "Please select 1-3 TIFF files maximum."
            )
            return

        self._tiff_paths = file_paths

        # Auto-detect element names from filenames
        self._element_names = []
        for path in file_paths:
            element = extract_element_from_filename(path)
            self._element_names.append(element)

        # Validate and warn about unrecognized elements
        invalid_elements = [elem for elem in self._element_names
                           if not validate_element_symbol(elem)]
        if invalid_elements:
            QMessageBox.warning(
                self, "Element Detection Warning",
                f"Could not recognize these as valid element symbols: {', '.join(invalid_elements)}\n\n"
                f"Detected elements: {', '.join(self._element_names)}\n\n"
                "Please verify the filenames follow the pattern:\n"
                "  • detsum_Ca_K.tiff\n"
                "  • Ca_K.tiff\n"
                "  • Ca.tiff"
            )

        # Update UI
        file_list = "\n".join([f"• {elem}: {Path(p).name}"
                               for elem, p in zip(self._element_names, file_paths)])
        self.tiff_list_label.setText(file_list)
        self.tiff_list_label.setStyleSheet("color: #000; margin-left: 10px;")

        # Load and display images
        try:
            # Initialize finder first to create the composite figure
            api_key = self.api_key_input.text().strip()
            endpoint = self.endpoint_input.text().strip()
            model = self.model_input.text().strip()
            provider = "claude" if "Claude" in self.provider_combo.currentText() else "openai"

            # Set default model if empty
            if not model:
                model = "claude-sonnet-4-6" if provider == "claude" else "gpt-5.6-luna"

            self.finder = LLMVisionParticleFinder(
                api_key=api_key if api_key else None,
                base_url=endpoint if endpoint else None,
                model=model,
                provider=provider
            )

            # Load TIFFs into finder - this will create the matplotlib composite
            self.finder.load_tiff_images(
                self._tiff_paths,
                self._element_names,
                self.microns_per_px_x.value(),
                self.microns_per_px_y.value()
            )

            # Force create the composite to get the matplotlib figure
            element_names = list(self._element_names)
            _ = self.finder._create_composite_image(element_names)

            # Load the saved matplotlib composite for display
            if hasattr(self.finder, '_last_composite_path') and os.path.exists(self.finder._last_composite_path):
                # Load the matplotlib figure
                composite_img = cv2.imread(self.finder._last_composite_path)
                composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)

                # Display in graphics view
                height, width = composite_img.shape[:2]
                q_img = QImage(composite_img.data, width, height, width * 3, QImage.Format_RGB888)

                # Clear old scene
                self.graphics_scene.clear()
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
                self.graphics_scene.addItem(pixmap_item)
                self.graphics_view.resetTransform()
            else:
                raise ValueError("Failed to create composite image")

            # Show element-to-color mapping
            color_names = ["RED", "GREEN", "BLUE"]
            mapping_text = "RGB Mapping: "
            mapping_parts = []
            for i, elem in enumerate(self._element_names[:3]):
                color = color_names[i]
                mapping_parts.append(f"<span style='color: {color.lower()}; font-weight: bold;'>{color}</span> = {elem}")
            mapping_text += " | ".join(mapping_parts)

            self.element_mapping_label.setText(mapping_text)
            self.element_mapping_label.show()

            # Finder already initialized above
            self.execute_btn.setEnabled(True)

            self.status_label.setText(
                f"✓ Loaded {len(file_paths)} TIFF file(s) - Ready to query"
            )
            self.status_label.setStyleSheet(
                "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
                "border-radius: 3px; color: #155724;"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIFF files:\n{str(e)}")

    def _execute_query(self):
        """Execute LLM vision query."""
        query_text = self.query_input.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "No Query", "Please enter a query.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(
                self, "No API Key",
                "Please enter your Anthropic API key or set ANTHROPIC_API_KEY."
            )
            return

        if not self._tiff_paths:
            QMessageBox.warning(self, "No Images", "Load TIFF files first.")
            return

        # Update configuration
        endpoint = self.endpoint_input.text().strip()
        model = self.model_input.text().strip() or "claude-sonnet-4-6"

        self.finder.api_key = api_key
        self.finder.base_url = endpoint if endpoint else None
        self.finder.model = model
        self.finder.microns_per_pixel_x = self.microns_per_px_x.value()
        self.finder.microns_per_pixel_y = self.microns_per_px_y.value()

        # Disable UI during query
        self.execute_btn.setEnabled(False)
        self.execute_btn.setText("⏳ Analyzing...")
        self.status_label.setText("LLM analyzing images...")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 3px;"
        )

        # Run in background
        self._query_thread = VisionQueryThread(
            self.finder,
            query_text,
            self.max_results_spin.value(),
            parent=self
        )
        self._query_thread.finished.connect(self._on_query_finished)
        self._query_thread.error.connect(self._on_query_error)
        self._query_thread.start()

    def _on_query_finished(self, results):
        """Handle successful query completion."""
        self.current_results = results

        # Re-enable UI
        self.execute_btn.setEnabled(True)
        self.execute_btn.setText("🔍 Find Particles")

        count = results['count']
        provider = results.get('provider', 'unknown')
        model = results.get('model', 'unknown')
        self.results_summary.setText(f"Found {count} particles ({provider}: {model})")
        self.results_summary.setStyleSheet("font-weight: bold; color: #155724;")

        # Redraw matplotlib figure with circles, then reload into GUI
        if hasattr(self.finder, 'redraw_composite_with_circles'):
            self.finder.redraw_composite_with_circles(results['particles'])

            # Reload the updated figure
            if hasattr(self.finder, '_last_composite_path') and os.path.exists(self.finder._last_composite_path):
                composite_img = cv2.imread(self.finder._last_composite_path)
                composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)

                height, width = composite_img.shape[:2]
                q_img = QImage(composite_img.data, width, height, width * 3, QImage.Format_RGB888)

                # Clear and reload
                self.graphics_scene.clear()
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
                self.graphics_scene.addItem(pixmap_item)
                self.graphics_view.resetTransform()
        else:
            # Fallback: draw circles with Qt (old method)
            axis_bounds = getattr(self.finder, '_axis_bounds_px', None)
            self.graphics_view.draw_particles(results['particles'], axis_bounds)

        # Populate results list
        self.results_list.clear()
        for particle in results['particles']:
            x_px, y_px = particle['center_px']
            x_um, y_um = particle.get('center_um', [0, 0])
            confidence = particle.get('confidence', 1.0)
            size = particle.get('size_px', '?')
            desc = particle.get('description', 'No description')

            item_text = (
                f"#{particle.get('id', '?')}: ({x_px:.0f}, {y_px:.0f}) px = "
                f"({x_um:.2f}, {y_um:.2f}) µm | "
                f"size={size}px | conf={confidence:.2f}\n"
                f"  → {desc}"
            )

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, particle)
            self.results_list.addItem(item)

        # Enable action buttons
        self.select_all_btn.setEnabled(count > 0)
        self.export_coords_btn.setEnabled(count > 0)
        self.send_to_queue_btn.setEnabled(count > 0)

        self.status_label.setText(f"✓ Found {count} particles - circles drawn on image")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
            "border-radius: 3px; color: #155724;"
        )

    def _on_query_error(self, error_msg):
        """Handle query errors."""
        self.execute_btn.setEnabled(True)
        self.execute_btn.setText("🔍 Find Particles")
        self.status_label.setText("Query failed")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #f8d7da; border: 1px solid #f5c6cb; "
            "border-radius: 3px; color: #721c24;"
        )
        QMessageBox.critical(self, "Query Error", f"Vision query failed:\n\n{error_msg}")

    def _on_selection_changed(self):
        """Highlight selected particles on the image."""
        selected_items = self.results_list.selectedItems()

        if hasattr(self, 'finder') and hasattr(self.finder, 'highlight_particles'):
            # Get selected particles
            selected_particles = [item.data(Qt.UserRole) for item in selected_items if item.data(Qt.UserRole)]

            # Redraw with matplotlib
            self.finder.highlight_particles(selected_particles)

            # Reload the updated figure
            if hasattr(self.finder, '_last_composite_path') and os.path.exists(self.finder._last_composite_path):
                composite_img = cv2.imread(self.finder._last_composite_path)
                composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)

                height, width = composite_img.shape[:2]
                q_img = QImage(composite_img.data, width, height, width * 3, QImage.Format_RGB888)

                self.graphics_scene.clear()
                pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
                self.graphics_scene.addItem(pixmap_item)
                self.graphics_view.resetTransform()
        else:
            # Fallback: Qt-based highlighting
            axis_bounds = getattr(self.finder, '_axis_bounds_px', None) if hasattr(self, 'finder') else None
            self.graphics_view.highlight_selected_particles(selected_items, axis_bounds)

    def _export_coordinates(self):
        """Export particle coordinates as JSON."""
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Select particles to export first.")
            return

        particles = [item.data(Qt.UserRole) for item in selected_items]

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Coordinates", "particle_coordinates.json",
            "JSON files (*.json)"
        )

        if file_path:
            try:
                # Get scan coordinates
                scan_coords = self.finder.get_scan_coordinates(particles)

                export_data = {
                    'query': self.current_results['query'],
                    'count': len(particles),
                    'calibration': self.current_results['calibration'],
                    'particles': particles,
                    'scan_coordinates': scan_coords
                }

                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {len(particles)} particle coordinates to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def _send_to_queue(self):
        """Send selected particles to scan queue."""
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Select particles to scan first.")
            return

        particles = [item.data(Qt.UserRole) for item in selected_items]

        # Get scan coordinates
        scan_coords = self.finder.get_scan_coordinates(particles)

        # Emit signal (parent should connect to this)
        self.particles_selected.emit(scan_coords)

        QMessageBox.information(
            self, "Sent to Queue",
            f"Sent {len(scan_coords)} particles to scan queue."
        )
