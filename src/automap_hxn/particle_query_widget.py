"""GUI widget for LLM-powered particle querying."""

import os
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QGroupBox, QSplitter,
    QCheckBox, QLineEdit, QComboBox, QMessageBox, QFileDialog,
    QProgressDialog
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont

from .particle_query_llm import ParticleQueryLLM
from .blobs.detection import detect_blobs
from .blobs.processing import find_union_blobs
from .utils import resize_if_needed, normalize_and_dilate


class QueryThread(QThread):
    """Background thread for LLM query execution."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, query_engine, query_text, return_format, parent=None):
        super().__init__(parent)
        self.query_engine = query_engine
        self.query_text = query_text
        self.return_format = return_format

    def run(self):
        try:
            results = self.query_engine.query(self.query_text, self.return_format)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DetectionThread(QThread):
    """Background thread for particle detection."""

    finished = Signal(dict, dict)  # blobs_by_color, union_blobs
    progress = Signal(int, int)  # current, total
    error = Signal(str)

    def __init__(self, image_paths, element_names, detection_config, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.element_names = element_names
        self.detection_config = detection_config

    def run(self):
        try:
            # Load images
            images = []
            for path in self.image_paths:
                img = tiff.imread(path).astype(np.float32)
                images.append(img)

            # Get target shape from most common shape
            from collections import Counter
            shapes = [img.shape for img in images]
            target_shape = Counter(shapes).most_common(1)[0][0]

            # Resize if needed
            for i, img in enumerate(images):
                images[i] = resize_if_needed(img, f"image_{i}", target_shape)

            # Normalize and dilate
            morph_params = self.detection_config.get('morphology_params', {})
            kernel_size = tuple(morph_params.get('normalize_kernel_size', [3, 3]))
            iterations = morph_params.get('dilate_iterations', 2)

            norm_dilated = [normalize_and_dilate(img, kernel_size=kernel_size, iterations=iterations)
                          for img in images]

            # Run detection
            colors = ['red', 'green', 'blue'][:len(images)]
            blobs_by_color = {}

            seg_params = self.detection_config.get('segmentation_params', {})
            method = seg_params.get('blob_detection_method', 'simple')
            method_params = self.detection_config.get('detection_methods', {}).get(method, {})

            min_threshold = method_params.get('min_threshold', 100)
            min_area = seg_params.get('min_threshold_area', method_params.get('min_size', 200))

            total = len(images)
            for i, (color, element) in enumerate(zip(colors, self.element_names)):
                self.progress.emit(i, total)

                dilated_img = norm_dilated[i][1]  # Get dilated image
                orig_img = images[i]

                blobs = detect_blobs(
                    dilated_img, orig_img,
                    min_threshold, min_area,
                    color, f"{element}.tif",
                    method=method,
                    **{k: v for k, v in method_params.items()
                       if k not in {'min_threshold', 'min_area', 'min_size'}}
                )
                blobs_by_color[color] = blobs

            self.progress.emit(total, total)

            # Find unions
            union_params = self.detection_config.get('union_params', {})
            union_blobs = find_union_blobs(
                blobs_by_color,
                element_names=self.element_names,
                colors=colors,
                image_shape=target_shape,
                **union_params
            )

            self.finished.emit(blobs_by_color, union_blobs)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")


class ParticleQueryWidget(QWidget):
    """Widget for natural language particle querying."""

    # Signal emitted when particles are selected
    particles_selected = Signal(list)  # List of particle IDs or coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.query_engine = ParticleQueryLLM()
        self.current_results = None
        self._query_thread = None
        self._detection_thread = None

        # Data loading state
        self.detection_config = None
        self.element_names = []
        self.tiff_paths = []
        self.loaded_blobs = None
        self.loaded_unions = None
        self.image_shape = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === HEADER ===
        header = QLabel("<b>Natural Language Particle Query</b>")
        header.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header)

        info = QLabel(
            "Load TIFF images → Run detection → Ask questions in plain English"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; padding: 5px; font-size: 11px;")
        layout.addWidget(info)

        # === DATA LOADING SECTION ===
        loading_group = QGroupBox("Data Loading")
        loading_layout = QVBoxLayout(loading_group)

        # Quick load: TIFF files directly
        quick_row = QHBoxLayout()
        self.load_tiffs_btn = QPushButton("📁 Load TIFF Files")
        self.load_tiffs_btn.setStyleSheet("font-weight: bold;")
        self.load_tiffs_btn.clicked.connect(self._load_tiffs_directly)
        quick_row.addWidget(self.load_tiffs_btn)
        quick_row.addWidget(QLabel("(Select 1-3 TIFF files)"))
        quick_row.addStretch()
        loading_layout.addLayout(quick_row)

        # OR divider
        or_label = QLabel("— OR —")
        or_label.setAlignment(Qt.AlignCenter)
        or_label.setStyleSheet("color: #888; margin: 5px 0;")
        loading_layout.addWidget(or_label)

        # Advanced: Config-based loading
        config_row = QHBoxLayout()
        self.load_config_btn = QPushButton("⚙️ Load JSON Config (Advanced)")
        self.load_config_btn.clicked.connect(self._load_config)
        config_row.addWidget(self.load_config_btn)
        self.config_path_label = QLabel("No config loaded")
        self.config_path_label.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        config_row.addWidget(self.config_path_label, 1)
        loading_layout.addLayout(config_row)

        # TIFF file selection (for config-based workflow)
        self.tiff_labels = []
        self.tiff_buttons = []
        self.tiff_element_inputs = []
        self.tiff_rows_layout = QVBoxLayout()
        loading_layout.addLayout(self.tiff_rows_layout)

        # Run detection button
        detect_row = QHBoxLayout()
        detect_row.addStretch()
        self.run_detection_btn = QPushButton("🔍 Run Detection")
        self.run_detection_btn.setEnabled(False)
        self.run_detection_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self.run_detection_btn.clicked.connect(self._run_detection)
        detect_row.addWidget(self.run_detection_btn)
        loading_layout.addLayout(detect_row)

        layout.addWidget(loading_group)

        # === API KEY SECTION ===
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout(api_group)

        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Anthropic API key or set ANTHROPIC_API_KEY env var")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.api_key_input.setText(api_key)
        api_row.addWidget(self.api_key_input, 1)

        self.show_key_checkbox = QCheckBox("Show")
        self.show_key_checkbox.toggled.connect(self._toggle_key_visibility)
        api_row.addWidget(self.show_key_checkbox)
        api_layout.addLayout(api_row)

        layout.addWidget(api_group)

        # === EXAMPLES ===
        examples_row = QHBoxLayout()
        examples_row.addWidget(QLabel("Examples:"))
        self.examples_combo = QComboBox()
        self._load_examples()
        self.examples_combo.currentIndexChanged.connect(self._on_example_selected)
        examples_row.addWidget(self.examples_combo, 1)
        layout.addLayout(examples_row)

        # === QUERY INPUT ===
        query_group = QGroupBox("Your Query")
        query_layout = QVBoxLayout(query_group)

        self.query_input = QTextEdit()
        self.query_input.setPlaceholderText(
            "Examples:\n"
            "- top 5 spots with Fe and S overlapping but no Ca\n"
            "- all calcium particles brighter than 150\n"
            "- 10 largest iron particles in the upper right\n"
            "- particles with high sulfur intensity\n"
            "- union boxes with calcium and iron, sorted by size"
        )
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.query_input.setFont(font)
        self.query_input.setMaximumHeight(100)
        query_layout.addWidget(self.query_input)

        # Query buttons
        query_buttons = QHBoxLayout()

        self.explain_btn = QPushButton("❓ Explain Query")
        self.explain_btn.clicked.connect(self._explain_query)
        query_buttons.addWidget(self.explain_btn)

        self.execute_btn = QPushButton("🔍 Execute Query")
        self.execute_btn.setStyleSheet("font-weight: bold;")
        self.execute_btn.clicked.connect(self._execute_query)
        query_buttons.addWidget(self.execute_btn)

        query_buttons.addStretch()
        query_layout.addLayout(query_buttons)

        layout.addWidget(query_group)

        # === RESULTS SECTION ===
        results_group = QGroupBox("Query Results")
        results_layout = QVBoxLayout(results_group)

        # Results summary
        self.results_summary = QLabel("No query executed yet")
        self.results_summary.setStyleSheet("font-style: italic; color: #666;")
        results_layout.addWidget(self.results_summary)

        # Results list
        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QListWidget.ExtendedSelection)
        results_layout.addWidget(self.results_list, 1)

        # Results actions
        results_actions = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.results_list.selectAll)
        self.select_all_btn.setEnabled(False)
        results_actions.addWidget(self.select_all_btn)

        self.highlight_btn = QPushButton("Highlight in Image")
        self.highlight_btn.clicked.connect(self._highlight_particles)
        self.highlight_btn.setEnabled(False)
        results_actions.addWidget(self.highlight_btn)

        self.export_btn = QPushButton("Export Selected")
        self.export_btn.clicked.connect(self._export_particles)
        self.export_btn.setEnabled(False)
        results_actions.addWidget(self.export_btn)

        results_actions.addStretch()
        results_layout.addLayout(results_actions)

        layout.addWidget(results_group, 1)

        # === STATUS BAR ===
        self.status_label = QLabel("Ready - Click 'Load TIFF Files' to begin")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 3px;"
        )
        layout.addWidget(self.status_label)

    def _load_examples(self):
        """Populate examples dropdown."""
        examples = [
            ("-- Select an example --", ""),
            ("Top 5: Fe+S without Ca", "top 5 spots with Fe and S overlapping but no Ca"),
            ("Bright calcium particles", "all calcium particles with intensity > 150"),
            ("Large iron in upper right", "10 largest iron particles in the upper right quadrant"),
            ("High sulfur intensity", "particles with sulfur and max intensity above 200"),
            ("Any Fe or S, area > 500", "particles with iron or sulfur, area greater than 500 pixels"),
            ("Center region, bright", "particles in the center region with mean intensity > 100"),
            ("Union boxes sorted by size", "all union boxes sorted by area, largest first"),
            ("Small calcium particles", "calcium particles smaller than 100 pixels area"),
            ("Top 20 brightest any element", "20 brightest particles of any element"),
        ]

        for title, query in examples:
            self.examples_combo.addItem(title, query)

    def _on_example_selected(self, index):
        """Load selected example into query input."""
        query = self.examples_combo.itemData(index)
        if query:
            self.query_input.setPlainText(query)

    def _toggle_key_visibility(self, checked):
        """Toggle API key visibility."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_input.setEchoMode(QLineEdit.Password)

    def load_particle_data(self, blobs_by_color, union_blobs, image_shape, element_mapping):
        """
        Load detection results for querying.

        Args:
            blobs_by_color: Dict of {color: [blob1, blob2, ...]}
            union_blobs: Dict of union detection results
            image_shape: (height, width)
            element_mapping: Dict of {element: color}
        """
        api_key = self.api_key_input.text().strip()
        if api_key:
            self.query_engine.api_key = api_key

        self.query_engine.load_detection_results(
            blobs_by_color, union_blobs, image_shape, element_mapping
        )

        n_individual = len(self.query_engine.particles_data['individual_particles'])
        n_union = len(self.query_engine.particles_data['union_particles'])

        self.status_label.setText(
            f"Loaded: {n_individual} individual particles, {n_union} union particles"
        )
        self.status_label.setStyleSheet(
            "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
            "border-radius: 3px; color: #155724;"
        )

        self.execute_btn.setEnabled(True)
        self.explain_btn.setEnabled(True)

    def _explain_query(self):
        """Explain what the query will do."""
        query_text = self.query_input.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "No Query", "Please enter a query first.")
            return

        if not self.query_engine.particles_data:
            QMessageBox.warning(self, "No Data", "Load particle data first.")
            return

        try:
            explanation = self.query_engine.explain_query(query_text)
            QMessageBox.information(self, "Query Explanation", explanation)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to explain query:\n{str(e)}")

    def _execute_query(self):
        """Execute the natural language query."""
        query_text = self.query_input.toPlainText().strip()
        if not query_text:
            QMessageBox.warning(self, "No Query", "Please enter a query.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(
                self, "No API Key",
                "Please enter your Anthropic API key or set ANTHROPIC_API_KEY environment variable."
            )
            return

        if not self.query_engine.particles_data:
            QMessageBox.warning(
                self, "No Data",
                "Load particle data first:\n\n"
                "Quick: Click 'Load TIFF Files' → Run Detection\n\n"
                "OR\n\n"
                "Advanced: Load JSON Config → Select TIFFs → Run Detection"
            )
            return

        # Update API key
        self.query_engine.api_key = api_key

        # Disable UI during query
        self.execute_btn.setEnabled(False)
        self.execute_btn.setText("⏳ Querying...")
        self.status_label.setText("Executing query...")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 3px;"
        )

        # Run query in background thread
        self._query_thread = QueryThread(
            self.query_engine, query_text, return_format='summary', parent=self
        )
        self._query_thread.finished.connect(self._on_query_finished)
        self._query_thread.error.connect(self._on_query_error)
        self._query_thread.start()

    def _on_query_finished(self, results):
        """Handle successful query completion."""
        self.current_results = results

        # Update UI
        self.execute_btn.setEnabled(True)
        self.execute_btn.setText("🔍 Execute Query")

        count = results['count']
        self.results_summary.setText(f"Found {count} matching particles")
        self.results_summary.setStyleSheet("font-weight: bold; color: #155724;")

        # Populate results list
        self.results_list.clear()
        for particle in results['results']:
            # Format list item
            p_type = particle['type']
            if p_type == 'individual':
                element = particle['element']
                intensity = particle.get('max_intensity', 'N/A')
                item_text = (
                    f"{particle['id']}: {element} at {particle['center_px']}, "
                    f"area={particle['area_px']:.0f}px², intensity={intensity}"
                )
            else:
                elements = '+'.join(particle['element'])
                item_text = (
                    f"{particle['id']}: {elements} union at {particle['center_px']}, "
                    f"area={particle['area_px']:.0f}px²"
                )

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, particle)  # Store full particle data
            self.results_list.addItem(item)

        # Enable action buttons
        self.select_all_btn.setEnabled(count > 0)
        self.highlight_btn.setEnabled(count > 0)
        self.export_btn.setEnabled(count > 0)

        self.status_label.setText(f"Query complete: {count} results")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
            "border-radius: 3px; color: #155724;"
        )

    def _on_query_error(self, error_msg):
        """Handle query errors."""
        self.execute_btn.setEnabled(True)
        self.execute_btn.setText("🔍 Execute Query")
        self.status_label.setText("Query failed")
        self.status_label.setStyleSheet(
            "padding: 5px; background: #f8d7da; border: 1px solid #f5c6cb; "
            "border-radius: 3px; color: #721c24;"
        )
        QMessageBox.critical(self, "Query Error", f"Query failed:\n{error_msg}")

    def _highlight_particles(self):
        """Emit signal to highlight selected particles in main view."""
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Select particles from the results first.")
            return

        # Extract particle data
        particles = [item.data(Qt.UserRole) for item in selected_items]

        # Emit signal (parent widget should connect to this)
        self.particles_selected.emit(particles)

        self.status_label.setText(f"Highlighted {len(particles)} particles")

    def _export_particles(self):
        """Export selected particles as JSON."""
        selected_items = self.results_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "Select particles to export first.")
            return

        particles = [item.data(Qt.UserRole) for item in selected_items]

        from qtpy.QtWidgets import QFileDialog
        import json

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Particles", "query_results.json", "JSON files (*.json)"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump({
                        'query': self.current_results['query'],
                        'count': len(particles),
                        'particles': particles
                    }, f, indent=2)

                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {len(particles)} particles to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")

    def _load_tiffs_directly(self):
        """Load TIFF files directly without config - simple workflow."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select TIFF Files (1-3 files)", "",
            "TIFF files (*.tif *.tiff)"
        )

        if not file_paths:
            return

        if len(file_paths) > 3:
            QMessageBox.warning(
                self, "Too Many Files",
                "Please select 1-3 TIFF files (max 3 for RGB channels)."
            )
            return

        # Create simple detection config with defaults
        self.detection_config = {
            "segmentation_params": {
                "blob_detection_method": "simple",
                "min_threshold_area": 200
            },
            "detection_methods": {
                "simple": {
                    "min_threshold": 100,
                    "max_threshold": 255,
                    "min_area": 200,
                    "filter_by_area": True,
                    "filter_by_circularity": False,
                    "filter_by_convexity": False,
                    "filter_by_inertia": False
                }
            },
            "morphology_params": {
                "normalize_kernel_size": [3, 3],
                "dilate_iterations": 2
            },
            "union_params": {
                "overlap_threshold": 0.3,
                "max_distance": 10
            }
        }

        # Set TIFF paths
        self.tiff_paths = file_paths

        # Auto-extract element names from filenames
        self.element_names = []
        for path in file_paths:
            filename = Path(path).stem
            # Try common patterns: "Ca_K", "calcium", "Fe_K", etc.
            if '_' in filename:
                element = filename.split('_')[0]
            else:
                element = filename.split('.')[0]
            # Capitalize first letter
            element = element[:1].upper() + element[1:] if len(element) > 1 else element.upper()
            self.element_names.append(element)

        # Create editable UI rows for elements
        self._create_tiff_rows_with_editing(file_paths, self.element_names)

        self.run_detection_btn.setEnabled(True)
        self.status_label.setText(
            f"Loaded {len(file_paths)} TIFF file(s) - Edit element names if needed, then click 'Run Detection'"
        )
        self.status_label.setStyleSheet(
            "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
            "border-radius: 3px; color: #155724;"
        )

    def _create_tiff_rows_with_editing(self, file_paths, element_names):
        """Create UI rows showing TIFF files with editable element names."""
        # Clear existing rows
        while self.tiff_rows_layout.count():
            item = self.tiff_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.tiff_element_inputs = []

        for i, (path, element) in enumerate(zip(file_paths, element_names)):
            row = QHBoxLayout()

            # Element name input
            element_input = QLineEdit(element)
            element_input.setMaximumWidth(80)
            element_input.setPlaceholderText("Element")
            self.tiff_element_inputs.append(element_input)
            row.addWidget(QLabel("Element:"))
            row.addWidget(element_input)

            # File path label
            label = QLabel(Path(path).name)
            label.setStyleSheet("color: #000;")
            row.addWidget(label, 1)

            self.tiff_rows_layout.addLayout(row)

    def _load_config(self):
        """Load detection configuration from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Detection Config", "", "JSON files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                self.detection_config = json.load(f)

            # Extract element names from config
            scan_params = self.detection_config.get('scan_params', {})
            self.element_names = scan_params.get('element_lines', [])

            if not self.element_names:
                QMessageBox.warning(
                    self, "Invalid Config",
                    "Config file must contain 'scan_params.element_lines' list."
                )
                return

            # Update UI
            self.config_path_label.setText(Path(file_path).name)
            self.config_path_label.setStyleSheet("color: #000; font-size: 10px;")

            # Create TIFF selection rows
            self._create_tiff_selection_rows()

            self.status_label.setText(f"Config loaded: {len(self.element_names)} elements expected - Select TIFF files")
            self.status_label.setStyleSheet(
                "padding: 5px; background: #d4edda; border: 1px solid #c3e6cb; "
                "border-radius: 3px; color: #155724;"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config:\n{str(e)}")

    def _create_tiff_selection_rows(self):
        """Create UI rows for selecting TIFF files for each element."""
        # Clear existing rows
        while self.tiff_rows_layout.count():
            item = self.tiff_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.tiff_labels = []
        self.tiff_buttons = []
        self.tiff_paths = [None] * len(self.element_names)

        for i, element in enumerate(self.element_names):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{element}:"))

            label = QLabel("No file selected")
            label.setStyleSheet("color: #888; font-style: italic;")
            self.tiff_labels.append(label)
            row.addWidget(label, 1)

            btn = QPushButton("Select TIFF")
            btn.clicked.connect(lambda checked, idx=i: self._select_tiff(idx))
            self.tiff_buttons.append(btn)
            row.addWidget(btn)

            self.tiff_rows_layout.addLayout(row)

    def _select_tiff(self, element_index):
        """Select TIFF file for a specific element."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select TIFF for {self.element_names[element_index]}",
            "", "TIFF files (*.tif *.tiff)"
        )

        if file_path:
            self.tiff_paths[element_index] = file_path
            self.tiff_labels[element_index].setText(Path(file_path).name)
            self.tiff_labels[element_index].setStyleSheet("color: #000;")

            # Enable run button if all files are selected
            if all(path is not None for path in self.tiff_paths):
                self.run_detection_btn.setEnabled(True)

    def _run_detection(self):
        """Run particle detection on loaded TIFF files."""
        if not self.detection_config:
            QMessageBox.warning(self, "No Config", "Load TIFF files or a config first.")
            return

        if not self.tiff_paths or not all(self.tiff_paths):
            QMessageBox.warning(self, "Missing Files", "Load TIFF files first.")
            return

        # Get element names from input fields if they exist (direct load workflow)
        if self.tiff_element_inputs:
            self.element_names = [inp.text().strip() for inp in self.tiff_element_inputs]
            # Validate element names
            if not all(self.element_names):
                QMessageBox.warning(self, "Invalid Elements", "Please enter element names for all files.")
                return

        # Disable UI during detection
        self.run_detection_btn.setEnabled(False)
        self.run_detection_btn.setText("⏳ Detecting...")

        # Show progress dialog
        progress = QProgressDialog("Running detection...", "Cancel", 0, len(self.tiff_paths), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)

        # Run detection in background thread
        self._detection_thread = DetectionThread(
            self.tiff_paths,
            self.element_names,
            self.detection_config,
            parent=self
        )
        self._detection_thread.progress.connect(lambda curr, total: progress.setValue(curr))
        self._detection_thread.finished.connect(lambda blobs, unions: self._on_detection_finished(blobs, unions, progress))
        self._detection_thread.error.connect(lambda err: self._on_detection_error(err, progress))
        self._detection_thread.start()

    def _on_detection_finished(self, blobs_by_color, union_blobs, progress):
        """Handle successful detection completion."""
        progress.close()

        self.run_detection_btn.setEnabled(True)
        self.run_detection_btn.setText("🔍 Run Detection")

        # Store results
        self.loaded_blobs = blobs_by_color
        self.loaded_unions = union_blobs

        # Get image shape
        try:
            first_img = tiff.imread(self.tiff_paths[0])
            self.image_shape = first_img.shape
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not read image shape: {e}")
            self.image_shape = (512, 512)  # Default fallback

        # Build element mapping
        colors = ['red', 'green', 'blue'][:len(self.element_names)]
        element_mapping = {elem: color for elem, color in zip(self.element_names, colors)}

        # Load into query engine
        self.load_particle_data(
            blobs_by_color=blobs_by_color,
            union_blobs=union_blobs,
            image_shape=self.image_shape,
            element_mapping=element_mapping
        )

        # Count particles
        n_individual = sum(len(blobs) for blobs in blobs_by_color.values())
        n_union = len(union_blobs)

        QMessageBox.information(
            self, "Detection Complete",
            f"Detected:\n- {n_individual} individual particles\n- {n_union} union particles\n\n"
            "You can now run queries!"
        )

    def _on_detection_error(self, error_msg, progress):
        """Handle detection errors."""
        progress.close()

        self.run_detection_btn.setEnabled(True)
        self.run_detection_btn.setText("🔍 Run Detection")

        QMessageBox.critical(self, "Detection Error", f"Detection failed:\n\n{error_msg}")
