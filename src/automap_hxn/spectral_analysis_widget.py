"""
Fe-Dy Spectral Analysis Widget for XRF interference investigation.

This widget allows loading H5 files, visualizing spectral features,
and interactively selecting regions for XANES analysis.
"""

import numpy as np
import h5py
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGroupBox, QSpinBox, QFormLayout,
    QSplitter, QScrollArea
)
from qtpy.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush
from qtpy.QtCore import Qt, QRectF

from .feRatio import (
    checkFeRatioWithDy, checkFeKaFWHMvsDy, checkFeFWHMRatioVsDy,
    checkFeKaAsymmetryVsDy
)
from .elementalRatios import (
    checkDyResidualStandardizedMasked, create_particle_mask
)
from .queue import (
    build_fine_scan_requests,
    submit_fine_scan_requests,
)


class InteractiveSpectralView(QGraphicsView):
    """Graphics view for displaying spectral features with circle selection."""

    def __init__(self, parent_widget):
        super().__init__()
        self.parent_widget = parent_widget
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        self.current_image = None
        self.circle_radius = 8
        self.selection_circles = []

    def set_image(self, image_array, cmap='viridis', diverging=False, center_on_median=False):
        """Display a 2D numpy array as a colored image using matplotlib colormaps."""
        self.scene.clear()
        self.selection_circles.clear()

        if image_array is None or image_array.size == 0:
            return

        # Check for all NaN
        if np.isnan(image_array).all():
            return

        self.current_image = image_array

        # Get finite values for normalization
        finite_mask = np.isfinite(image_array)
        if not finite_mask.any():
            return

        finite_values = image_array[finite_mask]

        # Normalize data
        if diverging and center_on_median:
            # For ratio maps: center on median (matching original functions)
            median_value = np.nanmedian(finite_values)
            ratio_low = np.nanpercentile(finite_values, 1)
            ratio_high = np.nanpercentile(finite_values, 99)

            vmax_dist = max(
                abs(ratio_high - median_value),
                abs(median_value - ratio_low)
            )

            if vmax_dist == 0:
                vmax_dist = 1

            vmin = median_value - vmax_dist
            vmax = median_value + vmax_dist
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif diverging:
            # For asymmetry/residual maps: center on zero
            vmax = np.nanpercentile(np.abs(finite_values), 99)
            vmin = -vmax
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            # For sequential colormaps
            vmin = np.nanpercentile(finite_values, 1)
            vmax = np.nanpercentile(finite_values, 99)
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Get colormap and set NaN to white (matching original functions)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad('white')  # NaN values show as white

        # Apply colormap
        mapper = ScalarMappable(norm=norm, cmap=cmap_obj)
        rgba_image = mapper.to_rgba(image_array, bytes=True)

        # Convert RGBA to RGB for QImage
        height, width = image_array.shape
        rgb_image = rgba_image[:, :, :3].copy()  # Drop alpha channel and ensure C-contiguous

        # Ensure the array is C-contiguous (required by QImage)
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)

        # Create QImage from RGB data
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """Zoom in/out with mouse wheel."""
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """Handle mouse click to add selection circle."""
        if event.button() == Qt.LeftButton and self.current_image is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())

            # Check if coordinates are within image bounds
            if 0 <= x < self.current_image.shape[1] and 0 <= y < self.current_image.shape[0]:
                self.add_selection_circle(x, y)
                self.parent_widget.add_selected_region(x, y, self.circle_radius)

    def add_selection_circle(self, x, y):
        """Draw a circle at the specified coordinates."""
        radius = self.circle_radius

        # Create circle
        circle = QGraphicsEllipseItem(
            x - radius, y - radius,
            radius * 2, radius * 2
        )
        circle.setPen(QPen(QColor(0, 255, 255), 2))  # Cyan circle
        circle.setBrush(QBrush(QColor(0, 255, 255, 50)))  # Semi-transparent fill

        self.scene.addItem(circle)
        self.selection_circles.append((x, y, circle))

    def clear_selections(self):
        """Remove all selection circles."""
        for _, _, circle_item in self.selection_circles:
            self.scene.removeItem(circle_item)
        self.selection_circles.clear()

    def update_circle_radius(self, radius):
        """Update the radius for new selections."""
        self.circle_radius = radius


class SpectralAnalysisWidget(QWidget):
    """Main widget for Fe-Dy spectral interference analysis."""

    def __init__(self, app_state):
        super().__init__()
        self.app_state = app_state

        # Data storage
        self.h5_file_path = None
        self.counts = None  # Raw XRF spectra
        self.xrf_fit = None  # Fitted element maps
        self.xrf_names = None  # Element names
        self.positions = None  # Stage positions from h5 file
        self.element_maps = {}  # Dictionary of element maps
        self.spectral_features = {}  # Computed spectral features
        self.selected_regions = []  # List of (x, y, radius) tuples

        # Pixel to micron conversion (calculated from positions)
        self.microns_per_pixel_x = 1.0
        self.microns_per_pixel_y = 1.0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Top controls
        controls_layout = self.create_controls()
        main_layout.addLayout(controls_layout)

        # Splitter for left/right panels (images)
        image_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Original map
        left_panel = self.create_left_panel()
        image_splitter.addWidget(left_panel)

        # Right panel: Spectral features
        right_panel = self.create_right_panel()
        image_splitter.addWidget(right_panel)

        image_splitter.setSizes([500, 500])  # Equal size for both image panels

        # Vertical splitter for images vs selected regions
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(image_splitter)

        # Bottom panel: Selected regions
        selected_panel = self.create_selected_regions_panel()
        main_splitter.addWidget(selected_panel)

        # Make images much larger than selected panel (80% vs 20%)
        main_splitter.setSizes([800, 200])
        main_splitter.setStretchFactor(0, 4)  # Images get 4x more space
        main_splitter.setStretchFactor(1, 1)  # Selected regions get 1x

        main_layout.addWidget(main_splitter)

        self.setLayout(main_layout)

    def create_controls(self):
        """Create top control buttons."""
        layout = QVBoxLayout()

        # Top row: File info
        file_info_layout = QHBoxLayout()
        file_info_layout.addWidget(QLabel("<b>H5 File:</b>"))
        self.filename_label = QLabel("No file loaded")
        self.filename_label.setStyleSheet("color: #666; font-size: 11pt;")
        file_info_layout.addWidget(self.filename_label)
        file_info_layout.addStretch()
        layout.addLayout(file_info_layout)

        # Bottom row: Action buttons
        button_layout = QHBoxLayout()

        # Load H5 file button
        self.load_btn = QPushButton("📁 Load H5 File")
        self.load_btn.clicked.connect(self.load_h5_file)
        button_layout.addWidget(self.load_btn)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #0066cc; font-style: italic;")
        button_layout.addWidget(self.status_label, stretch=1)

        # Compute features button
        self.compute_btn = QPushButton("🔬 Compute Spectral Features")
        self.compute_btn.clicked.connect(self.compute_spectral_features)
        self.compute_btn.setEnabled(False)
        button_layout.addWidget(self.compute_btn)

        layout.addLayout(button_layout)

        return layout

    def create_left_panel(self):
        """Create left panel for original map display."""
        panel = QGroupBox("Original Particle Map")
        layout = QVBoxLayout()

        # Element selector for original map
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Display Element:"))

        self.left_element_combo = QComboBox()
        self.left_element_combo.addItems(["Fe", "Dy", "Fe+Dy Combined"])
        self.left_element_combo.currentTextChanged.connect(self.update_left_display)
        selector_layout.addWidget(self.left_element_combo)
        selector_layout.addStretch()

        layout.addLayout(selector_layout)

        # Graphics view for original map
        self.left_view = InteractiveSpectralView(self)
        layout.addWidget(self.left_view)

        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        """Create right panel for spectral feature display."""
        panel = QGroupBox("Spectral Features")
        layout = QVBoxLayout()

        # Feature selector dropdown
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Select Feature:"))

        self.feature_combo = QComboBox()
        self.feature_combo.addItems([
            "Fe Kα/Kβ Ratio Map",
            "Fe Kα FWHM Map",
            "Fe Kα/Kβ FWHM Ratio Map",
            "Fe Kα Asymmetry Map",
            "Dy Residual (Standardized)",
            "Apparent Dy Map",
        ])
        self.feature_combo.currentTextChanged.connect(self.update_feature_display)
        selector_layout.addWidget(self.feature_combo, stretch=1)

        layout.addLayout(selector_layout)

        # Circle radius control
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Circle Radius:"))

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(3, 50)
        self.radius_spin.setValue(8)
        self.radius_spin.valueChanged.connect(self.update_circle_radius)
        radius_layout.addWidget(self.radius_spin)
        radius_layout.addStretch()

        layout.addLayout(radius_layout)

        # Graphics view for spectral features
        self.right_view = InteractiveSpectralView(self)
        layout.addWidget(self.right_view)

        panel.setLayout(layout)
        return panel

    def create_selected_regions_panel(self):
        """Create panel for displaying selected regions."""
        panel = QGroupBox("Selected Regions for XANES")
        layout = QHBoxLayout()

        # List widget for selected regions
        self.regions_list = QListWidget()
        self.regions_list.setMaximumHeight(180)  # Limit height
        layout.addWidget(self.regions_list)

        # Control buttons
        button_layout = QVBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_selections)
        button_layout.addWidget(clear_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_region)
        button_layout.addWidget(remove_btn)

        export_btn = QPushButton("Export to Queue")
        export_btn.clicked.connect(self.export_to_queue)
        button_layout.addWidget(export_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        panel.setLayout(layout)
        panel.setMaximumHeight(250)  # Limit panel height
        return panel

    def load_h5_file(self):
        """Load H5 file with XRF data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select H5 File", "", "HDF5 Files (*.h5 *.hdf5)"
        )

        if not file_path:
            return

        try:
            with h5py.File(file_path, "r") as f:
                # Load XRF data
                self.counts = np.array(f["xrfmap/detsum/counts"])
                self.xrf_fit = np.array(f["xrfmap/detsum/xrf_fit"])

                xrf_names_raw = np.array(f["xrfmap/detsum/xrf_fit_name"])
                self.xrf_names = [
                    name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    for name in xrf_names_raw
                ]

                # Load positions for pixel-to-micron conversion
                self.positions = np.array(f["xrfmap/positions/pos"])

            # Extract element maps
            self.extract_element_maps()

            # Calculate pixel-to-micron conversion from positions
            self.calculate_pixel_to_micron_conversion()

            self.h5_file_path = file_path
            filename = Path(file_path).name
            shape_str = f"{self.counts.shape[0]}×{self.counts.shape[1]} pixels"
            self.filename_label.setText(f"{filename}  ({shape_str})")
            self.filename_label.setStyleSheet("color: #000; font-size: 11pt; font-weight: bold;")
            self.status_label.setText(f"✓ File loaded successfully - {len(self.element_maps)} elements found")
            self.status_label.setStyleSheet("color: #009900; font-style: italic;")
            self.compute_btn.setEnabled(True)

            # Update left display
            self.update_left_display()

            QMessageBox.information(
                self, "Success",
                f"Loaded H5 file successfully!\n\nShape: {self.counts.shape}\nElements found: {len(self.element_maps)}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading File",
                f"Failed to load H5 file:\n{str(e)}"
            )
            self.status_label.setText("✗ Failed to load file")
            self.status_label.setStyleSheet("color: #cc0000; font-style: italic;")

    def extract_element_maps(self):
        """Extract element maps from XRF fit data."""
        self.element_maps.clear()

        # Common elements to extract
        elements_to_extract = ["Fe_K", "Dy_L", "Al_K", "Si_K", "Ca_K", "S_K", "P_K"]

        for elem in elements_to_extract:
            try:
                elem_map = self.get_element_map(elem)
                if elem_map is not None:
                    self.element_maps[elem] = elem_map
            except:
                pass  # Element not found

    def get_element_map(self, element_name):
        """Extract a specific element map from XRF fit data."""
        try:
            idx = self.xrf_names.index(element_name)
            return self.xrf_fit[idx, :, :]
        except (ValueError, IndexError):
            return None

    def calculate_pixel_to_micron_conversion(self):
        """Calculate pixel-to-micron conversion from position data."""
        if self.positions is None or self.positions.size == 0:
            return

        try:
            # positions shape: (n_positions, 2) where columns are [x, y]
            # Positions are in microns from the h5 file
            nrows, ncols = self.counts.shape[:2]

            # Get unique x and y positions
            x_positions = self.positions[:, 0]
            y_positions = self.positions[:, 1]

            # Reshape to grid if possible
            x_grid = x_positions.reshape(nrows, ncols)
            y_grid = y_positions.reshape(nrows, ncols)

            # Calculate step sizes (microns per pixel)
            # Use the difference between adjacent positions
            if ncols > 1:
                x_steps = np.diff(x_grid[0, :])
                self.microns_per_pixel_x = np.abs(np.median(x_steps))

            if nrows > 1:
                y_steps = np.diff(y_grid[:, 0])
                self.microns_per_pixel_y = np.abs(np.median(y_steps))

            # Store origin (top-left position)
            self.origin_x = x_grid[0, 0]
            self.origin_y = y_grid[0, 0]

        except Exception as e:
            print(f"Warning: Could not calculate pixel-to-micron conversion: {e}")
            # Use default 1.0 values set in __init__

    def compute_spectral_features(self):
        """Compute all spectral features."""
        if self.counts is None or "Fe_K" not in self.element_maps or "Dy_L" not in self.element_maps:
            QMessageBox.warning(
                self, "Missing Data",
                "Please load an H5 file with Fe and Dy data first."
            )
            return

        try:
            Fe = self.element_maps.get("Fe_K")
            Dy = self.element_maps.get("Dy_L")

            # Show progress message
            self.status_label.setText("Computing spectral features...")
            self.status_label.setStyleSheet("color: #0066cc; font-style: italic;")
            QMessageBox.information(
                self, "Computing",
                "Computing spectral features...\n\nThis may take 30-60 seconds.\nPlease wait and close any matplotlib windows that appear."
            )

            # Close any existing matplotlib figures
            plt.close('all')

            # Compute Fe Kα/Kβ ratio
            self.status_label.setText("⏳ Computing Fe Kα/Kβ ratio...")
            self.status_label.setStyleSheet("color: #0066cc; font-style: italic;")
            ratio_results = checkFeRatioWithDy(self.counts, Dy)
            plt.close('all')  # Close matplotlib plots
            self.spectral_features["fe_ratio_map"] = ratio_results["ratio_map"]

            # Compute Fe Kα FWHM
            self.status_label.setText("⏳ Computing Fe Kα FWHM...")
            self.spectral_features["fe_ka_fwhm_map"] = self.compute_fe_ka_fwhm(self.counts, Dy)

            # Compute FWHM ratio
            self.status_label.setText("⏳ Computing FWHM ratio...")
            self.spectral_features["fwhm_ratio_map"] = self.compute_fwhm_ratio(self.counts, Dy)

            # Compute asymmetry
            self.status_label.setText("⏳ Computing asymmetry...")
            self.spectral_features["asymmetry_map"] = self.compute_asymmetry(self.counts, Dy)

            # Compute Dy residual
            self.status_label.setText("⏳ Computing Dy residual...")
            particle_mask = create_particle_mask(Fe + Dy, min_particle_size=20)

            # This function creates plots - suppress them
            residual_results = checkDyResidualStandardizedMasked(
                Fe, Dy, "Fe", "Dy",
                particle_mask=particle_mask
            )
            plt.close('all')  # Close matplotlib plots

            # Check if the key exists
            if "standardized_residual_map" not in residual_results:
                raise KeyError(
                    f"'standardized_residual_map' not found in results. "
                    f"Available keys: {list(residual_results.keys())}"
                )

            self.spectral_features["dy_residual_map"] = residual_results["standardized_residual_map"]

            # Store apparent Dy
            self.spectral_features["apparent_dy"] = Dy

            self.status_label.setText("✓ Spectral features computed successfully!")
            self.status_label.setStyleSheet("color: #009900; font-style: italic; font-weight: bold;")

            # Update feature display
            self.update_feature_display()

            QMessageBox.information(
                self, "Success",
                "Spectral features computed successfully!"
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Computation Error",
                f"Failed to compute spectral features:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.status_label.setText("✗ Error during computation")
            self.status_label.setStyleSheet("color: #cc0000; font-style: italic; font-weight: bold;")
        finally:
            # Always close matplotlib figures
            plt.close('all')

    def compute_fe_ka_fwhm(self, counts, dy_map):
        """Compute Fe Kα FWHM map (without plotting)."""
        from scipy.signal import find_peaks, peak_widths

        energies = np.arange(counts.shape[-1]) / 100.0
        nrows, ncols = counts.shape[:2]

        fe_ka_energy = 6.40
        window_width = 0.15
        min_peak_height = 10
        min_prominence = 5

        ka_indices = np.where(
            (energies >= fe_ka_energy - window_width) &
            (energies <= fe_ka_energy + window_width)
        )[0]

        ka_fwhm_map = np.full((nrows, ncols), np.nan)

        for i in range(nrows):
            for j in range(ncols):
                spectrum = counts[i, j, :].astype(float)
                y = spectrum[ka_indices]

                baseline = np.median(
                    np.concatenate([
                        y[:max(2, len(y)//10)],
                        y[-max(2, len(y)//10):]
                    ])
                )

                y = y - baseline
                y[y < 0] = 0

                peaks, properties = find_peaks(
                    y, height=min_peak_height, prominence=min_prominence
                )

                if len(peaks) == 0:
                    continue

                peak = peaks[np.argmax(y[peaks])]
                widths, _, _, _ = peak_widths(y, [peak], rel_height=0.5)

                energy_step = np.mean(np.diff(energies[ka_indices]))
                ka_fwhm_map[i, j] = widths[0] * energy_step

        return ka_fwhm_map

    def compute_fwhm_ratio(self, counts, dy_map):
        """Compute FWHM ratio map (without plotting)."""
        from scipy.signal import find_peaks, peak_widths

        energies = np.arange(counts.shape[-1]) / 100.0
        nrows, ncols = counts.shape[:2]

        fe_ka_energy = 6.40
        fe_kb_energy = 7.06
        window_width = 0.15
        min_peak_height = 0
        min_prominence = 3

        def get_fwhm(spectrum, center):
            indices = np.where(
                (energies >= center - window_width) &
                (energies <= center + window_width)
            )[0]

            y = spectrum[indices].astype(float)
            if len(y) < 5:
                return np.nan

            n_edge = max(2, len(y) // 10)
            baseline = np.median(np.concatenate([y[:n_edge], y[-n_edge:]]))
            y = y - baseline
            y[y < 0] = 0

            peaks, properties = find_peaks(y, height=min_peak_height, prominence=min_prominence)
            if len(peaks) == 0:
                return np.nan

            peak = peaks[np.argmax(y[peaks])]
            widths, _, _, _ = peak_widths(y, [peak], rel_height=0.5)

            energy_step = np.mean(np.diff(energies[indices]))
            return widths[0] * energy_step

        ka_fwhm_map = np.full((nrows, ncols), np.nan)
        kb_fwhm_map = np.full((nrows, ncols), np.nan)

        for i in range(nrows):
            for j in range(ncols):
                spectrum = counts[i, j, :]
                ka_fwhm_map[i, j] = get_fwhm(spectrum, fe_ka_energy)
                kb_fwhm_map[i, j] = get_fwhm(spectrum, fe_kb_energy)

        valid = (
            np.isfinite(ka_fwhm_map) &
            np.isfinite(kb_fwhm_map) &
            (kb_fwhm_map > 0)
        )

        fwhm_ratio_map = np.full((nrows, ncols), np.nan)
        fwhm_ratio_map[valid] = ka_fwhm_map[valid] / kb_fwhm_map[valid]

        return fwhm_ratio_map

    def compute_asymmetry(self, counts, dy_map):
        """Compute Fe Kα asymmetry map (without plotting)."""
        from scipy.signal import find_peaks, peak_widths

        energies = np.arange(counts.shape[-1]) / 100.0
        nrows, ncols = counts.shape[:2]

        fe_ka_energy = 6.40
        window_width = 0.15
        min_peak_height = 10
        min_prominence = 5

        ka_indices = np.where(
            (energies >= fe_ka_energy - window_width) &
            (energies <= fe_ka_energy + window_width)
        )[0]

        asymmetry_map = np.full((nrows, ncols), np.nan)

        for i in range(nrows):
            for j in range(ncols):
                spectrum = counts[i, j, :].astype(float)
                y = spectrum[ka_indices]

                n_edge = max(2, len(y) // 10)
                baseline = np.median(
                    np.concatenate([y[:n_edge], y[-n_edge:]])
                )

                y = y - baseline
                y[y < 0] = 0

                peaks, properties = find_peaks(
                    y, height=min_peak_height, prominence=min_prominence
                )

                if len(peaks) == 0:
                    continue

                peak = peaks[np.argmax(y[peaks])]
                widths, width_heights, left_ips, right_ips = peak_widths(
                    y, [peak], rel_height=0.5
                )

                left_ip = left_ips[0]
                right_ip = right_ips[0]

                left_width = peak - left_ip
                right_width = right_ip - peak

                denominator = left_width + right_width
                if denominator <= 0:
                    continue

                asymmetry_map[i, j] = (right_width - left_width) / denominator

        return asymmetry_map

    def update_left_display(self):
        """Update the left panel display."""
        if not self.element_maps:
            return

        selected = self.left_element_combo.currentText()

        if selected == "Fe":
            display_map = self.element_maps.get("Fe_K")
        elif selected == "Dy":
            display_map = self.element_maps.get("Dy_L")
        elif selected == "Fe+Dy Combined":
            Fe = self.element_maps.get("Fe_K", np.zeros((10, 10)))
            Dy = self.element_maps.get("Dy_L", np.zeros((10, 10)))
            display_map = Fe + Dy
        else:
            return

        if display_map is not None:
            # Use 'jet' colormap for element maps (matches original visualization)
            self.left_view.set_image(display_map, cmap='jet', diverging=False, center_on_median=False)

    def update_feature_display(self):
        """Update the right panel with selected spectral feature."""
        if not self.spectral_features:
            return

        selected = self.feature_combo.currentText()

        # Define colormap, diverging, and centering for each feature type
        # Format: (key, colormap, diverging, center_on_median)
        feature_config = {
            "Fe Kα/Kβ Ratio Map": ("fe_ratio_map", "seismic", True, True),
            "Fe Kα FWHM Map": ("fe_ka_fwhm_map", "seismic", True, True),
            "Fe Kα/Kβ FWHM Ratio Map": ("fwhm_ratio_map", "seismic", True, True),
            "Fe Kα Asymmetry Map": ("asymmetry_map", "seismic", True, False),
            "Dy Residual (Standardized)": ("dy_residual_map", "seismic", True, False),
            "Apparent Dy Map": ("apparent_dy", "jet", False, False),
        }

        if selected in feature_config:
            feature_key, cmap, diverging, center_on_median = feature_config[selected]
            if feature_key in self.spectral_features:
                self.right_view.set_image(
                    self.spectral_features[feature_key],
                    cmap=cmap,
                    diverging=diverging,
                    center_on_median=center_on_median
                )

    def update_circle_radius(self, radius):
        """Update circle radius for both views."""
        self.left_view.update_circle_radius(radius)
        self.right_view.update_circle_radius(radius)

    def add_selected_region(self, x, y, radius):
        """Add a selected region to the list."""
        self.selected_regions.append((x, y, radius))

        # Add to list widget
        item_text = f"Region {len(self.selected_regions)}: (x={x}, y={y}, r={radius})"
        self.regions_list.addItem(item_text)

        # Also add circle to left view
        self.left_view.add_selection_circle(x, y)

    def clear_all_selections(self):
        """Clear all selected regions."""
        self.selected_regions.clear()
        self.regions_list.clear()
        self.left_view.clear_selections()
        self.right_view.clear_selections()

    def remove_selected_region(self):
        """Remove the currently selected region from the list."""
        current_row = self.regions_list.currentRow()
        if current_row >= 0:
            self.regions_list.takeItem(current_row)
            del self.selected_regions[current_row]

            # Redraw all circles
            self.left_view.clear_selections()
            self.right_view.clear_selections()

            for x, y, r in self.selected_regions:
                self.left_view.add_selection_circle(x, y)

    def export_to_queue(self):
        """Export selected regions to the scan queue."""
        if not self.selected_regions:
            QMessageBox.warning(
                self, "No Regions Selected",
                "Please select at least one region before exporting."
            )
            return

        # Convert pixel coordinates to fine-scan rows in microns
        fine_scan_rows = self._convert_regions_to_scan_rows()

        if not fine_scan_rows:
            QMessageBox.warning(
                self, "No Valid Regions",
                "No valid regions could be converted for export."
            )
            return

        # Ask user to select configuration file
        config_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select fine-scan configuration",
            "",
            "JSON files (*.json)",
        )
        if not config_path:
            return

        # Build scan requests
        try:
            mode, requests = build_fine_scan_requests(config_path, pd.DataFrame(fine_scan_rows))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.warning(
                self, "Queue Preview",
                f"Could not build scan requests:\n{error}"
            )
            return

        if mode != "real":
            QMessageBox.warning(
                self,
                "Queue submission disabled",
                "This configuration is not in real mode, so no plans were sent.\n\n"
                "For the local HXN QueueServer simulator, use a configuration with "
                "execution_params.mode set to 'real'.",
            )
            return

        # Show confirmation dialog
        first = requests[0]
        summary = (
            "The following plans will be sent to the configured local QueueServer.\n"
            "The queue will start immediately after all plans are accepted.\n\n"
            f"Planned scans: {len(requests)}\n\n"
            f"First scan: {first['label']}\n"
            f"Center: X={first['center']['x']:.2f} µm, Y={first['center']['y']:.2f} µm\n"
            f"Requested size: {first['requested_size']['x']:.2f} × {first['requested_size']['y']:.2f} µm\n"
            f"Padded size: {first['padded_size']['x']:.2f} × {first['padded_size']['y']:.2f} µm\n"
            f"Motors: {first['motors']['x']} / {first['motors']['y']}\n"
            f"Points: {first['points']['x']} × {first['points']['y']}\n"
            f"Dwell: {first['dwell']:.4f} s\n"
            f"Plan: {first['plan_name']}\n\n"
            "Submit all selected regions to QueueServer?"
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Queue Submission")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(summary)
        dialog.setDetailedText(json.dumps(requests, indent=2))
        dialog.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        dialog.button(QMessageBox.Ok).setText("Submit and Start Queue")
        dialog.button(QMessageBox.Cancel).setText("Cancel")
        if dialog.exec_() != QMessageBox.Ok:
            return

        # Submit to queue server
        try:
            result = submit_fine_scan_requests(requests)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Queue submission failed",
                f"No further plans were sent.\n\n{error}",
            )
            return

        submitted_labels = "\n".join(f"• {item['label']}" for item in result["submitted"])
        status = result["status"]
        QMessageBox.information(
            self,
            "Plans submitted",
            f"QueueServer accepted and started {len(result['submitted'])} plan(s):\n"
            f"{submitted_labels}\n\n"
            f"Queue state: {status.get('manager_state', 'unknown')}\n"
            "For the local simulator, receipt files are written in hxn-qserver-sim/output/.",
        )

    def _convert_regions_to_scan_rows(self):
        """Convert selected pixel regions to fine-scan table rows in microns."""
        rows = []

        for idx, (x_pixel, y_pixel, radius_pixel) in enumerate(self.selected_regions):
            # Convert pixel coordinates to microns
            # x_pixel, y_pixel are the CENTER of the circle
            cx_microns = self.origin_x + (x_pixel * self.microns_per_pixel_x)
            cy_microns = self.origin_y + (y_pixel * self.microns_per_pixel_y)

            # Convert radius to size (diameter in pixels, then to microns)
            size_x_pixels = radius_pixel * 2
            size_y_pixels = radius_pixel * 2
            size_x_microns = size_x_pixels * self.microns_per_pixel_x
            size_y_microns = size_y_pixels * self.microns_per_pixel_y

            # Create label
            label = f"Spectral_Region_{idx+1}_({x_pixel},{y_pixel})"

            rows.append({
                'label': label,
                'cx': cx_microns,
                'cy': cy_microns,
                'num_x': size_x_microns,
                'num_y': size_y_microns,
            })

        return rows
