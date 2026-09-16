import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from matplotlib.patches import Circle

def checkFeRatio(counts):
    energies = np.arange(counts.shape[-1])[0:1001]/100
    fe_ka_energy = 6.40
    fe_kb_energy = 7.06
    window_width = 0.15 # keV window radius (adjust based on your detector's resolution)

    # 4. Find the channel indices corresponding to these energy windows
    ka_indices = np.where((energies >= fe_ka_energy - window_width) & 
                        (energies <= fe_ka_energy + window_width))[0]

    kb_indices = np.where((energies >= fe_kb_energy - window_width) & 
                        (energies <= fe_kb_energy + window_width))[0]

    # 5. Extract intensities by slicing and summing along the channel axis (axis=-1)
    # This collapses the (81, 81, channels) array into an (81, 81) intensity map
    fe_ka_map = np.sum(counts[:, :, ka_indices], axis=-1)
    fe_kb_map = np.sum(counts[:, :, kb_indices], axis=-1)

    # --- STEP 1: Calculate the Ratio Safely ---
    # Filter out low-count background pixels to avoid division-by-zero or massive noise
    min_counts_threshold = 10
    valid_pixels = (fe_ka_map > min_counts_threshold) & (fe_kb_map > min_counts_threshold)

    ratio_map = np.full_like(fe_ka_map, np.nan) # Fill background with NaN so it hides in the plot
    ratio_map[valid_pixels] = fe_ka_map[valid_pixels] / fe_kb_map[valid_pixels]

    # Find median ratio of valid pixels to center our diverging colormap
    median_ratio = np.nanmedian(ratio_map)

    # --- STEP 2: Plotting ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: 2D Spatial Ratio Map
    # Using a diverging colormap 'seismic' centered on the median ratio
    vmax_dist = max(np.nanmax(ratio_map) - median_ratio, median_ratio - np.nanmin(ratio_map))
    im = ax[0].imshow(ratio_map, cmap='seismic', 
                    vmin=median_ratio - vmax_dist*0.5, vmax=median_ratio + vmax_dist*0.5)
    ax[0].set_title("Spatial Map: Fe K-alpha / K-beta Ratio\n(Look for localized color clusters)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax[0], label='Ratio Value')
    cbar.ax.axhline(median_ratio, color='black', lw=2) # Line showing expected ratio

    # Chart 2: Correlation Scatter Plot
    # Flatten the 2D arrays to 1D for scattering
    ka_flat = fe_ka_map[valid_pixels]
    kb_flat = fe_kb_map[valid_pixels]
    ratio_flat = ratio_map[valid_pixels]

    scatter = ax[1].scatter(kb_flat, ka_flat, c=ratio_flat, cmap='seismic', 
                            vmin=median_ratio - vmax_dist*0.5, vmax=median_ratio + vmax_dist*0.5, alpha=0.6, edgecolors='none')
    ax[1].set_xlabel("Fe K-beta Intensity (Counts)")
    ax[1].set_ylabel("Fe K-alpha Intensity (Counts)")
    ax[1].set_title("Scatter Plot: K-alpha vs K-beta\n(Look for outliers off the main linear trend)", fontsize=12)

    # Draw the theoretical expectation line based on the median
    x_line = np.array([np.min(kb_flat), np.max(kb_flat)])
    ax[1].plot(x_line, x_line * median_ratio, color='black', linestyle='--', label=f'Expected Linear Trend ({median_ratio:.2f}x)')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def checkFeRatioWithDy(counts, Dy_count):
    energies = np.arange(counts.shape[-1])[0:1001] / 100

    fe_ka_energy = 6.40
    fe_kb_energy = 7.06
    window_width = 0.15  # keV window radius

    # ============================================================
    # STEP 1: Find energy-channel indices
    # ============================================================

    ka_indices = np.where(
        (energies >= fe_ka_energy - window_width) &
        (energies <= fe_ka_energy + window_width)
    )[0]

    kb_indices = np.where(
        (energies >= fe_kb_energy - window_width) &
        (energies <= fe_kb_energy + window_width)
    )[0]


    # ============================================================
    # STEP 2: Calculate Fe K-alpha and K-beta intensity maps
    # ============================================================

    fe_ka_map = np.sum(
        counts[:, :, ka_indices],
        axis=-1
    )

    fe_kb_map = np.sum(
        counts[:, :, kb_indices],
        axis=-1
    )


    # ============================================================
    # STEP 3: Select valid pixels
    # ============================================================

    min_counts_threshold = 10

    valid_pixels = (
        (fe_ka_map > min_counts_threshold) &
        (fe_kb_map > min_counts_threshold) &
        np.isfinite(Dy_count)
    )


    # ============================================================
    # STEP 4: Calculate Fe K-alpha / K-beta ratio
    # ============================================================

    ratio_map = np.full_like(
        fe_ka_map,
        np.nan,
        dtype=float
    )

    ratio_map[valid_pixels] = (
        fe_ka_map[valid_pixels] /
        fe_kb_map[valid_pixels]
    )


    # ============================================================
    # STEP 5: Median ratio
    # ============================================================

    ratio_values = ratio_map[valid_pixels]

    if len(ratio_values) == 0:
        raise ValueError(
            "No valid pixels found."
        )

    median_ratio = np.nanmedian(
        ratio_values
    )


    # ============================================================
    # STEP 6: Flatten data for scatter plots
    # ============================================================

    ka_flat = fe_ka_map[valid_pixels]
    kb_flat = fe_kb_map[valid_pixels]
    ratio_flat = ratio_map[valid_pixels]
    dy_flat = Dy_count[valid_pixels]


    # ============================================================
    # STEP 7: Ratio color scale
    # ============================================================

    ratio_low = np.nanpercentile(
        ratio_flat,
        1
    )

    ratio_high = np.nanpercentile(
        ratio_flat,
        99
    )

    vmax_dist = max(
        abs(ratio_high - median_ratio),
        abs(median_ratio - ratio_low)
    )

    if vmax_dist == 0:
        vmax_dist = 1

    vmin = median_ratio - vmax_dist
    vmax = median_ratio + vmax_dist


    # ============================================================
    # STEP 8: Create figure
    # ============================================================

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(20, 6)
    )


    # ============================================================
    # PLOT 1: Fe K-alpha / K-beta spatial ratio map
    # ============================================================

    ratio_cmap = plt.get_cmap(
        "seismic"
    ).copy()

    ratio_cmap.set_bad("white")

    im = ax[0].imshow(
        ratio_map,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax[0].set_title(
        "Spatial Map: Fe K-alpha / K-beta Ratio",
        fontsize=12
    )

    ax[0].set_xlabel(
        "Column (pixel)"
    )

    ax[0].set_ylabel(
        "Row (pixel)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax[0],
        label="K-alpha / K-beta Ratio"
    )

    cbar.ax.axhline(
        median_ratio,
        color="black",
        linewidth=2
    )


    # ============================================================
    # PLOT 2: Fe K-alpha vs K-beta
    # ============================================================

    scatter1 = ax[1].scatter(
        kb_flat,
        ka_flat,
        c=ratio_flat,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        edgecolors="none"
    )

    ax[1].set_xlabel(
        "Fe K-beta Intensity (Counts)"
    )

    ax[1].set_ylabel(
        "Fe K-alpha Intensity (Counts)"
    )

    ax[1].set_title(
        "Fe K-alpha vs K-beta",
        fontsize=12
    )

    # Expected linear relationship
    x_line = np.array([
        np.min(kb_flat),
        np.max(kb_flat)
    ])

    ax[1].plot(
        x_line,
        x_line * median_ratio,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Median ratio = {median_ratio:.2f}"
    )

    ax[1].legend()

    fig.colorbar(
        scatter1,
        ax=ax[1],
        label="K-alpha / K-beta Ratio"
    )


    # ============================================================
    # PLOT 3: Dy vs Fe K-alpha / K-beta ratio
    # ============================================================

    scatter2 = ax[2].scatter(
        ratio_flat,
        dy_flat,
        c=ratio_flat,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        edgecolors="none"
    )

    ax[2].set_xlabel(
        "Fe K-alpha / K-beta Ratio"
    )

    ax[2].set_ylabel(
        "Dy Intensity (Counts)"
    )

    ax[2].set_title(
        "Dy Intensity vs Fe K-alpha / K-beta Ratio",
        fontsize=12
    )

    # Median ratio reference
    ax[2].axvline(
        median_ratio,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Median ratio = {median_ratio:.2f}"
    )

    ax[2].legend()

    fig.colorbar(
        scatter2,
        ax=ax[2],
        label="K-alpha / K-beta Ratio"
    )


    # ============================================================
    # STEP 9: Display
    # ============================================================

    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 10: Print useful statistics
    # ============================================================

    print("\n" + "=" * 60)
    print("Fe K-alpha / K-beta ANALYSIS")
    print("=" * 60)

    print(
        f"Valid pixels: {len(ratio_flat):,}"
    )

    print(
        f"Median Fe K-alpha / K-beta ratio: "
        f"{median_ratio:.4f}"
    )

    print(
        f"Dy count range: "
        f"{np.nanmin(dy_flat):.2f} - "
        f"{np.nanmax(dy_flat):.2f}"
    )


    # ============================================================
    # STEP 11: Return results
    # ============================================================

    return {
        "fe_ka_map": fe_ka_map,
        "fe_kb_map": fe_kb_map,
        "ratio_map": ratio_map,
        "dy_flat": dy_flat,
        "ratio_flat": ratio_flat,
        "valid_pixels": valid_pixels,
        "median_ratio": median_ratio
    }

def checkFeFWHMRatio(
    counts,
    fe_ka_energy=6.40,
    fe_kb_energy=7.06,
    window_width=0.15,
    min_peak_height=5,
    min_peak_prominence=5
):
    """
    Calculate and plot the FWHM ratio of Fe K-alpha and Fe K-beta.

    FWHM ratio:

        FWHM(Fe K-alpha) / FWHM(Fe K-beta)

    Parameters
    ----------
    counts : ndarray
        XRF spectra with shape:

            (rows, columns, energy_channels)

    fe_ka_energy : float
        Approximate Fe K-alpha energy in keV.

    fe_kb_energy : float
        Approximate Fe K-beta energy in keV.

    window_width : float
        Half-width of the energy window around each peak in keV.

    min_peak_height : float
        Minimum peak height above baseline.

    min_peak_prominence : float
        Minimum peak prominence required to accept a peak.

    Returns
    -------
    results : dict
        Dictionary containing:

            ka_fwhm_map
            kb_fwhm_map
            fwhm_ratio_map
    """

    # ============================================================
    # STEP 1: Energy axis
    # ============================================================

    energies = np.arange(counts.shape[-1]) / 100.0


    # ============================================================
    # STEP 2: Define Fe K-alpha and K-beta windows
    # ============================================================

    ka_indices = np.where(
        (energies >= fe_ka_energy - window_width) &
        (energies <= fe_ka_energy + window_width)
    )[0]

    kb_indices = np.where(
        (energies >= fe_kb_energy - window_width) &
        (energies <= fe_kb_energy + window_width)
    )[0]


    if len(ka_indices) == 0 or len(kb_indices) == 0:
        raise ValueError(
            "No energy channels found inside one or both peak windows."
        )


    # ============================================================
    # STEP 3: Dimensions
    # ============================================================

    nrows, ncols = counts.shape[:2]


    # ============================================================
    # STEP 4: Initialize FWHM maps
    # ============================================================

    ka_fwhm_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )

    kb_fwhm_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )


    # ============================================================
    # Helper function to calculate FWHM
    # ============================================================

    def calculate_fwhm(spectrum, indices):

        # Extract local peak region
        y = spectrum[indices].astype(float)

        # Corresponding energy axis
        x = energies[indices]

        # --------------------------------------------------------
        # Estimate local baseline
        # --------------------------------------------------------

        n_edge = max(2, len(y) // 10)

        baseline = np.median(
            np.concatenate([
                y[:n_edge],
                y[-n_edge:]
            ])
        )

        y_corrected = y - baseline

        # Negative values are not useful for peak detection
        y_corrected[y_corrected < 0] = 0


        # --------------------------------------------------------
        # Find candidate peaks
        # --------------------------------------------------------

        peaks, properties = find_peaks(
            y_corrected,
            height=min_peak_height,
            prominence=min_peak_prominence
        )

        if len(peaks) == 0:
            return np.nan


        # --------------------------------------------------------
        # Select strongest peak
        # --------------------------------------------------------

        strongest_peak = peaks[
            np.argmax(
                y_corrected[peaks]
            )
        ]


        # --------------------------------------------------------
        # Calculate FWHM
        # --------------------------------------------------------

        widths, width_heights, left_ips, right_ips = peak_widths(
            y_corrected,
            [strongest_peak],
            rel_height=0.5
        )

        width = widths[0]


        # --------------------------------------------------------
        # Convert width from channels to keV
        # --------------------------------------------------------

        # Energy spacing in keV/channel
        energy_step = np.mean(
            np.diff(x)
        )

        fwhm = width * energy_step

        return fwhm


    # ============================================================
    # STEP 5: Calculate FWHM for every pixel
    # ============================================================

    print("Calculating Fe K-alpha / K-beta FWHM...")

    for i in range(nrows):

        for j in range(ncols):

            spectrum = counts[i, j, :]

            # Fe K-alpha
            ka_fwhm_map[i, j] = calculate_fwhm(
                spectrum,
                ka_indices
            )

            # Fe K-beta
            kb_fwhm_map[i, j] = calculate_fwhm(
                spectrum,
                kb_indices
            )


    # ============================================================
    # STEP 6: Calculate FWHM ratio
    # ============================================================

    valid_pixels = (
        np.isfinite(ka_fwhm_map) &
        np.isfinite(kb_fwhm_map) &
        (ka_fwhm_map > 0) &
        (kb_fwhm_map > 0)
    )

    fwhm_ratio_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )

    fwhm_ratio_map[valid_pixels] = (
        ka_fwhm_map[valid_pixels] /
        kb_fwhm_map[valid_pixels]
    )


    # ============================================================
    # STEP 7: Statistics
    # ============================================================

    ratio_values = fwhm_ratio_map[
        np.isfinite(fwhm_ratio_map)
    ]

    if len(ratio_values) == 0:
        raise ValueError(
            "No valid FWHM ratios were calculated."
        )

    median_ratio = np.median(ratio_values)

    print("\n==========================================")
    print("Fe K-alpha / K-beta FWHM RATIO")
    print("==========================================")

    print(
        f"Valid pixels:        {len(ratio_values):,}"
    )

    print(
        f"Median FWHM ratio:   {median_ratio:.4f}"
    )

    print(
        f"Mean FWHM ratio:     {np.mean(ratio_values):.4f}"
    )

    print(
        f"Minimum ratio:       {np.min(ratio_values):.4f}"
    )

    print(
        f"Maximum ratio:       {np.max(ratio_values):.4f}"
    )


    # ============================================================
    # STEP 8: Color scale for ratio map
    # ============================================================

    ratio_low = np.percentile(
        ratio_values,
        1
    )

    ratio_high = np.percentile(
        ratio_values,
        99
    )

    vmax_dist = max(
        abs(ratio_high - median_ratio),
        abs(median_ratio - ratio_low)
    )

    if vmax_dist == 0:
        vmax_dist = 1

    vmin = median_ratio - vmax_dist
    vmax = median_ratio + vmax_dist


    # ============================================================
    # STEP 9: Plot
    # ============================================================

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(24, 6)
    )


    # ------------------------------------------------------------
    # Fe K-alpha FWHM map
    # ------------------------------------------------------------

    im1 = ax[0].imshow(
        ka_fwhm_map,
        cmap="viridis"
    )

    ax[0].set_title(
        "Fe K-alpha FWHM"
    )

    ax[0].set_xlabel("Column (pixel)")
    ax[0].set_ylabel("Row (pixel)")

    fig.colorbar(
        im1,
        ax=ax[0],
        label="FWHM (keV)"
    )


    # ------------------------------------------------------------
    # Fe K-beta FWHM map
    # ------------------------------------------------------------

    im2 = ax[1].imshow(
        kb_fwhm_map,
        cmap="viridis"
    )

    ax[1].set_title(
        "Fe K-beta FWHM"
    )

    ax[1].set_xlabel("Column (pixel)")
    ax[1].set_ylabel("Row (pixel)")

    fig.colorbar(
        im2,
        ax=ax[1],
        label="FWHM (keV)"
    )


    # ------------------------------------------------------------
    # FWHM ratio map
    # ------------------------------------------------------------

    ratio_cmap = plt.get_cmap(
        "seismic"
    ).copy()

    ratio_cmap.set_bad("white")

    im3 = ax[2].imshow(
        fwhm_ratio_map,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax[2].set_title(
        "Fe K-alpha / K-beta FWHM Ratio"
    )

    ax[2].set_xlabel("Column (pixel)")
    ax[2].set_ylabel("Row (pixel)")

    cbar = fig.colorbar(
        im3,
        ax=ax[2],
        label="FWHM K-alpha / FWHM K-beta"
    )

    cbar.ax.axhline(
        median_ratio,
        color="black",
        linewidth=2
    )


    # ------------------------------------------------------------
    # FWHM correlation scatter
    # ------------------------------------------------------------

    ka_flat = ka_fwhm_map[valid_pixels]
    kb_flat = kb_fwhm_map[valid_pixels]

    scatter = ax[3].scatter(
        kb_flat,
        ka_flat,
        c=fwhm_ratio_map[valid_pixels],
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        edgecolors="none"
    )

    ax[3].set_xlabel(
        "Fe K-beta FWHM (keV)"
    )

    ax[3].set_ylabel(
        "Fe K-alpha FWHM (keV)"
    )

    ax[3].set_title(
        "K-alpha vs K-beta FWHM"
    )

    # Reference line corresponding to median FWHM ratio
    x_line = np.array([
        np.min(kb_flat),
        np.max(kb_flat)
    ])

    ax[3].plot(
        x_line,
        x_line * median_ratio,
        color="black",
        linestyle="--",
        linewidth=2,
        label=(
            f"Median ratio = "
            f"{median_ratio:.3f}"
        )
    )

    ax[3].legend()

    fig.colorbar(
        scatter,
        ax=ax[3],
        label="FWHM Ratio"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 10: Return results
    # ============================================================

    # return {
    #     "ka_fwhm_map": ka_fwhm_map,
    #     "kb_fwhm_map": kb_fwhm_map,
    #     "fwhm_ratio_map": fwhm_ratio_map,
    #     "valid_pixels": valid_pixels,
    #     "median_ratio": median_ratio
    # }

def checkFeRatioNew(
    counts,
    fe_ka_energy=6.40,
    fe_kb_energy=7.06,
    peak_half_width=0.15,
    background_width=0.05,
    min_peak_area=10
):
    """
    Calculate the background-subtracted integrated-area ratio:

        Fe Kα area / Fe Kβ area

    Parameters
    ----------
    counts : ndarray
        XRF data with shape:

            (rows, columns, energy_channels)

    fe_ka_energy : float
        Fe Kα energy in keV.

    fe_kb_energy : float
        Fe Kβ energy in keV.

    peak_half_width : float
        Half-width of the integration window in keV.

        Example:
            0.15 -> integrate +/- 0.15 keV around the peak.

    background_width : float
        Width of the background regions on each side
        of the peak window.

    min_peak_area : float
        Minimum background-subtracted peak area required
        for a pixel to be considered valid.

    Returns
    -------
    results : dict
        Dictionary containing:

            ka_area_map
            kb_area_map
            ratio_map
            valid_pixels
    """

    # ============================================================
    # STEP 1: Energy axis
    # ============================================================

    energies = (
        np.arange(counts.shape[-1]) / 100.0
    )

    nrows, ncols = counts.shape[:2]


    # ============================================================
    # STEP 2: Define peak and background windows
    # ============================================================

    def get_windows(center):

        peak = (
            (energies >= center - peak_half_width) &
            (energies <= center + peak_half_width)
        )

        left_background = (
            (energies >= center - peak_half_width - background_width) &
            (energies < center - peak_half_width)
        )

        right_background = (
            (energies > center + peak_half_width) &
            (energies <= center + peak_half_width + background_width)
        )

        return peak, left_background, right_background


    ka_peak, ka_bg_left, ka_bg_right = get_windows(
        fe_ka_energy
    )

    kb_peak, kb_bg_left, kb_bg_right = get_windows(
        fe_kb_energy
    )


    # ============================================================
    # STEP 3: Extract energy indices
    # ============================================================

    ka_indices = np.where(ka_peak)[0]
    kb_indices = np.where(kb_peak)[0]

    ka_bg_left_indices = np.where(ka_bg_left)[0]
    ka_bg_right_indices = np.where(ka_bg_right)[0]

    kb_bg_left_indices = np.where(kb_bg_left)[0]
    kb_bg_right_indices = np.where(kb_bg_right)[0]


    # ============================================================
    # STEP 4: Calculate background-subtracted peak area
    # ============================================================

    ka_area_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )

    kb_area_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )


    print("Calculating background-subtracted Fe peak areas...")


    for i in range(nrows):

        for j in range(ncols):

            spectrum = counts[i, j, :].astype(float)


            # ====================================================
            # Fe K-alpha
            # ====================================================

            ka_signal = spectrum[ka_indices]

            # Background values
            ka_bg_values = np.concatenate([
                spectrum[ka_bg_left_indices],
                spectrum[ka_bg_right_indices]
            ])

            if len(ka_bg_values) == 0:
                continue

            # Estimate local background
            ka_background = np.median(
                ka_bg_values
            )

            # Background-subtracted spectrum
            ka_corrected = (
                ka_signal - ka_background
            )

            # Prevent negative contribution
            ka_corrected = np.maximum(
                ka_corrected,
                0
            )

            # Integrated area
            ka_area = np.sum(
                ka_corrected
            )


            # ====================================================
            # Fe K-beta
            # ====================================================

            kb_signal = spectrum[kb_indices]

            kb_bg_values = np.concatenate([
                spectrum[kb_bg_left_indices],
                spectrum[kb_bg_right_indices]
            ])

            if len(kb_bg_values) == 0:
                continue

            # Estimate local background
            kb_background = np.median(
                kb_bg_values
            )

            # Background-subtracted spectrum
            kb_corrected = (
                kb_signal - kb_background
            )

            kb_corrected = np.maximum(
                kb_corrected,
                0
            )

            # Integrated area
            kb_area = np.sum(
                kb_corrected
            )


            # ====================================================
            # Save
            # ====================================================

            ka_area_map[i, j] = ka_area
            kb_area_map[i, j] = kb_area


    # ============================================================
    # STEP 5: Valid pixels
    # ============================================================

    valid_pixels = (
        np.isfinite(ka_area_map) &
        np.isfinite(kb_area_map) &
        (ka_area_map > min_peak_area) &
        (kb_area_map > min_peak_area)
    )


    # ============================================================
    # STEP 6: Calculate ratio
    # ============================================================

    ratio_map = np.full(
        (nrows, ncols),
        np.nan,
        dtype=float
    )

    ratio_map[valid_pixels] = (
        ka_area_map[valid_pixels] /
        kb_area_map[valid_pixels]
    )


    ratio_values = ratio_map[
        valid_pixels
    ]


    if len(ratio_values) == 0:
        raise ValueError(
            "No valid pixels were found. "
            "Try lowering min_peak_area."
        )


    # ============================================================
    # STEP 7: Statistics
    # ============================================================

    median_ratio = np.median(
        ratio_values
    )

    mean_ratio = np.mean(
        ratio_values
    )

    ratio_low = np.percentile(
        ratio_values,
        1
    )

    ratio_high = np.percentile(
        ratio_values,
        99
    )

    print("\n==============================================")
    print("BACKGROUND-SUBTRACTED Fe Kα / Kβ AREA RATIO")
    print("==============================================")

    print(
        f"Valid pixels:       {len(ratio_values):,}"
    )

    print(
        f"Median ratio:       {median_ratio:.4f}"
    )

    print(
        f"Mean ratio:         {mean_ratio:.4f}"
    )

    print(
        f"1st percentile:     {ratio_low:.4f}"
    )

    print(
        f"99th percentile:    {ratio_high:.4f}"
    )


    # ============================================================
    # STEP 8: Ratio color scale
    # ============================================================

    vmax_dist = max(
        abs(ratio_high - median_ratio),
        abs(median_ratio - ratio_low)
    )

    if vmax_dist == 0:
        vmax_dist = 1

    vmin = (
        median_ratio - vmax_dist
    )

    vmax = (
        median_ratio + vmax_dist
    )


    # ============================================================
    # STEP 9: Plot
    # ============================================================

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(24, 6)
    )


    # ------------------------------------------------------------
    # K-alpha area map
    # ------------------------------------------------------------

    im1 = ax[0].imshow(
        ka_area_map,
        cmap="viridis"
    )

    ax[0].set_title(
        "Fe Kα\nBackground-Subtracted Area"
    )

    ax[0].set_xlabel("Column (pixel)")
    ax[0].set_ylabel("Row (pixel)")

    fig.colorbar(
        im1,
        ax=ax[0],
        label="Integrated Counts"
    )


    # ------------------------------------------------------------
    # K-beta area map
    # ------------------------------------------------------------

    im2 = ax[1].imshow(
        kb_area_map,
        cmap="viridis"
    )

    ax[1].set_title(
        "Fe Kβ\nBackground-Subtracted Area"
    )

    ax[1].set_xlabel("Column (pixel)")
    ax[1].set_ylabel("Row (pixel)")

    fig.colorbar(
        im2,
        ax=ax[1],
        label="Integrated Counts"
    )


    # ------------------------------------------------------------
    # Ratio map
    # ------------------------------------------------------------

    ratio_cmap = plt.get_cmap(
        "seismic"
    ).copy()

    ratio_cmap.set_bad("white")

    im3 = ax[2].imshow(
        ratio_map,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax[2].set_title(
        "Fe Kα / Kβ Area Ratio"
    )

    ax[2].set_xlabel("Column (pixel)")
    ax[2].set_ylabel("Row (pixel)")

    cbar = fig.colorbar(
        im3,
        ax=ax[2],
        label="Kα Area / Kβ Area"
    )

    cbar.ax.axhline(
        median_ratio,
        color="black",
        linewidth=2
    )


    # ------------------------------------------------------------
    # Correlation scatter
    # ------------------------------------------------------------

    ka_flat = ka_area_map[
        valid_pixels
    ]

    kb_flat = kb_area_map[
        valid_pixels
    ]

    ratio_flat = ratio_map[
        valid_pixels
    ]


    scatter = ax[3].scatter(
        kb_flat,
        ka_flat,
        c=ratio_flat,
        cmap=ratio_cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.6,
        edgecolors="none"
    )

    ax[3].set_xlabel(
        "Fe Kβ Area"
    )

    ax[3].set_ylabel(
        "Fe Kα Area"
    )

    ax[3].set_title(
        "Fe Kα vs Kβ Area"
    )


    # Expected relationship from median ratio
    x_line = np.array([
        np.min(kb_flat),
        np.max(kb_flat)
    ])

    ax[3].plot(
        x_line,
        x_line * median_ratio,
        color="black",
        linestyle="--",
        linewidth=2,
        label=(
            f"Median ratio = "
            f"{median_ratio:.3f}"
        )
    )

    ax[3].legend()


    fig.colorbar(
        scatter,
        ax=ax[3],
        label="Kα / Kβ Area Ratio"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 10: Return results
    # ============================================================

    # return {
    #     "ka_area_map": ka_area_map,
    #     "kb_area_map": kb_area_map,
    #     "ratio_map": ratio_map,
    #     "valid_pixels": valid_pixels,
    #     "median_ratio": median_ratio,
    #     "mean_ratio": mean_ratio
    # }

def findFeRatioDyAnomalies(
    results,
    dy_count,
    dy_percentile=90,
    ratio_low_percentile=10,
    ratio_high_percentile=90,
    circle_radius=5,
    max_points=20
):
    """
    Find pixels with HIGH Dy and anomalous Fe K-alpha/K-beta ratios.

    Two anomaly classes are identified:

        1. HIGH ratio + HIGH Dy
        2. LOW ratio + HIGH Dy

    Parameters
    ----------
    results : dict
        Dictionary returned by checkFeRatio().

    dy_count : 2D numpy array
        Dy intensity map.

    dy_percentile : float
        Percentile above which Dy is considered high.

    ratio_low_percentile : float
        Percentile below which the Fe ratio is considered low.

    ratio_high_percentile : float
        Percentile above which the Fe ratio is considered high.

    circle_radius : float
        Radius of circles in pixels.

    max_points : int
        Maximum number of points to circle for each anomaly type.

    Returns
    -------
    anomalies : dict
        Coordinates and values of selected anomalous pixels.
    """

    # ============================================================
    # STEP 1: Extract Fe ratio map
    # ============================================================

    ratio_map = results["ratio_map"]

    if ratio_map.shape != dy_count.shape:
        raise ValueError(
            "ratio_map and dy_count must have the same shape."
        )


    # ============================================================
    # STEP 2: Valid pixels
    # ============================================================

    valid = (
        np.isfinite(ratio_map) &
        np.isfinite(dy_count)
    )

    ratio_values = ratio_map[valid]
    dy_values = dy_count[valid]


    if len(ratio_values) == 0:
        raise ValueError(
            "No valid pixels found."
        )


    # ============================================================
    # STEP 3: Calculate thresholds
    # ============================================================

    dy_threshold = np.percentile(
        dy_values,
        dy_percentile
    )

    ratio_low_threshold = np.percentile(
        ratio_values,
        ratio_low_percentile
    )

    ratio_high_threshold = np.percentile(
        ratio_values,
        ratio_high_percentile
    )


    print("\n" + "=" * 70)
    print("Fe RATIO / Dy ANOMALY SELECTION")
    print("=" * 70)

    print(
        f"High Dy threshold ({dy_percentile}th percentile): "
        f"{dy_threshold:.3f}"
    )

    print(
        f"Low Fe ratio threshold ({ratio_low_percentile}th percentile): "
        f"{ratio_low_threshold:.3f}"
    )

    print(
        f"High Fe ratio threshold ({ratio_high_percentile}th percentile): "
        f"{ratio_high_threshold:.3f}"
    )


    # ============================================================
    # STEP 4: Define anomaly masks
    # ============================================================

    # HIGH Dy + HIGH Fe ratio
    high_ratio_high_dy = (
        valid &
        (dy_count >= dy_threshold) &
        (ratio_map >= ratio_high_threshold)
    )


    # HIGH Dy + LOW Fe ratio
    low_ratio_high_dy = (
        valid &
        (dy_count >= dy_threshold) &
        (ratio_map <= ratio_low_threshold)
    )


    # ============================================================
    # STEP 5: Get coordinates
    # ============================================================

    high_y, high_x = np.where(
        high_ratio_high_dy
    )

    low_y, low_x = np.where(
        low_ratio_high_dy
    )


    # ============================================================
    # STEP 6: Sort by strength of anomaly
    # ============================================================

    # For high-ratio anomaly:
    #
    # Larger ratio = stronger anomaly
    #
    high_strength = ratio_map[
        high_ratio_high_dy
    ]

    high_order = np.argsort(
        high_strength
    )[::-1]

    high_x = high_x[high_order]
    high_y = high_y[high_order]


    # For low-ratio anomaly:
    #
    # Smaller ratio = stronger anomaly
    #
    low_strength = ratio_map[
        low_ratio_high_dy
    ]

    low_order = np.argsort(
        low_strength
    )

    low_x = low_x[low_order]
    low_y = low_y[low_order]


    # Limit number of displayed points
    high_x = high_x[:max_points]
    high_y = high_y[:max_points]

    low_x = low_x[:max_points]
    low_y = low_y[:max_points]


    # ============================================================
    # STEP 7: Print selected points
    # ============================================================

    print("\nHIGH Fe RATIO + HIGH Dy")
    print("-" * 70)

    for x, y in zip(high_x, high_y):

        print(
            f"Pixel (x, y)=({x}, {y}) | "
            f"Fe ratio={ratio_map[y, x]:.4f} | "
            f"Dy={dy_count[y, x]:.3f}"
        )


    print("\nLOW Fe RATIO + HIGH Dy")
    print("-" * 70)

    for x, y in zip(low_x, low_y):

        print(
            f"Pixel (x, y)=({x}, {y}) | "
            f"Fe ratio={ratio_map[y, x]:.4f} | "
            f"Dy={dy_count[y, x]:.3f}"
        )


    # ============================================================
    # STEP 8: Create Plot A
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    cmap = plt.get_cmap(
        "seismic"
    ).copy()

    cmap.set_bad("white")


    # Determine ratio color limits
    ratio_low = np.percentile(
        ratio_values,
        1
    )

    ratio_high = np.percentile(
        ratio_values,
        99
    )


    im = ax.imshow(
        ratio_map,
        cmap=cmap,
        vmin=ratio_low,
        vmax=ratio_high
    )


    ax.set_title(
        "Fe K-alpha / K-beta Ratio\n"
        "High-Dy Anomalous Pixels",
        fontsize=14
    )

    ax.set_xlabel(
        "X / Column (pixel)"
    )

    ax.set_ylabel(
        "Y / Row (pixel)"
    )


    fig.colorbar(
        im,
        ax=ax,
        label="Fe K-alpha / K-beta Ratio"
    )


    # ============================================================
    # STEP 9: Circle HIGH ratio + HIGH Dy points
    # ============================================================

    for x, y in zip(high_x, high_y):

        circle = Circle(
            (x, y),
            radius=circle_radius,
            fill=False,
            edgecolor="lime",
            linewidth=2.5
        )

        ax.add_patch(circle)


    # ============================================================
    # STEP 10: Circle LOW ratio + HIGH Dy points
    # ============================================================

    for x, y in zip(low_x, low_y):

        circle = Circle(
            (x, y),
            radius=circle_radius,
            fill=False,
            edgecolor="yellow",
            linewidth=2.5
        )

        ax.add_patch(circle)


    # ============================================================
    # STEP 11: Legend
    # ============================================================

    high_handle = Circle(
        (0, 0),
        radius=5,
        fill=False,
        edgecolor="lime",
        linewidth=2.5,
        label="High ratio + High Dy"
    )

    low_handle = Circle(
        (0, 0),
        radius=5,
        fill=False,
        edgecolor="yellow",
        linewidth=2.5,
        label="Low ratio + High Dy"
    )

    ax.legend(
        handles=[
            high_handle,
            low_handle
        ],
        loc="upper right"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 12: Return results
    # ============================================================

    anomalies = {

        "high_ratio_high_dy": {
            "x": high_x,
            "y": high_y,
            "ratio": ratio_map[
                high_y,
                high_x
            ],
            "dy": dy_count[
                high_y,
                high_x
            ]
        },

        "low_ratio_high_dy": {
            "x": low_x,
            "y": low_y,
            "ratio": ratio_map[
                low_y,
                low_x
            ],
            "dy": dy_count[
                low_y,
                low_x
            ]
        },

        "thresholds": {
            "dy": dy_threshold,
            "ratio_low": ratio_low_threshold,
            "ratio_high": ratio_high_threshold
        }
    }


    return anomalies

def checkFeKaFWHMvsDy(
    counts,
    dy_map,
    fe_ka_energy=6.40,
    window_width=0.15,
    min_peak_height=10,
    min_prominence=5
):
    """
    Calculate Fe K-alpha FWHM at every pixel and plot:

        1. Fe K-alpha FWHM map
        2. Fe K-alpha FWHM vs apparent Dy

    Returns
    -------
    results : dict
    """

    energies = np.arange(counts.shape[-1]) / 100.0

    nrows, ncols = counts.shape[:2]

    ka_indices = np.where(
        (energies >= fe_ka_energy - window_width) &
        (energies <= fe_ka_energy + window_width)
    )[0]

    ka_fwhm_map = np.full(
        (nrows, ncols),
        np.nan
    )

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
                y,
                height=min_peak_height,
                prominence=min_prominence
            )

            if len(peaks) == 0:
                continue

            # Strongest peak
            peak = peaks[
                np.argmax(y[peaks])
            ]

            widths, _, _, _ = peak_widths(
                y,
                [peak],
                rel_height=0.5
            )

            energy_step = np.mean(
                np.diff(
                    energies[ka_indices]
                )
            )

            ka_fwhm_map[i, j] = (
                widths[0] * energy_step
            )

    valid = (
        np.isfinite(ka_fwhm_map) &
        np.isfinite(dy_map) &
        (dy_map > 0)
    )

    fwhm_flat = ka_fwhm_map[valid]
    dy_flat = dy_map[valid]

    print("\n========== Fe K-alpha FWHM vs Dy ==========")
    print(f"Valid pixels: {len(fwhm_flat):,}")

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    im = ax[0].imshow(
        ka_fwhm_map,
        cmap="viridis"
    )

    ax[0].set_title(
        "Fe K-alpha FWHM"
    )

    fig.colorbar(
        im,
        ax=ax[0],
        label="FWHM (keV)"
    )

    ax[1].scatter(
        fwhm_flat,
        dy_flat,
        alpha=0.5,
        edgecolors="none"
    )

    ax[1].set_xlabel(
        "Fe K-alpha FWHM (keV)"
    )

    ax[1].set_ylabel(
        "Apparent Dy intensity"
    )

    ax[1].set_title(
        "Fe K-alpha FWHM vs Apparent Dy"
    )

    plt.tight_layout()
    plt.show()

    # return {
    #     "ka_fwhm_map": ka_fwhm_map,
    #     "fwhm_flat": fwhm_flat,
    #     "dy_flat": dy_flat,
    #     "valid_pixels": valid
    # }

def checkFeFWHMRatioVsDy(
    counts,
    dy_map,
    fe_ka_energy=6.40,
    fe_kb_energy=7.06,
    window_width=0.15,
    min_peak_height=0,
    min_prominence=3
):
    """
    Calculate:

        FWHM(K-alpha) / FWHM(K-beta)

    and plot:

        1. K-alpha FWHM map
        2. K-beta FWHM map
        3. FWHM ratio map
        4. FWHM ratio vs apparent Dy
    """

    energies = np.arange(
        counts.shape[-1]
    ) / 100.0

    nrows, ncols = counts.shape[:2]

    def get_fwhm(spectrum, center):

        indices = np.where(
            (energies >= center - window_width) &
            (energies <= center + window_width)
        )[0]

        y = spectrum[indices].astype(float)

        if len(y) < 5:
            return np.nan

        n_edge = max(
            2,
            len(y) // 10
        )

        baseline = np.median(
            np.concatenate([
                y[:n_edge],
                y[-n_edge:]
            ])
        )

        y = y - baseline
        y[y < 0] = 0

        peaks, properties = find_peaks(
            y,
            height=min_peak_height,
            prominence=min_prominence
        )

        if len(peaks) == 0:
            return np.nan

        peak = peaks[
            np.argmax(y[peaks])
        ]

        widths, _, _, _ = peak_widths(
            y,
            [peak],
            rel_height=0.5
        )

        energy_step = np.mean(
            np.diff(
                energies[indices]
            )
        )

        return widths[0] * energy_step

    ka_fwhm_map = np.full(
        (nrows, ncols),
        np.nan
    )

    kb_fwhm_map = np.full(
        (nrows, ncols),
        np.nan
    )

    for i in range(nrows):

        for j in range(ncols):

            spectrum = counts[i, j, :]

            ka_fwhm_map[i, j] = get_fwhm(
                spectrum,
                fe_ka_energy
            )

            kb_fwhm_map[i, j] = get_fwhm(
                spectrum,
                fe_kb_energy
            )

    valid = (
        np.isfinite(ka_fwhm_map) &
        np.isfinite(kb_fwhm_map) &
        (kb_fwhm_map > 0) &
        np.isfinite(dy_map)
    )

    fwhm_ratio_map = np.full(
        (nrows, ncols),
        np.nan
    )

    fwhm_ratio_map[valid] = (
        ka_fwhm_map[valid] /
        kb_fwhm_map[valid]
    )

    ratio_flat = fwhm_ratio_map[valid]
    dy_flat = dy_map[valid]

    median_ratio = np.median(
        ratio_flat
    )

    print(
        "\n========== Fe K-alpha/K-beta FWHM RATIO =========="
    )

    print(
        f"Valid pixels: {len(ratio_flat):,}"
    )

    print(
        f"Median ratio: {median_ratio:.4f}"
    )

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(24, 6)
    )

    im1 = ax[0].imshow(
        ka_fwhm_map,
        cmap="viridis"
    )

    ax[0].set_title(
        "Fe K-alpha FWHM"
    )

    fig.colorbar(
        im1,
        ax=ax[0],
        label="FWHM (keV)"
    )

    im2 = ax[1].imshow(
        kb_fwhm_map,
        cmap="viridis"
    )

    ax[1].set_title(
        "Fe K-beta FWHM"
    )

    fig.colorbar(
        im2,
        ax=ax[1],
        label="FWHM (keV)"
    )

    cmap = plt.get_cmap(
        "seismic"
    ).copy()

    cmap.set_bad("white")

    limit = np.percentile(
        np.abs(ratio_flat - median_ratio),
        99
    )

    im3 = ax[2].imshow(
        fwhm_ratio_map,
        cmap=cmap,
        vmin=median_ratio - limit,
        vmax=median_ratio + limit
    )

    ax[2].set_title(
        "K-alpha / K-beta FWHM Ratio"
    )

    fig.colorbar(
        im3,
        ax=ax[2],
        label="FWHM Ratio"
    )

    ax[3].scatter(
        ratio_flat,
        dy_flat,
        alpha=0.5,
        edgecolors="none"
    )

    ax[3].set_xlabel(
        "K-alpha / K-beta FWHM"
    )

    ax[3].set_ylabel(
        "Apparent Dy intensity"
    )

    ax[3].set_title(
        "FWHM Ratio vs Apparent Dy"
    )

    plt.tight_layout()
    plt.show()

    # return {
    #     "ka_fwhm_map": ka_fwhm_map,
    #     "kb_fwhm_map": kb_fwhm_map,
    #     "fwhm_ratio_map": fwhm_ratio_map,
    #     "ratio_flat": ratio_flat,
    #     "dy_flat": dy_flat,
    #     "valid_pixels": valid,
    #     "median_ratio": median_ratio
    # }

def checkFeKaAsymmetryVsDy(
    counts,
    dy_map,
    fe_ka_energy=6.40,
    window_width=0.15,
    min_peak_height=10,
    min_prominence=5
):
    """
    Calculate Fe K-alpha peak asymmetry and compare it with
    apparent Dy.

    Asymmetry:

        A = (FWHM_right - FWHM_left)
            / (FWHM_right + FWHM_left)

    Positive A:
        Peak is more extended toward higher energy.

    Negative A:
        Peak is more extended toward lower energy.
    """

    energies = np.arange(
        counts.shape[-1]
    ) / 100.0

    nrows, ncols = counts.shape[:2]

    ka_indices = np.where(
        (energies >= fe_ka_energy - window_width) &
        (energies <= fe_ka_energy + window_width)
    )[0]

    asymmetry_map = np.full(
        (nrows, ncols),
        np.nan
    )

    for i in range(nrows):

        for j in range(ncols):

            spectrum = counts[i, j, :].astype(float)

            y = spectrum[ka_indices]

            n_edge = max(
                2,
                len(y) // 10
            )

            baseline = np.median(
                np.concatenate([
                    y[:n_edge],
                    y[-n_edge:]
                ])
            )

            y = y - baseline
            y[y < 0] = 0

            peaks, properties = find_peaks(
                y,
                height=min_peak_height,
                prominence=min_prominence
            )

            if len(peaks) == 0:
                continue

            peak = peaks[
                np.argmax(y[peaks])
            ]

            widths, width_heights, left_ips, right_ips = (
                peak_widths(
                    y,
                    [peak],
                    rel_height=0.5
                )
            )

            left_ip = left_ips[0]
            right_ip = right_ips[0]

            left_width = (
                peak - left_ip
            )

            right_width = (
                right_ip - peak
            )

            denominator = (
                left_width +
                right_width
            )

            if denominator <= 0:
                continue

            asymmetry_map[i, j] = (
                (right_width - left_width) /
                denominator
            )

    valid = (
        np.isfinite(asymmetry_map) &
        np.isfinite(dy_map)
    )

    asym_flat = asymmetry_map[valid]
    dy_flat = dy_map[valid]

    print(
        "\n========== Fe K-alpha ASYMMETRY vs Dy =========="
    )

    print(
        f"Valid pixels: {len(asym_flat):,}"
    )

    print(
        f"Median asymmetry: {np.median(asym_flat):.5f}"
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    cmap = plt.get_cmap(
        "seismic"
    ).copy()

    cmap.set_bad("white")

    limit = np.percentile(
        np.abs(asym_flat),
        99
    )

    im = ax[0].imshow(
        asymmetry_map,
        cmap=cmap,
        vmin=-limit,
        vmax=limit
    )

    ax[0].set_title(
        "Fe K-alpha Peak Asymmetry"
    )

    ax[0].set_xlabel(
        "Column (pixel)"
    )

    ax[0].set_ylabel(
        "Row (pixel)"
    )

    fig.colorbar(
        im,
        ax=ax[0],
        label="Asymmetry"
    )

    ax[1].scatter(
        asym_flat,
        dy_flat,
        alpha=0.5,
        edgecolors="none"
    )

    ax[1].axvline(
        0,
        color="black",
        linestyle="--"
    )

    ax[1].set_xlabel(
        "Fe K-alpha Asymmetry"
    )

    ax[1].set_ylabel(
        "Apparent Dy intensity"
    )

    ax[1].set_title(
        "K-alpha Asymmetry vs Apparent Dy"
    )

    plt.tight_layout()
    plt.show()

    # return {
    #     "asymmetry_map": asymmetry_map,
    #     "asymmetry_flat": asym_flat,
    #     "dy_flat": dy_flat,
    #     "valid_pixels": valid
    # }


