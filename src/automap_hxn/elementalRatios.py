import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk
)

def checkElementalRatio(element1_counts, element2_counts, element1_name="Element 1", element2_name="Element 2"):

 
    # --- STEP 1: Calculate the Ratio Safely ---
    # Filter out low-count background pixels to avoid division-by-zero or massive noise
    min_counts_threshold = 10
    valid_pixels = (element1_counts > min_counts_threshold) & (element2_counts > min_counts_threshold)

    ratio_map = np.full_like(element1_counts, np.nan) # Fill background with NaN so it hides in the plot
    ratio_map[valid_pixels] = element1_counts[valid_pixels] / element2_counts[valid_pixels]

    # Find median ratio of valid pixels to center our diverging colormap
    median_ratio = np.nanmedian(ratio_map)

    # --- STEP 2: Plotting ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: 2D Spatial Ratio Map
    # Using a diverging colormap 'seismic' centered on the median ratio
    vmax_dist = max(np.nanmax(ratio_map) - median_ratio, median_ratio - np.nanmin(ratio_map))
    im = ax[0].imshow(ratio_map, cmap='seismic', 
                    vmin=median_ratio - vmax_dist*0.5, vmax=median_ratio + vmax_dist*0.5)
    ax[0].set_title(f"Spatial Map: {element1_name} / {element2_name} Ratio\n(Look for localized color clusters)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax[0], label='Ratio Value')
    cbar.ax.axhline(median_ratio, color='black', lw=2) # Line showing expected ratio

    # Chart 2: Correlation Scatter Plot
    # Flatten the 2D arrays to 1D for scattering
    ka_flat = element1_counts[valid_pixels]
    kb_flat = element2_counts[valid_pixels]
    ratio_flat = ratio_map[valid_pixels]

    scatter = ax[1].scatter(kb_flat, ka_flat, c=ratio_flat, cmap='seismic', 
                            vmin=median_ratio - vmax_dist*0.5, vmax=median_ratio + vmax_dist*0.5, alpha=0.6, edgecolors='none')
    ax[1].set_xlabel(f"{element2_name} Intensity (Counts)")
    ax[1].set_ylabel(f"{element1_name} Intensity (Counts)")
    ax[1].set_title(f"Scatter Plot: {element1_name} vs {element2_name}\n(Look for outliers off the main linear trend)", fontsize=12)

    # Draw the theoretical expectation line based on the median
    x_line = np.array([np.min(kb_flat), np.max(kb_flat)])
    ax[1].plot(x_line, x_line * median_ratio, color='black', linestyle='--', label=f'Expected Linear Trend ({median_ratio:.2f}x)')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def checkDyResidual(
    fe_counts,
    dy_counts,
    fe_name="Fe",
    dy_name="Dy",
    min_fe_counts=10,
    percentile=99
):
    """
    Fit a robust regression model:

        Dy_predicted = f(Fe)

    and calculate:

        Dy_residual = Dy_measured - Dy_predicted

    Positive residual:
        More apparent Dy than expected from Fe.

    Negative residual:
        Less apparent Dy than expected from Fe.
    """

    # ============================================================
    # STEP 1: Select valid pixels
    # ============================================================

    # Important:
    # Do NOT require Dy > threshold because low/zero Dy pixels are
    # scientifically important for this problem.
    valid_pixels = (
        np.isfinite(fe_counts) &
        np.isfinite(dy_counts) &
        (fe_counts > min_fe_counts)
    )

    fe_flat = fe_counts[valid_pixels].astype(float)
    dy_flat = dy_counts[valid_pixels].astype(float)

    if len(fe_flat) < 10:
        raise ValueError("Not enough valid pixels for regression.")


    # ============================================================
    # STEP 2: Fit robust Fe -> Dy regression
    # ============================================================

    # X must be 2D for sklearn
    X = fe_flat.reshape(-1, 1)

    # Scaling Fe helps numerical stability because XRF counts
    # can be very large.
    model = make_pipeline(
        StandardScaler(),
        HuberRegressor()
    )

    model.fit(X, dy_flat)

    # Predicted Dy based only on Fe
    dy_pred = model.predict(X)


    # ============================================================
    # STEP 3: Calculate Dy residual
    # ============================================================

    residual_flat = dy_flat - dy_pred

    residual_map = np.full(
        fe_counts.shape,
        np.nan,
        dtype=float
    )

    residual_map[valid_pixels] = residual_flat


    # ============================================================
    # STEP 4: Choose symmetric color limits around zero
    # ============================================================

    # Use percentile rather than absolute maximum so that one
    # extreme outlier does not destroy the color scale.
    residual_limit = np.nanpercentile(
        np.abs(residual_flat),
        percentile
    )

    if residual_limit == 0:
        residual_limit = 1


    # ============================================================
    # STEP 5: Plot
    # ============================================================

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )


    # ------------------------------------------------------------
    # Plot 1: Spatial Dy residual map
    # ------------------------------------------------------------

    im = ax[0].imshow(
        residual_map,
        cmap="seismic",
        vmin=-residual_limit,
        vmax=residual_limit
    )

    ax[0].set_title(
        f"Spatial Map: {dy_name} Residual\n"
        f"{dy_name} measured - {dy_name} predicted from {fe_name}",
        fontsize=12
    )

    cbar = fig.colorbar(
        im,
        ax=ax[0],
        label=f"{dy_name} Residual (counts)"
    )

    # Mark zero residual on the colorbar
    cbar.ax.axhline(
        0,
        color="black",
        linewidth=2
    )


    # ------------------------------------------------------------
    # Plot 2: Fe vs Dy scatter colored by residual
    # ------------------------------------------------------------

    scatter = ax[1].scatter(
        fe_flat,
        dy_flat,
        c=residual_flat,
        cmap="seismic",
        vmin=-residual_limit,
        vmax=residual_limit,
        alpha=0.6,
        edgecolors="none"
    )

    ax[1].set_xlabel(
        f"{fe_name} Intensity (Counts)"
    )

    ax[1].set_ylabel(
        f"{dy_name} Intensity (Counts)"
    )

    ax[1].set_title(
        f"{dy_name} vs {fe_name}\n"
        f"Colored by {dy_name} residual",
        fontsize=12
    )


    # ============================================================
    # STEP 6: Draw fitted regression trend
    # ============================================================

    fe_line = np.linspace(
        np.min(fe_flat),
        np.max(fe_flat),
        500
    )

    dy_line = model.predict(
        fe_line.reshape(-1, 1)
    )

    ax[1].plot(
        fe_line,
        dy_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Robust {fe_name} → {dy_name} trend"
    )

    ax[1].legend()


    # Add residual colorbar for scatter plot
    fig.colorbar(
        scatter,
        ax=ax[1],
        label=f"{dy_name} Residual (counts)"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 7: Print useful statistics
    # ============================================================

    print("\n========== Dy RESIDUAL ANALYSIS ==========")
    print(f"Valid pixels: {len(fe_flat):,}")
    print(
        f"Median residual: "
        f"{np.median(residual_flat):.2f} counts"
    )
    print(
        f"Residual standard deviation: "
        f"{np.std(residual_flat):.2f} counts"
    )
    print(
        f"Color scale: "
        f"{-residual_limit:.2f} to {residual_limit:.2f}"
    )


    # ============================================================
    # STEP 8: Return results for later autonomous selection
    # ============================================================

    return {
        "model": model,
        "residual_map": residual_map,
        "residual_flat": residual_flat,
        "predicted_dy": dy_pred,
        "fe": fe_flat,
        "dy": dy_flat,
        "valid_pixels": valid_pixels
    }


def checkDyResidualStandardized(
    fe_counts,
    dy_counts,
    fe_name="Fe",
    dy_name="Dy",
    min_fe_counts=10,
    n_bins=20,
    min_points_per_bin=20,
    color_percentile=99
):
    """
    Fit a robust Fe -> Dy regression model and calculate both:

        raw residual = Dy_measured - Dy_predicted

    and

        standardized residual =
            raw residual / local residual spread

    The local residual spread is estimated as a function of Fe intensity
    using the robust Median Absolute Deviation (MAD).

    Interpretation:
        standardized residual > 0
            More Dy than expected for that Fe intensity.

        standardized residual ~ 0
            Dy follows the normal Fe-Dy relationship.

        standardized residual < 0
            Less Dy than expected for that Fe intensity.

    Large positive standardized residuals are potential candidates
    for autonomous Dy XANES selection.
    """


    # ============================================================
    # STEP 1: Select valid pixels
    # ============================================================

    # Do NOT threshold Dy.
    #
    # Low-Dy / high-Fe pixels are important negative/control
    # observations for this experiment.

    valid_pixels = (
        np.isfinite(fe_counts) &
        np.isfinite(dy_counts) &
        (fe_counts > min_fe_counts)
    )

    fe_flat = fe_counts[valid_pixels].astype(float)
    dy_flat = dy_counts[valid_pixels].astype(float)

    if len(fe_flat) < 50:
        raise ValueError(
            "Too few valid pixels for reliable regression "
            "and residual standardization."
        )


    # ============================================================
    # STEP 2: Fit robust Fe -> Dy regression
    # ============================================================

    X = fe_flat.reshape(-1, 1)

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor()
    )

    model.fit(X, dy_flat)

    # Expected apparent Dy based on Fe
    dy_pred = model.predict(X)


    # ============================================================
    # STEP 3: Calculate RAW Dy residual
    # ============================================================

    raw_residual = dy_flat - dy_pred


    # ============================================================
    # STEP 4: Estimate LOCAL residual variability
    # ============================================================
    #
    # We do NOT want:
    #
    #     residual / global_std
    #
    # because the scatter may get wider as Fe increases.
    #
    # Instead:
    #
    #     residual / local_std(Fe)
    #
    # First divide Fe values into quantile bins. Quantile bins are
    # useful because they try to put similar numbers of pixels into
    # each Fe interval.
    # ============================================================

    quantiles = np.linspace(0, 1, n_bins + 1)

    bin_edges = np.quantile(
        fe_flat,
        quantiles
    )

    # Remove duplicate bin boundaries
    # (can happen when many pixels have identical Fe counts)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        raise ValueError(
            "Not enough unique Fe values to estimate "
            "local residual variability."
        )


    bin_centers = []
    bin_scales = []


    # ============================================================
    # STEP 5: Calculate robust scatter in each Fe bin
    # ============================================================

    for i in range(len(bin_edges) - 1):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Include right edge for the final bin
        if i == len(bin_edges) - 2:
            mask = (
                (fe_flat >= lower) &
                (fe_flat <= upper)
            )
        else:
            mask = (
                (fe_flat >= lower) &
                (fe_flat < upper)
            )


        if np.sum(mask) < min_points_per_bin:
            continue


        residual_bin = raw_residual[mask]
        fe_bin = fe_flat[mask]


        # --------------------------------------------------------
        # Robust residual spread using MAD
        # --------------------------------------------------------
        #
        # MAD = median(|x - median(x)|)
        #
        # 1.4826 converts MAD to approximately standard deviation
        # for normally distributed residuals.
        # --------------------------------------------------------

        median_residual = np.median(residual_bin)

        mad = np.median(
            np.abs(
                residual_bin - median_residual
            )
        )

        robust_sigma = 1.4826 * mad


        # Ignore bins with zero/invalid variability
        if (
            np.isfinite(robust_sigma) and
            robust_sigma > 0
        ):
            bin_centers.append(
                np.median(fe_bin)
            )

            bin_scales.append(
                robust_sigma
            )


    bin_centers = np.asarray(bin_centers)
    bin_scales = np.asarray(bin_scales)


    # ============================================================
    # STEP 6: Fall back to global robust scatter if necessary
    # ============================================================

    if len(bin_centers) < 2:

        median_residual = np.median(raw_residual)

        global_mad = np.median(
            np.abs(
                raw_residual - median_residual
            )
        )

        global_sigma = 1.4826 * global_mad

        if global_sigma <= 0:
            global_sigma = np.std(raw_residual)

        local_sigma = np.full(
            len(raw_residual),
            global_sigma
        )

    else:

        # --------------------------------------------------------
        # Interpolate the estimated variability across Fe
        # --------------------------------------------------------

        local_sigma = np.interp(
            fe_flat,
            bin_centers,
            bin_scales,
            left=bin_scales[0],
            right=bin_scales[-1]
        )


    # Avoid division by zero
    minimum_sigma = np.finfo(float).eps

    local_sigma = np.maximum(
        local_sigma,
        minimum_sigma
    )


    # ============================================================
    # STEP 7: Calculate STANDARDIZED residual
    # ============================================================

    standardized_residual = (
        raw_residual / local_sigma
    )


    # ============================================================
    # STEP 8: Put results back into spatial maps
    # ============================================================

    raw_residual_map = np.full(
        fe_counts.shape,
        np.nan,
        dtype=float
    )

    standardized_residual_map = np.full(
        fe_counts.shape,
        np.nan,
        dtype=float
    )


    raw_residual_map[valid_pixels] = (
        raw_residual
    )

    standardized_residual_map[valid_pixels] = (
        standardized_residual
    )


    # ============================================================
    # STEP 9: Set symmetric color scale around zero
    # ============================================================

    z_limit = np.nanpercentile(
        np.abs(standardized_residual),
        color_percentile
    )

    if not np.isfinite(z_limit) or z_limit == 0:
        z_limit = 1


    # ============================================================
    # STEP 10: Plot
    # ============================================================

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )


    # ------------------------------------------------------------
    # Plot 1: Spatial standardized Dy residual
    # ------------------------------------------------------------

    im = ax[0].imshow(
        standardized_residual_map,
        cmap="seismic",
        vmin=-z_limit,
        vmax=z_limit
    )

    ax[0].set_title(
        f"Spatial Map: Standardized {dy_name} Residual\n"
        f"Excess {dy_name} after accounting for {fe_name}",
        fontsize=12
    )

    cbar = fig.colorbar(
        im,
        ax=ax[0],
        label="Standardized Residual (z-like score)"
    )

    cbar.ax.axhline(
        0,
        color="black",
        linewidth=2
    )


    # ------------------------------------------------------------
    # Plot 2: Fe vs Dy colored by standardized residual
    # ------------------------------------------------------------

    scatter = ax[1].scatter(
        fe_flat,
        dy_flat,
        c=standardized_residual,
        cmap="seismic",
        vmin=-z_limit,
        vmax=z_limit,
        alpha=0.6,
        edgecolors="none"
    )

    ax[1].set_xlabel(
        f"{fe_name} Intensity (Counts)"
    )

    ax[1].set_ylabel(
        f"{dy_name} Intensity (Counts)"
    )

    ax[1].set_title(
        f"{dy_name} vs {fe_name}\n"
        f"Colored by Standardized {dy_name} Residual",
        fontsize=12
    )


    # ============================================================
    # STEP 11: Plot fitted Fe -> Dy relationship
    # ============================================================

    fe_line = np.linspace(
        np.min(fe_flat),
        np.max(fe_flat),
        500
    )

    dy_line = model.predict(
        fe_line.reshape(-1, 1)
    )

    ax[1].plot(
        fe_line,
        dy_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Robust {fe_name} → {dy_name} trend"
    )

    ax[1].legend()


    fig.colorbar(
        scatter,
        ax=ax[1],
        label="Standardized Residual (z-like score)"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 12: Print summary
    # ============================================================

    print("\n========== STANDARDIZED Dy RESIDUAL ANALYSIS ==========")

    print(
        f"Valid pixels: {len(fe_flat):,}"
    )

    print(
        f"Median raw residual: "
        f"{np.median(raw_residual):.3f} counts"
    )

    print(
        f"Median local residual sigma: "
        f"{np.median(local_sigma):.3f} counts"
    )

    print(
        f"Median standardized residual: "
        f"{np.median(standardized_residual):.3f}"
    )

    print(
        f"99th percentile |standardized residual|: "
        f"{np.percentile(np.abs(standardized_residual), 99):.3f}"
    )

    print(
        f"Pixels with standardized residual > 2: "
        f"{np.sum(standardized_residual > 2):,}"
    )

    print(
        f"Pixels with standardized residual > 3: "
        f"{np.sum(standardized_residual > 3):,}"
    )


    # ============================================================
    # STEP 13: Return everything for autonomous point selection
    # ============================================================

    return {

        "model":
            model,

        "raw_residual_map":
            raw_residual_map,

        "standardized_residual_map":
            standardized_residual_map,

        "raw_residual":
            raw_residual,

        "standardized_residual":
            standardized_residual,

        "local_sigma":
            local_sigma,

        "predicted_dy":
            dy_pred,

        "fe":
            fe_flat,

        "dy":
            dy_flat,

        "valid_pixels":
            valid_pixels,

        "bin_centers":
            bin_centers,

        "bin_scales":
            bin_scales
    }


def checkDyResidualStandardizedMasked(
    fe_counts,
    dy_counts,
    fe_name="Fe",
    dy_name="Dy",
    particle_mask=None,
    min_fe_counts=0,
    n_bins=20,
    min_points_per_bin=20,
    color_percentile=99
):
    """
    Fit a robust Fe -> Dy regression model and calculate both:

        raw residual = Dy_measured - Dy_predicted

    and

        standardized residual =
            raw residual / local residual spread

    The local residual spread is estimated as a function of Fe intensity
    using the robust Median Absolute Deviation (MAD).

    Interpretation:
        standardized residual > 0
            More Dy than expected for that Fe intensity.

        standardized residual ~ 0
            Dy follows the normal Fe-Dy relationship.

        standardized residual < 0
            Less Dy than expected for that Fe intensity.

    Large positive standardized residuals are potential candidates
    for autonomous Dy XANES selection.
    """


    # ============================================================
    # STEP 1: Select valid pixels
    # ============================================================

    # Do NOT threshold Dy.
    #
    # Low-Dy / high-Fe pixels are important negative/control
    # observations for this experiment.

    valid_pixels = (
        np.isfinite(fe_counts) &
        np.isfinite(dy_counts)
    )


    # Apply particle mask
    if particle_mask is not None:

        if particle_mask.shape != fe_counts.shape:
            raise ValueError(
                "particle_mask must have the same shape "
                "as fe_counts and dy_counts."
            )

        valid_pixels &= particle_mask


    # Optional Fe threshold INSIDE particles
    if min_fe_counts is not None:

        valid_pixels &= (
            fe_counts >= min_fe_counts
        )

    fe_flat = fe_counts[valid_pixels].astype(float)
    dy_flat = dy_counts[valid_pixels].astype(float)

    if len(fe_flat) < 50:
        raise ValueError(
            "Too few valid pixels for reliable regression "
            "and residual standardization."
        )


    # ============================================================
    # STEP 2: Fit robust Fe -> Dy regression
    # ============================================================

    X = fe_flat.reshape(-1, 1)

    model = make_pipeline(
        StandardScaler(),
        HuberRegressor()
    )

    model.fit(X, dy_flat)

    # Expected apparent Dy based on Fe
    dy_pred = model.predict(X)


    # ============================================================
    # STEP 3: Calculate RAW Dy residual
    # ============================================================

    raw_residual = dy_flat - dy_pred


    # ============================================================
    # STEP 4: Estimate LOCAL residual variability
    # ============================================================
    #
    # We do NOT want:
    #
    #     residual / global_std
    #
    # because the scatter may get wider as Fe increases.
    #
    # Instead:
    #
    #     residual / local_std(Fe)
    #
    # First divide Fe values into quantile bins. Quantile bins are
    # useful because they try to put similar numbers of pixels into
    # each Fe interval.
    # ============================================================

    quantiles = np.linspace(0, 1, n_bins + 1)

    bin_edges = np.quantile(
        fe_flat,
        quantiles
    )

    # Remove duplicate bin boundaries
    # (can happen when many pixels have identical Fe counts)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        raise ValueError(
            "Not enough unique Fe values to estimate "
            "local residual variability."
        )


    bin_centers = []
    bin_scales = []


    # ============================================================
    # STEP 5: Calculate robust scatter in each Fe bin
    # ============================================================

    for i in range(len(bin_edges) - 1):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Include right edge for the final bin
        if i == len(bin_edges) - 2:
            mask = (
                (fe_flat >= lower) &
                (fe_flat <= upper)
            )
        else:
            mask = (
                (fe_flat >= lower) &
                (fe_flat < upper)
            )


        if np.sum(mask) < min_points_per_bin:
            continue


        residual_bin = raw_residual[mask]
        fe_bin = fe_flat[mask]


        # --------------------------------------------------------
        # Robust residual spread using MAD
        # --------------------------------------------------------
        #
        # MAD = median(|x - median(x)|)
        #
        # 1.4826 converts MAD to approximately standard deviation
        # for normally distributed residuals.
        # --------------------------------------------------------

        median_residual = np.median(residual_bin)

        mad = np.median(
            np.abs(
                residual_bin - median_residual
            )
        )

        robust_sigma = 1.4826 * mad


        # Ignore bins with zero/invalid variability
        if (
            np.isfinite(robust_sigma) and
            robust_sigma > 0
        ):
            bin_centers.append(
                np.median(fe_bin)
            )

            bin_scales.append(
                robust_sigma
            )


    bin_centers = np.asarray(bin_centers)
    bin_scales = np.asarray(bin_scales)


    # ============================================================
    # STEP 6: Fall back to global robust scatter if necessary
    # ============================================================

    if len(bin_centers) < 2:

        median_residual = np.median(raw_residual)

        global_mad = np.median(
            np.abs(
                raw_residual - median_residual
            )
        )

        global_sigma = 1.4826 * global_mad

        if global_sigma <= 0:
            global_sigma = np.std(raw_residual)

        local_sigma = np.full(
            len(raw_residual),
            global_sigma
        )

    else:

        # --------------------------------------------------------
        # Interpolate the estimated variability across Fe
        # --------------------------------------------------------

        local_sigma = np.interp(
            fe_flat,
            bin_centers,
            bin_scales,
            left=bin_scales[0],
            right=bin_scales[-1]
        )


    # Avoid division by zero
    minimum_sigma = np.finfo(float).eps

    local_sigma = np.maximum(
        local_sigma,
        minimum_sigma
    )


    # ============================================================
    # STEP 7: Calculate STANDARDIZED residual
    # ============================================================

    standardized_residual = (
        raw_residual / local_sigma
    )


    # ============================================================
    # STEP 8: Put results back into spatial maps
    # ============================================================

    raw_residual_map = np.full(
        fe_counts.shape,
        np.nan,
        dtype=float
    )

    standardized_residual_map = np.full(
        fe_counts.shape,
        np.nan,
        dtype=float
    )


    raw_residual_map[valid_pixels] = (
        raw_residual
    )

    standardized_residual_map[valid_pixels] = (
        standardized_residual
    )


    # ============================================================
    # STEP 9: Set symmetric color scale around zero
    # ============================================================

    z_limit = np.nanpercentile(
        np.abs(standardized_residual),
        color_percentile
    )

    if not np.isfinite(z_limit) or z_limit == 0:
        z_limit = 1


    # ============================================================
    # STEP 10: Plot
    # ============================================================

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )


    # ------------------------------------------------------------
    # Plot 1: Spatial standardized Dy residual
    # ------------------------------------------------------------

    im = ax[0].imshow(
        standardized_residual_map,
        cmap="seismic",
        vmin=-z_limit,
        vmax=z_limit
    )

    ax[0].set_title(
        f"Spatial Map: Standardized {dy_name} Residual\n"
        f"Excess {dy_name} after accounting for {fe_name}",
        fontsize=12
    )

    cbar = fig.colorbar(
        im,
        ax=ax[0],
        label="Standardized Residual (z-like score)"
    )

    cbar.ax.axhline(
        0,
        color="black",
        linewidth=2
    )


    # ------------------------------------------------------------
    # Plot 2: Fe vs Dy colored by standardized residual
    # ------------------------------------------------------------

    scatter = ax[1].scatter(
        fe_flat,
        dy_flat,
        c=standardized_residual,
        cmap="seismic",
        vmin=-z_limit,
        vmax=z_limit,
        alpha=0.6,
        edgecolors="none"
    )

    ax[1].set_xlabel(
        f"{fe_name} Intensity (Counts)"
    )

    ax[1].set_ylabel(
        f"{dy_name} Intensity (Counts)"
    )

    ax[1].set_title(
        f"{dy_name} vs {fe_name}\n"
        f"Colored by Standardized {dy_name} Residual",
        fontsize=12
    )


    # ============================================================
    # STEP 11: Plot fitted Fe -> Dy relationship
    # ============================================================

    fe_line = np.linspace(
        np.min(fe_flat),
        np.max(fe_flat),
        500
    )

    dy_line = model.predict(
        fe_line.reshape(-1, 1)
    )

    ax[1].plot(
        fe_line,
        dy_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Robust {fe_name} → {dy_name} trend"
    )

    ax[1].legend()


    fig.colorbar(
        scatter,
        ax=ax[1],
        label="Standardized Residual (z-like score)"
    )


    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 12: Print summary
    # ============================================================

    # print("\n========== STANDARDIZED Dy RESIDUAL ANALYSIS ==========")

    # print(
        # f"Valid pixels: {len(fe_flat):,}"
    # )

    # print(
        # f"Median raw residual: "
        # f"{np.median(raw_residual):.3f} counts"
    # )

    # print(
        # f"Median local residual sigma: "
        # f"{np.median(local_sigma):.3f} counts"
    # )

    # print(
        # f"Median standardized residual: "
        # f"{np.median(standardized_residual):.3f}"
    # )

    # print(
        # f"99th percentile |standardized residual|: "
        # f"{np.percentile(np.abs(standardized_residual), 99):.3f}"
    # )

    # print(
        # f"Pixels with standardized residual > 2: "
        # f"{np.sum(standardized_residual > 2):,}"
    # )

    # print(
        # f"Pixels with standardized residual > 3: "
        # f"{np.sum(standardized_residual > 3):,}"
    # )


    # ============================================================
    # STEP 13: Return everything for autonomous point selection
    # ============================================================

    return {

        "model":
            model,

        "raw_residual_map":
            raw_residual_map,

        "standardized_residual_map":
            standardized_residual_map,

        "raw_residual":
            raw_residual,

        "standardized_residual":
            standardized_residual,

        "local_sigma":
            local_sigma,

        "predicted_dy":
            dy_pred,

        "fe":
            fe_flat,

        "dy":
            dy_flat,

        "valid_pixels":
            valid_pixels,

        "bin_centers":
            bin_centers,

        "bin_scales":
            bin_scales
    }

def create_particle_mask(
    total_counts,
    min_particle_size=20,
    hole_size=20
):
    """
    Create a binary mask:

        True  = particle
        False = background

    total_counts should preferably be:
        - total XRF counts, or
        - sum of several reliable elemental maps
    """

    # ------------------------------------------------------------
    # 1. Log transform
    # Helps separate weak particles from very bright particles
    # ------------------------------------------------------------

    log_counts = np.log1p(total_counts)


    # ------------------------------------------------------------
    # 2. Calculate automatic threshold using Otsu
    # ------------------------------------------------------------

    finite_values = log_counts[np.isfinite(log_counts)]

    threshold = threshold_otsu(finite_values)

    particle_mask = log_counts > threshold


    # ------------------------------------------------------------
    # 3. Remove tiny isolated noise
    # ------------------------------------------------------------

    particle_mask = remove_small_objects(
        particle_mask,
        min_size=min_particle_size
    )


    # ------------------------------------------------------------
    # 4. Fill small holes inside particles
    # ------------------------------------------------------------

    particle_mask = remove_small_holes(
        particle_mask,
        area_threshold=hole_size
    )


    # ------------------------------------------------------------
    # 5. Smooth particle boundaries
    # ------------------------------------------------------------

    particle_mask = binary_closing(
        particle_mask,
        disk(1)
    )


    return particle_mask


def selectResidualExtremes(
    results,
    fe_counts,
    dy_counts,
    circle_radius=5,
    custom_circle_cord=None,
    cmap_element="viridis",
    cmap_residual="seismic"
):
    """
    Find the pixels with the highest and lowest standardized
    Dy residual and plot Fe, Dy, and residual maps.

    Parameters
    ----------
    results : dict
        Dictionary returned by checkDyResidual().

    fe_counts : 2D numpy array
        Raw Fe intensity map.

    dy_counts : 2D numpy array
        Raw Dy intensity map.

    circle_radius : float, default=5
        Radius of circles in pixels.

    custom_circle_cord : tuple/list/None, default=None
        Optional custom coordinate to mark on all maps.

        IMPORTANT:
            Coordinate format is (x, y) = (column, row)

        Example:
            custom_circle_cord=(120, 85)

        Set to None if no custom point is required.

    cmap_element : str, default="viridis"
        Colormap for Fe and Dy maps.

    cmap_residual : str, default="seismic"
        Colormap for standardized residual map.

    Returns
    -------
    selected : dict
        Information about highest, lowest, and optional custom point.
    """

    # ============================================================
    # STEP 1: Extract residual results
    # ============================================================

    z_map = results["standardized_residual_map"]
    raw_map = results["raw_residual_map"]
    model = results["model"]


    # ============================================================
    # STEP 2: Validate array dimensions
    # ============================================================

    if fe_counts.shape != dy_counts.shape:
        raise ValueError(
            "fe_counts and dy_counts must have the same shape."
        )

    if fe_counts.shape != z_map.shape:
        raise ValueError(
            "fe_counts, dy_counts, and residual map "
            "must have the same shape."
        )


    # ============================================================
    # STEP 3: Find highest and lowest residual
    # ============================================================

    if not np.any(np.isfinite(z_map)):
        raise ValueError(
            "No valid standardized residual values found."
        )

    max_flat_index = np.nanargmax(z_map)
    min_flat_index = np.nanargmin(z_map)

    max_row, max_col = np.unravel_index(
        max_flat_index,
        z_map.shape
    )

    min_row, min_col = np.unravel_index(
        min_flat_index,
        z_map.shape
    )


    # ============================================================
    # STEP 4: Extract extreme-point values
    # ============================================================

    max_fe = fe_counts[max_row, max_col]
    max_dy = dy_counts[max_row, max_col]

    min_fe = fe_counts[min_row, min_col]
    min_dy = dy_counts[min_row, min_col]

    max_z = z_map[max_row, max_col]
    min_z = z_map[min_row, min_col]

    max_raw = raw_map[max_row, max_col]
    min_raw = raw_map[min_row, min_col]


    # ============================================================
    # STEP 5: Calculate predicted Dy
    # ============================================================

    max_predicted_dy = model.predict(
        np.array([[max_fe]])
    )[0]

    min_predicted_dy = model.predict(
        np.array([[min_fe]])
    )[0]


    # ============================================================
    # STEP 6: Print highest residual
    # ============================================================

    print("\n" + "=" * 65)
    print("HIGHEST STANDARDIZED Dy RESIDUAL")
    print("=" * 65)

    print(
        f"Pixel coordinate (row, col): "
        f"({max_row}, {max_col})"
    )

    print(
        f"Plot coordinate (x, y): "
        f"({max_col}, {max_row})"
    )

    print(f"Fe intensity:             {max_fe:.3f}")
    print(f"Measured Dy:              {max_dy:.3f}")
    print(f"Predicted Dy from Fe:     {max_predicted_dy:.3f}")
    print(f"Raw Dy residual:          {max_raw:.3f}")
    print(f"Standardized residual:     {max_z:.3f}")


    # ============================================================
    # STEP 7: Print lowest residual
    # ============================================================

    print("\n" + "=" * 65)
    print("LOWEST STANDARDIZED Dy RESIDUAL")
    print("=" * 65)

    print(
        f"Pixel coordinate (row, col): "
        f"({min_row}, {min_col})"
    )

    print(
        f"Plot coordinate (x, y): "
        f"({min_col}, {min_row})"
    )

    print(f"Fe intensity:             {min_fe:.3f}")
    print(f"Measured Dy:              {min_dy:.3f}")
    print(f"Predicted Dy from Fe:     {min_predicted_dy:.3f}")
    print(f"Raw Dy residual:          {min_raw:.3f}")
    print(f"Standardized residual:     {min_z:.3f}")


    # ============================================================
    # STEP 8: Handle custom coordinate
    # ============================================================

    custom_point = None

    if custom_circle_cord is not None:

        if len(custom_circle_cord) != 2:
            raise ValueError(
                "custom_circle_cord must contain exactly "
                "two values: (x, y)."
            )

        custom_x = float(custom_circle_cord[0])
        custom_y = float(custom_circle_cord[1])

        # Check that point is approximately inside image
        nrows, ncols = z_map.shape

        if not (
            0 <= custom_x < ncols and
            0 <= custom_y < nrows
        ):
            raise ValueError(
                f"custom_circle_cord {custom_circle_cord} "
                f"is outside image bounds. "
                f"Valid x range: 0-{ncols - 1}, "
                f"valid y range: 0-{nrows - 1}."
            )

        custom_col = int(round(custom_x))
        custom_row = int(round(custom_y))

        custom_fe = fe_counts[
            custom_row,
            custom_col
        ]

        custom_dy = dy_counts[
            custom_row,
            custom_col
        ]

        custom_z = z_map[
            custom_row,
            custom_col
        ]

        custom_point = {
            "x": custom_x,
            "y": custom_y,
            "row": custom_row,
            "col": custom_col,
            "fe": custom_fe,
            "dy": custom_dy,
            "standardized_residual": custom_z
        }

        print("\n" + "=" * 65)
        print("CUSTOM POINT")
        print("=" * 65)

        print(
            f"Coordinate (x, y): "
            f"({custom_x:.2f}, {custom_y:.2f})"
        )

        print(
            f"Nearest pixel (row, col): "
            f"({custom_row}, {custom_col})"
        )

        print(
            f"Fe intensity: "
            f"{custom_fe:.3f}"
        )

        print(
            f"Dy intensity: "
            f"{custom_dy:.3f}"
        )

        print(
            f"Standardized residual: "
            f"{custom_z:.3f}"
        )


    # ============================================================
    # STEP 9: Residual color limits
    # ============================================================

    z_limit = np.nanpercentile(
        np.abs(z_map),
        99
    )

    if not np.isfinite(z_limit) or z_limit == 0:
        z_limit = 1


    # ============================================================
    # STEP 10: Create figure
    # ============================================================

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )


    # ============================================================
    # STEP 11: RAW Fe MAP
    # ============================================================

    im_fe = ax[0].imshow(
        fe_counts,
        cmap=cmap_element
    )

    ax[0].set_title(
        "Fe Raw Intensity",
        fontsize=13
    )

    ax[0].set_xlabel("X / Column (pixel)")
    ax[0].set_ylabel("Y / Row (pixel)")

    fig.colorbar(
        im_fe,
        ax=ax[0],
        label="Fe Counts"
    )


    # ============================================================
    # STEP 12: RAW Dy MAP
    # ============================================================

    im_dy = ax[1].imshow(
        dy_counts,
        cmap=cmap_element
    )

    ax[1].set_title(
        "Dy Raw Intensity",
        fontsize=13
    )

    ax[1].set_xlabel("X / Column (pixel)")
    ax[1].set_ylabel("Y / Row (pixel)")

    fig.colorbar(
        im_dy,
        ax=ax[1],
        label="Dy Counts"
    )


    # ============================================================
    # STEP 13: STANDARDIZED RESIDUAL MAP
    # ============================================================

    residual_cmap = plt.get_cmap(
        cmap_residual
    ).copy()

    residual_cmap.set_bad("white")

    im_residual = ax[2].imshow(
        z_map,
        cmap=residual_cmap,
        vmin=-z_limit,
        vmax=z_limit
    )

    ax[2].set_title(
        "Standardized Dy Residual",
        fontsize=13
    )

    ax[2].set_xlabel("X / Column (pixel)")
    ax[2].set_ylabel("Y / Row (pixel)")

    fig.colorbar(
        im_residual,
        ax=ax[2],
        label="Standardized Residual"
    )


    # ============================================================
    # STEP 14: Draw MAX and MIN circles on all maps
    # ============================================================

    for axis in ax:

        # --------------------------------------------------------
        # Highest residual
        # --------------------------------------------------------

        axis.add_patch(
            Circle(
                (max_col, max_row),
                radius=circle_radius,
                fill=False,
                edgecolor="lime",
                linewidth=3
            )
        )


        # --------------------------------------------------------
        # Lowest residual
        # --------------------------------------------------------

        axis.add_patch(
            Circle(
                (min_col, min_row),
                radius=circle_radius,
                fill=False,
                edgecolor="yellow",
                linewidth=3
            )
        )


    # ============================================================
    # STEP 15: Draw CUSTOM circle if provided
    # ============================================================

    if custom_point is not None:

        for axis in ax:

            axis.add_patch(
                Circle(
                    (
                        custom_point["x"],
                        custom_point["y"]
                    ),
                    radius=circle_radius,
                    fill=False,
                    edgecolor="magenta",
                    linewidth=3
                )
            )


    # ============================================================
    # STEP 16: Labels on residual map
    # ============================================================

    ax[2].text(
        max_col + circle_radius + 2,
        max_row,
        f"MAX\nz={max_z:.2f}",
        fontsize=10,
        fontweight="bold",
        color="black",
        verticalalignment="center"
    )

    ax[2].text(
        min_col + circle_radius + 2,
        min_row,
        f"MIN\nz={min_z:.2f}",
        fontsize=10,
        fontweight="bold",
        color="black",
        verticalalignment="center"
    )


    # Custom-point label
    if custom_point is not None:

        ax[2].text(
            custom_point["x"] + circle_radius + 2,
            custom_point["y"],
            f"CUSTOM\n"
            f"z={custom_point['standardized_residual']:.2f}",
            fontsize=10,
            fontweight="bold",
            color="black",
            verticalalignment="center"
        )


    # ============================================================
    # STEP 17: Overall title
    # ============================================================

    fig.suptitle(
        "Fe, Dy, and Standardized Dy Residual Analysis",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


    # ============================================================
    # STEP 18: Return results
    # ============================================================

    selected = {

        "highest": {
            "row": max_row,
            "col": max_col,
            "fe": max_fe,
            "dy_measured": max_dy,
            "dy_predicted": max_predicted_dy,
            "raw_residual": max_raw,
            "standardized_residual": max_z
        },

        "lowest": {
            "row": min_row,
            "col": min_col,
            "fe": min_fe,
            "dy_measured": min_dy,
            "dy_predicted": min_predicted_dy,
            "raw_residual": min_raw,
            "standardized_residual": min_z
        },

        "custom": custom_point
    }

    return selected

