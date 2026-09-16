import numpy as np

def print_h5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name}")
        print(f"    shape: {obj.shape}")
        print(f"    dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP:   {name}")

def get_element_map(element, xrf_names, xrf_fit):

    # --------------------------------------------------------
    # 1. Try exact match first
    # --------------------------------------------------------

    exact_matches = [
        i
        for i, name in enumerate(xrf_names)
        if name.strip().lower() == element.strip().lower()
    ]

    if len(exact_matches) > 0:
        return xrf_fit[exact_matches[0]]


    # --------------------------------------------------------
    # 2. If exact match doesn't exist, show possible matches
    # --------------------------------------------------------

    partial_matches = [
        (i, name)
        for i, name in enumerate(xrf_names)
        if element.strip().lower() in name.strip().lower()
    ]

    if len(partial_matches) > 0:

        raise ValueError(
            f"Exact map '{element}' was not found.\n"
            f"Possible matches: {partial_matches}"
        )


    # --------------------------------------------------------
    # 3. No match at all
    # --------------------------------------------------------

    raise ValueError(
        f"'{element}' was not found.\n"
        f"Available maps:\n{xrf_names}"
    )

def percentile_normalize(img, low=1, high=99):
    img = np.asarray(img, dtype=float)

    p_low = np.nanpercentile(img, low)
    p_high = np.nanpercentile(img, high)

    norm = (img - p_low) / (p_high - p_low)

    return np.clip(norm, 0, 1)

def threshold_map(img, percentile=75):
    threshold = np.nanpercentile(img, percentile)
    return img >= threshold