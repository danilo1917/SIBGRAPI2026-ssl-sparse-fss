import numpy as np
from skimage import measure, morphology


def sparse_points(msk: np.ndarray, sparsity_param: dict) -> np.ndarray:
    """Randomly label ``num_points`` foreground pixels and ``num_points`` background pixels as known; everything else becomes -1."""
    if not np.any(msk):
        return msk.astype(np.int8)

    msk_ravel = msk.ravel()

    # Start every pixel as "unknown".
    new_msk_ravel = np.full(msk_ravel.shape[0], -1, dtype=np.int8)

    # Background pixels.
    neg_indices = np.where(msk_ravel == 0)[0]
    chosen_neg  = np.random.permutation(neg_indices)[:sparsity_param["num_points"]]
    new_msk_ravel[chosen_neg] = 0

    # Foreground pixels.
    pos_indices = np.where(msk_ravel > 0)[0]
    chosen_pos  = np.random.permutation(pos_indices)[:sparsity_param["num_points"]]
    new_msk_ravel[chosen_pos] = 1

    new_msk = new_msk_ravel.reshape(msk.shape)

    # Dilate each labelled point into a small disk.
    if sparsity_param["radius"] > 0:
        selem = morphology.disk(sparsity_param["radius"])
        new_msk[morphology.binary_dilation(new_msk == 0, selem) & (msk == 0)] = 0
        new_msk[morphology.binary_dilation(new_msk == 1, selem) & (msk == 1)] = 1

    # Safety: never let a label contradict the source mask.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1

    return new_msk


def sparse_grid(msk: np.ndarray, sparsity_param: dict) -> np.ndarray:
    """Retain only pixels that fall on a regular grid; set everything else to -1."""
    if not np.any(msk):
        return msk.astype(np.int8)

    space_y = sparsity_param["space_y"]
    space_x = sparsity_param["space_x"]

    # Random starting offset so the grid is not always anchored at (0,0).
    start_y = np.random.randint(space_y)
    start_x = np.random.randint(space_x)

    new_msk = np.full_like(msk, -1, dtype=np.int8)
    new_msk[start_y::space_y, start_x::space_x] = msk[start_y::space_y, start_x::space_x]

    # Dilate grid points into small disks.
    if sparsity_param["radius"] > 0:
        selem = morphology.disk(sparsity_param["radius"])
        new_msk[morphology.binary_dilation(new_msk == 0, selem) & (msk == 0)] = 0
        new_msk[morphology.binary_dilation(new_msk == 1, selem) & (msk == 1)] = 1

    # Safety check.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1

    return new_msk


def sparse_scribbles(msk: np.ndarray, sparsity_param: dict) -> np.ndarray:
    """Simulate a user drawing a scribble along the inside and outside boundaries."""
    if not np.any(msk):
        return msk.astype(np.int8)

    prop         = sparsity_param["prop"]
    radius_dist  = sparsity_param["dist"]
    radius_thick = sparsity_param["thick"]

    selem_dist = morphology.disk(radius_dist)

    # Erode -> inner boundary (positive class scribble source).
    msk_pos = morphology.binary_erosion(msk > 0, selem_dist)
    if not np.any(msk_pos):
        msk_pos = msk > 0          # fall back if organ is too small

    # Dilate -> outer boundary (negative class scribble source).
    msk_neg = morphology.binary_dilation(msk > 0, selem_dist)

    pos_contours = measure.find_contours(msk_pos)
    neg_contours = measure.find_contours(msk_neg)

    msk_pos_bound = np.zeros_like(msk, dtype=bool)
    msk_neg_bound = np.zeros_like(msk, dtype=bool)

    def _fill_bound(contours, bound_arr):
        for obj in contours:
            if len(obj) < 2:
                continue
            start = np.random.randint(1, max(2, len(obj)))
            n_pts = max(1, round(len(obj) * prop))
            for pt in np.roll(obj, start, axis=0)[:n_pts]:
                bound_arr[int(pt[0]), int(pt[1])] = True

    _fill_bound(pos_contours, msk_pos_bound)
    _fill_bound(neg_contours, msk_neg_bound)

    # Give scribbles some thickness via dilation.
    if radius_thick > 0:
        selem_thick = morphology.disk(radius_thick)
        msk_pos_bound = morphology.binary_dilation(msk_pos_bound, selem_thick)
        msk_neg_bound = morphology.binary_dilation(msk_neg_bound, selem_thick)

    # Clip so scribbles stay inside their correct region.
    msk_pos_bound = msk_pos_bound & (msk == 1)
    msk_neg_bound = msk_neg_bound & (msk == 0)

    new_msk = np.full_like(msk, -1, dtype=np.int8)
    new_msk[msk_pos_bound] = 1
    new_msk[msk_neg_bound] = 0

    # Safety check.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1

    return new_msk


def sparse_contours(msk: np.ndarray, sparsity_param: dict) -> np.ndarray:
    """Retain only a partial arc of the object boundary; dilate it into a strip."""
    if not np.any(msk):
        return msk.astype(np.int8)

    prop         = sparsity_param["prop"]
    radius_thick = sparsity_param["thick"]

    contours = measure.find_contours(msk)

    new_msk = np.full_like(msk, -1, dtype=np.int8)

    for obj in contours:
        if len(obj) < 2:
            continue
        start = np.random.randint(1, max(2, len(obj)))
        n_pts = max(1, round(len(obj) * prop))
        for pt in np.roll(obj, start, axis=0)[:n_pts]:
            new_msk[int(pt[0]), int(pt[1])] = 1

    # Dilate the arc and label the strip correctly (both sides).
    if radius_thick > 0:
        selem       = morphology.disk(radius_thick)
        contour_msk = morphology.binary_dilation(new_msk != -1, selem)
        new_msk[contour_msk] = msk[contour_msk]

    # Safety check.
    new_msk[(new_msk == 1) & (msk == 0)] = -1
    new_msk[(new_msk == 0) & (msk == 1)] = -1

    return new_msk


def aplicar_esparsidade(msk: np.ndarray) -> np.ndarray:
    """Apply a randomly chosen sparse-annotation style to ``msk``."""
    msk_int = msk.astype(np.int8)   # work with integers internally

    mode = np.random.randint(4)     # 0=points  1=grid  2=scribbles  3=contours

    if mode == 0:
        # Points: 1–10 dots per class, disk radius 3–5 px.
        params = {
            "num_points": np.random.randint(1, 11),
            "radius":     np.random.randint(3, 6),
        }
        sparse = sparse_points(msk_int, params)

    elif mode == 1:
        # Grid: spacing 8–41 px on each axis, disk radius 3–4 px.
        params = {
            "space_y": np.random.randint(8, 42),
            "space_x": np.random.randint(8, 42),
            "radius":  np.random.randint(3, 5),
        }
        sparse = sparse_grid(msk_int, params)

    elif mode == 2:
        # Scribbles: 0–100 % of contour, erosion/dilation radius 4–11 px,
        # scribble thickness 2–5 px.
        params = {
            "prop":  np.random.random(),
            "dist":  np.random.randint(4, 12),
            "thick": np.random.randint(2, 6),
        }
        sparse = sparse_scribbles(msk_int, params)

    else:
        # Contours: 0–100 % of boundary arc, strip half-width 2–9 px.
        params = {
            "prop":  np.random.random(),
            "thick": np.random.randint(2, 10),
        }
        sparse = sparse_contours(msk_int, params)

    # Return as float32 so it fits the existing tensor pipeline unchanged.
    return sparse.astype(np.float32)


# Ordered list of named modes (index matches the mode integer in aplicar_esparsidade).
SPARSE_MODES = ["points", "grid", "scribbles", "contours"]
SPARSE_MODES_ALL = SPARSE_MODES + ["mixed"]


def aplicar_esparsidade_modo(msk: np.ndarray, modo: str) -> np.ndarray:
    """Apply a specific sparsification mode deterministically."""
    if modo not in SPARSE_MODES_ALL:
        raise ValueError(f"Unknown mode '{modo}'. Options: {SPARSE_MODES_ALL}")

    msk_int = msk.astype(np.int8)

    if modo == "mixed":
        mode_idx = np.random.randint(4)
    else:
        mode_idx = SPARSE_MODES.index(modo)

    if mode_idx == 0:
        params = {"num_points": np.random.randint(1, 11), "radius": np.random.randint(3, 6)}
        sparse = sparse_points(msk_int, params)
    elif mode_idx == 1:
        params = {"space_y": np.random.randint(8, 42), "space_x": np.random.randint(8, 42), "radius": np.random.randint(3, 5)}
        sparse = sparse_grid(msk_int, params)
    elif mode_idx == 2:
        params = {"prop": np.random.random(), "dist": np.random.randint(4, 12), "thick": np.random.randint(2, 6)}
        sparse = sparse_scribbles(msk_int, params)
    else:
        params = {"prop": np.random.random(), "thick": np.random.randint(2, 10)}
        sparse = sparse_contours(msk_int, params)

    return sparse.astype(np.float32)
