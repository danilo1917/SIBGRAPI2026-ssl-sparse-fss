import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation


_MIN_FG_FRAC = 0.005   # 0.5 %  — reject near-empty masks
_MAX_FG_FRAC = 0.80    # 80 %   — anatomy is rarely > 80 % of a 128×128 slice
_MIN_SEG_PX  = 16      # ignore superpixels smaller than this


def _cleanup(mask: np.ndarray) -> np.ndarray:
    """Fill holes, remove debris, enforce fg fraction bounds."""
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=max(4, int(mask.size * _MIN_FG_FRAC)))
    mask = mask.astype(np.float32)
    fg = mask.mean()
    if fg < _MIN_FG_FRAC or fg > _MAX_FG_FRAC:
        return np.zeros_like(mask)
    return mask


def _anatomy_score(mask: np.ndarray, img: np.ndarray, median: float) -> float:
    """Score a binary region (0..1) by how anatomy-like it is."""
    area = int(mask.sum())
    if area < _MIN_SEG_PX:
        return -1.0

    # Brightness: anatomy is brighter than background
    brightness = float((img[mask.astype(bool)] >= median).mean())

    # Compactness: organs are blob-shaped, not thin/wispy
    boundary = morphology.binary_erosion(mask.astype(bool)) ^ mask.astype(bool)
    perimeter = int(boundary.sum()) + 1
    compactness = float(np.clip(4 * np.pi * area / perimeter ** 2, 0.0, 1.0))

    # Interior: anatomy does not touch the image edge
    h, w = mask.shape
    border_px = int(
        mask[0, :].sum() + mask[-1, :].sum() +
        mask[:, 0].sum() + mask[:, -1].sum()
    )
    interior = float(max(0.0, 1.0 - (border_px / area) * 5))

    return brightness + compactness + interior


def _pick_region(seg: np.ndarray, img: np.ndarray, top_k: int = 4) -> int:
    """Pick a superpixel seed label biased toward anatomy-like regions."""
    unique = np.unique(seg)
    median = float(np.median(img))

    scored = [(
        _anatomy_score(seg == lab, img, median), int(lab)
    ) for lab in unique]

    valid = [(s, lab) for s, lab in scored if s >= 0]
    if not valid:
        return int(unique[np.random.randint(len(unique))])

    valid.sort(key=lambda x: -x[0])
    k = min(top_k, len(valid))
    top_s = np.array([s for s, _ in valid[:k]], dtype=np.float64)
    top_s -= top_s.max()
    w = np.exp(top_s); w /= w.sum()
    return valid[int(np.random.choice(k, p=w))][1]


def _grow(
    seg: np.ndarray,
    seed: int,
    img: np.ndarray,
    n_extra: int,
) -> np.ndarray:
    """Grow n_extra anatomically-scored neighbours from seed, return binary mask."""
    selected = {seed}
    median = float(np.median(img))

    for _ in range(n_extra):
        cur = np.isin(seg, list(selected))
        dilated = morphology.binary_dilation(cur, morphology.disk(1))
        neighbours = set(np.unique(seg[dilated & ~cur])) - selected
        if not neighbours:
            break
        cands = sorted(
            [((_anatomy_score(seg == lab, img, median)), int(lab)) for lab in neighbours],
            key=lambda x: -x[0],
        )
        top = cands[:max(1, len(cands) // 2)]
        selected.add(top[np.random.randint(len(top))][1])

    return np.isin(seg, list(selected)).astype(np.float32)


def _slic_anatomy(img: np.ndarray) -> np.ndarray:
    n_seg   = int(np.random.choice([50, 75, 100]))
    compact = float(np.random.choice([0.05, 0.1, 0.3]))
    seg = segmentation.slic(img, n_segments=n_seg, compactness=compact,
                             sigma=1.0, channel_axis=None, start_label=0)
    seed = _pick_region(seg, img, top_k=4)
    n_extra = int(np.random.randint(0, 3))
    return _cleanup(_grow(seg, seed, img, n_extra))


def _felzenszwalb_anatomy(img: np.ndarray) -> np.ndarray:
    scale    = int(np.random.choice([50, 100, 150]))
    min_size = int(np.random.choice([30, 60, 100]))
    seg = segmentation.felzenszwalb(img, scale=scale, sigma=0.5, min_size=min_size)
    seed = _pick_region(seg, img, top_k=4)
    n_extra = int(np.random.randint(0, 2))
    return _cleanup(_grow(seg, seed, img, n_extra))


def _watershed_anatomy(img: np.ndarray) -> np.ndarray:
    n_markers = int(np.random.choice([10, 20, 40]))
    seg = segmentation.watershed(filters.sobel(img), markers=n_markers, compactness=0.001)
    seed = _pick_region(seg, img, top_k=4)
    n_extra = int(np.random.randint(0, 2))
    return _cleanup(_grow(seg, seed, img, n_extra))


def propor_mascara(img: np.ndarray) -> np.ndarray:
    """Generate an anatomy-biased binary pseudo-label for a 2-D slice."""
    if img.max() - img.min() < 1e-3:
        return np.zeros_like(img, dtype=np.float32)

    roll = np.random.randint(0,3)
    functions = [_slic_anatomy, _felzenszwalb_anatomy, _watershed_anatomy]
    fn = functions[roll]

    try:
        result = fn(img)
    except Exception:
        result = np.zeros_like(img, dtype=np.float32)

    if result.max() == 0:
        try:
            roll = np.random.randint(0,3)
            fn = functions[roll]
            result = fn(img)
        except Exception:
            pass

    return result
