import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Sørensen-Dice coefficient between two binary arrays."""
    pred   = pred.astype(bool)
    target = target.astype(bool)

    intersection = (pred & target).sum()
    union        = pred.sum() + target.sum()

    return float((2.0 * intersection + eps) / (union + eps))


def boundary_recall(
    superpixels: np.ndarray,
    gt: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Fraction of GT boundary pixels matched by a superpixel boundary."""
    gt_boundaries = find_boundaries(gt.astype(int), mode="inner")
    sp_boundaries = find_boundaries(superpixels, mode="inner")

    n_gt = int(gt_boundaries.sum())
    if n_gt == 0:
        return 1.0

    if tolerance == 0:
        matched = int((gt_boundaries & sp_boundaries).sum())
        return matched / n_gt

    # Distance transform from superpixel boundaries: for every pixel,
    # distance to the nearest superpixel boundary pixel.
    dist_to_sp = ndi.distance_transform_edt(~sp_boundaries)

    matched = int((gt_boundaries & (dist_to_sp <= tolerance)).sum())
    return matched / n_gt


def miou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Intersection-over-Union for binary segmentation."""
    pred   = pred.astype(bool)
    target = target.astype(bool)

    # Foreground IoU
    tp_fg   = int((pred  & target ).sum())
    fp_fg   = int((pred  & ~target).sum())
    fn_fg   = int((~pred & target ).sum())
    iou_fg  = (tp_fg + eps) / (tp_fg + fp_fg + fn_fg + eps)

    # Background IoU (swap roles)
    tp_bg   = int((~pred & ~target).sum())
    fp_bg   = fn_fg          # foreground false-negatives == background false-positives
    fn_bg   = fp_fg          # foreground false-positives == background false-negatives
    iou_bg  = (tp_bg + eps) / (tp_bg + fp_bg + fn_bg + eps)

    return float((iou_fg + iou_bg) / 2.0)
