import numpy as np


def intersection_over_union(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Calculate IoU between two binary segmentation masks."""

    if predicted.shape != ground_truth.shape:
        raise ValueError("Masks must have the same shape.")

    if predicted.ndim != 2:
        raise ValueError("Masks must be 2D arrays.")

    pred = predicted.astype(bool)
    gt = ground_truth.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0  # Both masks are empty.

    return float(intersection / union)
