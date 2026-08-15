"""Feature/representation construction: 2.5D slice triplets per plane.

Source: Dive into Deep Learning, ch. 8 (Modern CNNs — fine-tuning) and
ch. 14 (Computer Vision — image augmentation, fine-tuning in practice);
also both reviewed reference notebooks (pilkwang/rsna-knee-baseline-v1,
prvsiyan/rsna-knee-read-the-report-then-the-knee), which use [n-gap, n,
n+gap] slice triplets through a shared backbone, and flag two
preprocessing steps as decisive rather than optional (below).
"""

import numpy as np


def normalize_physical_scale(pixel_array: np.ndarray, pixel_spacing_mm: float,
                              target_mm_per_pixel: float) -> np.ndarray:
    """Resize using physical (mm) spacing, not a fixed pixel size.

    Both reference notebooks note that DICOM pixel spacing varies across
    studies, so a fixed-pixel resize hands the model images whose
    physical scale differs by several times. A feature narrower than
    ~2 pixels after resize does not survive — resize by physical
    millimetres covered, not raw pixel count.
    """
    raise NotImplementedError("Fill in once DICOM loading (src.data) works.")


def normalize_laterality(pixel_array: np.ndarray, is_right_knee: bool) -> np.ndarray:
    """Flip left/right so medial/lateral findings share one image axis.

    Five of the twelve findings (medial/lateral meniscus, medial/lateral
    tibiofemoral compartment OA, and MCL, named for its side) are defined
    relative to the body's midline, which falls on different sides of the
    image depending on which knee was scanned. Both reference notebooks
    treat this as a required normalization, not an optional augmentation.
    """
    raise NotImplementedError("Fill in once laterality metadata is loaded.")


def sample_slice_indices(n_slices: int, n_triplets: int, gap: int) -> "list[int]":
    """Uniformly sample center-slice indices for 2.5D triplet construction."""
    raise NotImplementedError("Fill in once series lengths are known from EDA.")


def build_25d_triplet(slices: "list[np.ndarray]", center_idx: int, gap: int) -> np.ndarray:
    """Stack [center-gap, center, center+gap] slices into a 3-channel image."""
    raise NotImplementedError("Fill in once DICOM loading (src.data) works.")
