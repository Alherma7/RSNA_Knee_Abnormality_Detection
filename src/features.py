"""Feature/representation construction: 2.5D slice triplets per plane.

Source: Dive into Deep Learning ch. 14 (Computer Vision) — specifically
14.1 Image Augmentation and 14.2 Fine-Tuning, verified against the actual
book text 2026-08-17 (see docs/superpowers/specs/2026-08-17-fase4-
baseline-cnn-design.md). Corrects an earlier citation in this docstring
that pointed to ch. 8 ("Modern Convolutional Neural Networks") for
fine-tuning — checked against the book's real table of contents: ch. 8
covers classic architectures (AlexNet, VGG, ResNet, ...), not
fine-tuning, which is entirely in 14.2. Also both reviewed reference
notebooks (pilkwang/rsna-knee-baseline-v1,
prvsiyan/rsna-knee-read-the-report-then-the-knee), which use [n-gap, n,
n+gap] slice triplets through a shared backbone, and flag two
preprocessing steps as decisive rather than optional (below).

select_group() (A3) has a separate source: stevenleehans/rsna-knee-
500gb-to-11gib-cpu-pixel-cache — see its RESOURCES.md entry and
src/data.py::load_slot_cache_shard()'s docstring for the pixel cache
this operates on.
"""

import numpy as np

from src import config


def normalize_physical_scale(pixel_array: np.ndarray, pixel_spacing_mm: float,
                              target_mm_per_pixel: float) -> np.ndarray:
    """Resize using physical (mm) spacing, not a fixed pixel size.

    Both reference notebooks note that DICOM pixel spacing varies across
    studies, so a fixed-pixel resize hands the model images whose
    physical scale differs by several times. Measured (notebooks/
    01_eda_dicom.ipynb, section F, 2026-08-16 kernel run, 165 series
    from 30 sample studies): PixelSpacing ranges 0.137-0.703 mm/pixel,
    a 5.14x max/min ratio, confirming the claim on this corpus — not
    dramatically different by plane (0.33-0.35 mm mean across
    sagittal/coronal/axial), so the variation is per-study/protocol, not
    per-plane. Row and column spacing were identical on every sampled
    series (isotropic in-plane). SliceThickness ranged 0.6-5.0 mm
    (median 3.0); SpacingBetweenSlices was present on 151/165 series
    (missing on 14) — fall back to SliceThickness or derive spacing from
    consecutive SliceLocation values when it's absent. A feature
    narrower than ~2 pixels after resize does not survive — resize by
    physical millimetres covered, not raw pixel count.
    """
    raise NotImplementedError("Fill in once DICOM loading (src.data) works.")


def normalize_laterality(pixel_array: np.ndarray, is_right_knee: bool) -> np.ndarray:
    """Flip left/right so medial/lateral findings share one image axis.

    Five of the twelve findings (medial/lateral meniscus, medial/lateral
    tibiofemoral compartment OA, and MCL, named for its side) are defined
    relative to the body's midline, which falls on different sides of the
    image depending on which knee was scanned. Both reference notebooks
    treat this as a required normalization, not an optional augmentation.

    `is_right_knee` can be read directly from the DICOM `Laterality` tag
    (confirmed present, notebooks/01_eda_dicom.ipynb section D,
    2026-08-16 kernel run) rather than inferred — src/data.py's loader
    should surface it per series instead of guessing from other fields.
    """
    raise NotImplementedError("Fill in once laterality metadata is loaded.")


def sample_slice_indices(n_slices: int, n_triplets: int, gap: int) -> "list[int]":
    """Uniformly sample center-slice indices for 2.5D triplet construction."""
    raise NotImplementedError("Fill in once series lengths are known from EDA.")


def build_25d_triplet(slices: "list[np.ndarray]", center_idx: int, gap: int) -> np.ndarray:
    """Stack [center-gap, center, center+gap] slices into a 3-channel image."""
    raise NotImplementedError("Fill in once DICOM loading (src.data) works.")


def select_group(cache_slot_stack: np.ndarray, group_index) -> np.ndarray:
    """Select one or more anchor groups from a slot cache's slice axis.

    `cache_slot_stack`: array shaped `(..., 9, H, W)` — the slice axis
    from `src.data.load_slot_cache_shard()`'s `cache` (whole array or
    any indexed slice of it, e.g. one study/one slot).

    `group_index`: an int (one group, `config.SLOT_CACHE_GROUP_SIZE`
    channels out) or a sequence of ints (multiple groups, concatenated
    on the channel axis — e.g. `[0, 1, 2]` for all 9 slices). Group 0/1/2
    are the low/centre/high sampling anchors, in that order — mirrors
    stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache's own
    `take_group(rows, g) = rows[:, :, g*group:(g+1)*group]` exactly, so
    group 1 is the centre anchor.

    Deliberately no default for `group_index`: which group(s) to
    actually use for A2 is an open question (see the source's own
    `exp-016` 3-vs-9-slices finding, measured confounded with anchor
    position — full detail in RESOURCES.md), left to A2's own
    cross-validation rather than pre-committed here.

    Graduated 2026-08-26 from notebooks/04v2_slot_cache_integration.ipynb
    (A3), where this matched the source's `take_group` indexing exactly
    on real cache data (group 1 alone == the middle third of `[0, 1, 2]`
    concatenated).
    """
    if isinstance(group_index, int):
        group_index = [group_index]
    size = config.SLOT_CACHE_GROUP_SIZE
    groups = [cache_slot_stack[..., g * size:(g + 1) * size, :, :] for g in group_index]
    return np.concatenate(groups, axis=-3)


def expand_slot_groups(cache_slot_stack: np.ndarray, slot_mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Turn one study's 6 real slots x 3 anchor groups into 18 independent
    pseudo-slots, instead of `select_group`'s single chosen group per slot.

    `cache_slot_stack`: `(n_slots, 9, H, W)` — one study, every slot's full
    9-slice pool (from `src.data.load_slot_cache_shard()`'s `cache`).
    `slot_mask`: `(n_slots,)` — one presence bit per real slot (the cache
    has no per-group presence, only per-slot: a slot's 3 groups are always
    jointly present or jointly absent).

    Returns `(images, mask)`:
    - `images`: `(n_slots * SLOT_CACHE_N_GROUPS, SLOT_CACHE_GROUP_SIZE, H, W)`,
      **slot-major order** — pseudo-slot index `s * SLOT_CACHE_N_GROUPS + g`
      for real slot `s` and group `g`, i.e.
      `slot0_g0, slot0_g1, slot0_g2, slot1_g0, ...`.
    - `mask`: `(n_slots * SLOT_CACHE_N_GROUPS,)`, each real slot's bit
      replicated `SLOT_CACHE_N_GROUPS` times.

    Motivation: A2 v1 fixes `group_index=1` (centre anchor only). The
    pretrained backbone expects exactly 3 input channels, so the 3 groups
    can't be concatenated on the channel axis the way `select_group([0,1,2])`
    does (that produces a 9-channel image) — each group must become its
    own 3-channel pseudo-slot instead, letting the model's existing
    per-finding masked attention (already generic over slot count) learn
    which group matters per finding.

    Calls `select_group()` internally for the per-group slice extraction
    rather than re-implementing that indexing.

    Graduated 2026-08-29 from notebooks/09v1_a2v2_multigroup_baseline.ipynb
    / notebooks/10v1_a2v2_pooled_4fold_cv.ipynb (A2 v2) — real pooled
    4-fold gold macro-AUC 0.8009, beating A2 v1's 0.7512 pooled baseline;
    see docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md
    section 5.2 for the full gate.
    """
    n_slots = cache_slot_stack.shape[0]
    h, w = cache_slot_stack.shape[-2:]
    n_groups = config.SLOT_CACHE_N_GROUPS
    groups = [select_group(cache_slot_stack, g) for g in range(n_groups)]
    stacked = np.stack(groups, axis=1)  # (n_slots, n_groups, group_size, H, W)
    images = stacked.reshape(n_slots * n_groups, config.SLOT_CACHE_GROUP_SIZE, h, w)
    mask = np.repeat(slot_mask, n_groups)
    return images, mask
