"""Unit tests for src/features.py.

select_group() is graduated (2026-08-26, A3, from
notebooks/04v2_slot_cache_integration.ipynb — see that notebook and
RESOURCES.md for the real slot-cache numbers these unit tests don't
repeat). expand_slot_groups() is graduated 2026-08-29 (A2 v2, from
notebooks/09v1_a2v2_multigroup_baseline.ipynb /
10v1_a2v2_pooled_4fold_cv.ipynb — real pooled 4-fold gold macro-AUC
0.8009, see docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md
section 5.2). normalize_physical_scale(), normalize_laterality(),
sample_slice_indices(), and build_25d_triplet() remain
NotImplementedError until Fase 4's ad-hoc DICOM-loading notebook code is
itself graduated (see README.md Next steps) — no tests for those yet.
"""

import numpy as np
import pytest

from src.config import SLOT_CACHE_GROUP_SIZE, SLOT_CACHE_N_GROUPS
from src.features import expand_slot_groups, select_group


def _make_stack(n_groups=3, group_size=SLOT_CACHE_GROUP_SIZE, h=4, w=4):
    """A (n_groups * group_size, h, w) stack where each channel's pixels
    all equal that channel's index — makes it trivial to assert which
    channels select_group() picked out."""
    n_channels = n_groups * group_size
    stack = np.zeros((n_channels, h, w), dtype=np.uint8)
    for c in range(n_channels):
        stack[c] = c
    return stack


def test_select_group_with_int_returns_one_groups_worth_of_channels():
    stack = _make_stack()

    centre = select_group(stack, 1)

    assert centre.shape == (SLOT_CACHE_GROUP_SIZE, 4, 4)
    assert list(centre[:, 0, 0]) == [3, 4, 5]


def test_select_group_with_sequence_concatenates_groups_in_order():
    stack = _make_stack()

    low_and_high = select_group(stack, [0, 2])

    assert low_and_high.shape == (2 * SLOT_CACHE_GROUP_SIZE, 4, 4)
    assert list(low_and_high[:, 0, 0]) == [0, 1, 2, 6, 7, 8]


def test_select_group_all_three_groups_matches_manual_concatenation():
    stack = _make_stack()

    all_groups = select_group(stack, [0, 1, 2])
    centre_only = select_group(stack, 1)

    assert all_groups.shape == stack.shape
    assert np.array_equal(all_groups, stack)
    assert np.array_equal(all_groups[3:6], centre_only)


def test_select_group_preserves_leading_batch_dimensions():
    # (n_studies, n_slots, 9, H, W), matching load_slot_cache_shard()'s
    # real cache layout -- select_group must index the slice axis only.
    batched = np.zeros((2, 6, 9, 4, 4), dtype=np.uint8)
    for c in range(9):
        batched[:, :, c] = c

    centre = select_group(batched, 1)

    assert centre.shape == (2, 6, SLOT_CACHE_GROUP_SIZE, 4, 4)
    assert list(centre[0, 0, :, 0, 0]) == [3, 4, 5]


def _make_multi_slot_stack(n_slots=6, h=4, w=4):
    """(n_slots, 9, h, w) where every slot has the same _make_stack()
    content (each channel's pixels equal that channel's index)."""
    single = _make_stack(h=h, w=w)
    return np.stack([single] * n_slots, axis=0)


def test_expand_slot_groups_shape():
    stack = _make_multi_slot_stack()
    mask = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    images, out_mask = expand_slot_groups(stack, mask)

    assert images.shape == (6 * SLOT_CACHE_N_GROUPS, SLOT_CACHE_GROUP_SIZE, 4, 4)
    assert out_mask.shape == (6 * SLOT_CACHE_N_GROUPS,)


def test_expand_slot_groups_matches_select_group_per_slot_group_pair():
    """Spec section 2.1's core equation: pseudo-slot s*3+g must equal
    select_group(stack[s], g) exactly, slot-major order."""
    stack = _make_multi_slot_stack()
    mask = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    images, _ = expand_slot_groups(stack, mask)

    for s in range(6):
        for g in range(SLOT_CACHE_N_GROUPS):
            expected = select_group(stack[s], g)
            assert np.array_equal(images[s * SLOT_CACHE_N_GROUPS + g], expected), (s, g)


def test_expand_slot_groups_replicates_each_slots_mask_bit_three_times():
    stack = _make_multi_slot_stack()
    mask = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    _, out_mask = expand_slot_groups(stack, mask)

    assert np.array_equal(out_mask, np.repeat(mask, SLOT_CACHE_N_GROUPS))
    # explicit per-slot check, matching spec section 2.1's stated equation
    for s in range(6):
        for g in range(SLOT_CACHE_N_GROUPS):
            assert out_mask[s * SLOT_CACHE_N_GROUPS + g] == mask[s]


def test_expand_slot_groups_is_single_study_only_no_leading_batch_dim():
    # Unlike select_group(), expand_slot_groups() infers n_slots from
    # shape[0] directly (per spec section 2.1: "one study's full slot
    # stack (6, 9, H, W)") -- SlotCacheDataset.__getitem__ always calls it
    # per-study, one at a time. A leading batch dim breaks the reshape
    # rather than silently producing a wrong-but-plausible shape.
    single = _make_multi_slot_stack()
    batched = np.stack([single, single], axis=0)
    mask = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError):
        expand_slot_groups(batched, mask)
