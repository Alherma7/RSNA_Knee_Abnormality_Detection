"""Unit tests for src/features.py.

select_group() is graduated (2026-08-26, A3, from
notebooks/04v2_slot_cache_integration.ipynb — see that notebook and
RESOURCES.md for the real slot-cache numbers these unit tests don't
repeat). normalize_physical_scale(), normalize_laterality(),
sample_slice_indices(), and build_25d_triplet() remain
NotImplementedError until Fase 4's ad-hoc DICOM-loading notebook code is
itself graduated (see README.md Next steps) — no tests for those yet.
"""

import numpy as np

from src.config import SLOT_CACHE_GROUP_SIZE
from src.features import select_group


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
