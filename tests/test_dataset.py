"""Unit tests for src/dataset.py.

SlotCacheDataset is graduated (2026-08-26, A2) from
notebooks/05v2_slot_attention_baseline.ipynb, where this exact
construction trained for real on Kaggle (0.7689 gold macro-AUC, fold 0)
— that real-data training result isn't re-proven here, only the
Dataset's own indexing/alignment logic on small synthetic cache shards
(same technique as tests/test_data.py's load_slot_cache_shard tests).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import FINDINGS
from src.dataset import SlotCacheDataset


def _write_shard(cache_dir, shard, study_ids, fill_value=0):
    """A minimal but real shard: (n, 6, 9, 4, 4) uint8 cache instead of
    the real (n, 6, 9, 224, 224) — same shape family, cheap to write/read.
    Every pixel in study i's cache is set to `fill_value + i`, making it
    trivial to assert which study's data __getitem__ actually returned."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(study_ids)
    cache = np.zeros((n, 6, 9, 4, 4), dtype=np.uint8)
    for i in range(n):
        cache[i] = fill_value + i
    mask = np.ones((n, 6), dtype=np.float32)
    np.save(cache_dir / f"{shard}_cache.npy", cache)
    np.save(cache_dir / f"{shard}_mask.npy", mask)
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(
        cache_dir / f"{shard}_studies.csv", index=False
    )


def _labels_df(study_ids):
    return pd.DataFrame(
        {finding: [0.5] * len(study_ids) for finding in FINDINGS},
        index=pd.Index(study_ids, name="StudyInstanceUID"),
    )


def test_len_and_getitem_shapes_match_one_shard(tmp_path):
    study_ids = ["s0", "s1", "s2"]
    _write_shard(tmp_path, "train.s00of01", study_ids)
    ds = SlotCacheDataset(tmp_path, ["train.s00of01"], _labels_df(study_ids))

    assert len(ds) == 3
    images, mask, label = ds[0]
    assert images.shape == (6, 3, 4, 4)
    assert mask.shape == (6,)
    assert label.shape == (len(FINDINGS),)
    assert images.dtype == torch.float32


def test_getitem_returns_the_correct_study_across_shards(tmp_path):
    _write_shard(tmp_path, "train.s00of02", ["a0", "a1"], fill_value=0)
    _write_shard(tmp_path, "train.s01of02", ["b0", "b1"], fill_value=100)
    all_ids = ["a0", "a1", "b0", "b1"]
    ds = SlotCacheDataset(tmp_path, ["train.s00of02", "train.s01of02"], _labels_df(all_ids))

    assert len(ds) == 4
    # study b1 is index 3 overall (shard 1, local row 1), fill_value=100+1=101
    images, _, _ = ds[3]
    assert torch.allclose(images, torch.full((6, 3, 4, 4), 101.0 / 255.0))


def test_group_index_selects_the_requested_anchor(tmp_path):
    study_ids = ["s0"]
    cache_dir = tmp_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = np.zeros((1, 6, 9, 4, 4), dtype=np.uint8)
    for g in range(3):
        cache[0, :, g * 3:(g + 1) * 3] = g  # group 0/1/2 filled with 0/1/2
    np.save(cache_dir / "train.s00of01_cache.npy", cache)
    np.save(cache_dir / "train.s00of01_mask.npy", np.ones((1, 6), dtype=np.float32))
    pd.DataFrame({"StudyInstanceUID": study_ids}).to_csv(
        cache_dir / "train.s00of01_studies.csv", index=False
    )

    ds_centre = SlotCacheDataset(cache_dir, ["train.s00of01"], _labels_df(study_ids), group_index=1)
    images, _, _ = ds_centre[0]
    assert torch.allclose(images, torch.full((6, 3, 4, 4), 1.0 / 255.0))

    ds_low = SlotCacheDataset(cache_dir, ["train.s00of01"], _labels_df(study_ids), group_index=0)
    images, _, _ = ds_low[0]
    assert torch.allclose(images, torch.zeros(6, 3, 4, 4))


def test_study_ids_filter_restricts_to_the_given_subset(tmp_path):
    study_ids = ["s0", "s1", "s2", "s3"]
    _write_shard(tmp_path, "train.s00of01", study_ids)
    # labels_df only covers the subset -- would raise on the full 4 studies
    # without the study_ids filter narrowing the dataset down first.
    ds = SlotCacheDataset(
        tmp_path, ["train.s00of01"], _labels_df(["s1", "s3"]), study_ids=["s1", "s3"]
    )

    assert len(ds) == 2
    assert set(ds.study_ids) == {"s1", "s3"}


def test_raises_when_labels_df_does_not_cover_every_cache_study(tmp_path):
    study_ids = ["s0", "s1"]
    _write_shard(tmp_path, "train.s00of01", study_ids)

    with pytest.raises(ValueError):
        SlotCacheDataset(tmp_path, ["train.s00of01"], _labels_df(["s0"]))
