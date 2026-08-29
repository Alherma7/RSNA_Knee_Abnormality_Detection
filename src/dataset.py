"""PyTorch Dataset wrapping the A3 slot-attention pixel cache.

Source: A2 v1 design,
docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md,
section 2.2 — validated end-to-end on real Kaggle data 2026-08-26
(notebooks/05v2_slot_attention_baseline.ipynb), reaching 0.7689 gold
macro-AUC on fold 0. This module reuses src.data.load_slot_cache_shard()
and src.features.select_group() directly — unlike the Kaggle notebook,
which duplicates this logic inline, since Kaggle can't import this repo
(confirmed with the user 2026-08-26).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.data import load_slot_cache_shard
from src.features import expand_slot_groups, select_group


class SlotCacheDataset(Dataset):
    """One sample = one study: (slot_images, slot_mask, label_row).

    Opens each shard's cache as a memory-map via load_slot_cache_shard()
    (lazy — never loads the full ~12GB train cache into RAM at once).
    `group_index` selects which anchor group(s) feed the model, via
    select_group() — group 1 (centre anchor) is what A2 v1 validated,
    but is passed through as a plain parameter (not hardcoded), since
    which group(s) to use is deliberately still an open question, see
    select_group()'s own docstring.

    `labels_df` must be indexed by StudyInstanceUID with config.FINDINGS
    columns (e.g. src.data.load_training_labels()'s output) and must
    cover every study loaded from `shards`, unless `study_ids` restricts
    the dataset to a known-covered subset (e.g. a tiny debug/smoke-test
    slice) — raises ValueError otherwise, same coverage-guard reasoning
    as load_published_labels()/load_slot_cache_shard().

    Graduated 2026-08-26 from
    notebooks/05v2_slot_attention_baseline.ipynb (A2), where this exact
    construction (real cache, real labels, 3,100 train studies) trained
    without error across 12 real epochs on Kaggle.

    `expand_groups` (A2 v2, graduated 2026-08-29): when True, each item
    is all 3 anchor groups on all 6 slots (18 pseudo-slots) via
    src.features.expand_slot_groups(), instead of one group via
    select_group(group_index). Default False preserves A2 v1's exact
    behaviour byte-for-byte. `group_index` is ignored when
    `expand_groups=True` and must be left at its default (1) — passing
    any other value raises, since there is otherwise no way to tell
    "not passed" from "passed as 1" apart. See
    docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md
    section 2.2.
    """

    def __init__(self, cache_dir, shards, labels_df, group_index=1, expand_groups=False, study_ids=None):
        if expand_groups and group_index != 1:
            raise ValueError(
                "expand_groups=True ignores group_index; pass group_index=1 (default) or omit "
                f"it, not group_index={group_index!r}"
            )
        self.group_index = group_index
        self.expand_groups = expand_groups
        caches, masks, all_study_ids, shard_of, local_idx = [], [], [], [], []
        for shard_idx, shard in enumerate(shards):
            cache, mask, ids = load_slot_cache_shard(cache_dir, shard)
            caches.append(cache)
            masks.append(mask)
            all_study_ids.append(np.asarray(ids))
            shard_of.append(np.full(len(ids), shard_idx))
            local_idx.append(np.arange(len(ids)))

        self.caches = caches
        mask_all = np.concatenate(masks, axis=0).astype(np.float32)
        study_ids_all = np.concatenate(all_study_ids)
        shard_of_all = np.concatenate(shard_of)
        local_idx_all = np.concatenate(local_idx)

        if study_ids is not None:
            keep = np.isin(study_ids_all, np.asarray(list(study_ids)))
            mask_all, study_ids_all = mask_all[keep], study_ids_all[keep]
            shard_of_all, local_idx_all = shard_of_all[keep], local_idx_all[keep]

        self.mask = mask_all
        self.study_ids = study_ids_all
        self.shard_of = shard_of_all
        self.local_idx = local_idx_all

        aligned = labels_df.reindex(self.study_ids)[config.FINDINGS]
        if aligned.isna().any().any():
            missing = self.study_ids[aligned.isna().any(axis=1).to_numpy()]
            raise ValueError(f"{len(missing)} cache studies missing labels, e.g. {missing[:5]}")
        self.labels = aligned.to_numpy(dtype=np.float32).copy()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, i):
        shard_idx, row = self.shard_of[i], self.local_idx[i]
        full = self.caches[shard_idx][row]  # (6, 9, 224, 224) uint8
        if self.expand_groups:
            selected, slot_mask_row = expand_slot_groups(full, self.mask[i])  # (18, 3, 224, 224), (18,)
        else:
            selected = select_group(full, self.group_index)  # (6, 3, 224, 224)
            slot_mask_row = self.mask[i]
        images = torch.from_numpy(np.ascontiguousarray(selected)).float() / 255.0
        mask = torch.from_numpy(slot_mask_row)
        label = torch.from_numpy(self.labels[i])
        return images, mask, label
