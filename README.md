# RSNA Knee Abnormality Detection
<img width="559" height="280" alt="header" src="https://github.com/user-attachments/assets/1d5f97a3-e23f-4966-9797-75a20c2ff4fa" />

Detect 12 clinical findings in multiplanar knee MRI, training with a
small subset of officially-labeled ("gold") studies plus the rest
labeled only from free-text radiology reports. Metric: **macro
ROC-AUC** (unweighted mean of the 12 per-finding AUCs).

See `RESOURCES.md` for every technique's source, and the linked plan
below for full phase-by-phase detail and sourcing.

Working convention: code is validated in `notebooks/` before moving to
`src/`. No `src/` function is considered done until it has: (a) a
docstring citing its source, (b) validation against this project's
metric, (c) a measured improvement over a baseline, and (d) a test in
`tests/`.

## Key fact that drives the design

Of 4,407 training studies, only a small subset (58, confirmed exact)
has official labels; the rest have only the report. Also, `train.csv`
has a `Report` column but `test.csv` does not — text only exists at
train time. Direct consequence: report text is a training-label source
(weak supervision), never a model input at inference.

## Current plan (2026-08-25 reorientation)

After the original Fase-numbered roadmap reached real leaderboard 0.596
(see History below) against public top scores of 0.89–0.95+, the plan
was rebuilt from scratch by mining this competition's own discussion
forum (post 735304 and ~15 other threads) plus the RSNA Screening
Mammography Breast Cancer Detection competition's public solutions.
Full detail, sourcing, and effort estimates:
**[the strategy artifact](https://claude.ai/code/artifact/1fec15df-75b4-43dc-8b6e-88f0c8c8147a)**
(canonical reference — not duplicated here).

Priority order: **A0 (fix the CV gate) → A0b (verify slice ordering) →
A1a′ (validate published label sets) → A1a/A1b (build our own, only if
A1a′ falls short) → A3 (rebuild preprocessing in `src/`) → A2 (6 physical
slots + attention vs. a random slice bag) → Tier B items** (drop
`pos_weight`, EMA at 0.997, re-open augmentation, checkpoint/rank
ensembling, RadImageNet domain pretraining).

The 11 Fase-numbered notebooks (`00`–`06b`) that produced the History
results below have been retired from the repo (recoverable via `git
log` before this cleanup if ever needed) — superseded by `NNv2_*`
notebooks going forward, one per plan item.

## History

- **Fase 0–3** (schema/DICOM/report EDA, regex labeler): metric pinned
  (`src/evaluate.py::macro_roc_auc`), `src/labelers.py` graduated —
  0.686 macro-AUC vs. the 58 gold studies.
- **Fase 4** (baseline CNN, 58 gold studies only): 0.532 macro-AUC
  (3-fold CV), after an independent audit caught and fixed 3 real bugs
  (a laterality-flip bug, a metric-selection leak worth +0.042, and
  missing intensity normalization). Multi-plane MIL pooling was tested
  and showed no clear win at this scale (fold identity dominated over
  architecture at n=58).
- **Fase 5** (gold+weak `GroupKFold`, all 4,407 studies): 0.5711
  macro-AUC (3-seed mean CV). A calibration submission scored **real
  leaderboard 0.596** — above local CV, confirming the gate was
  informative, not disconnected from the real metric — but the gap to
  public top scores (0.89–0.95+) triggered the 2026-08-25 reorientation.
- **A0b** (verify DICOM slice ordering): filename/bare-`SliceLocation`
  order was wrong on 41 of 58 (71%) real gold studies. Fixed with a
  geometric sort — `src/data.py::geometric_slice_order()`.
- **A0** (fix the CV gate): report-only `GroupKFold` leaked scanner
  identity across train/val for 4,399 of 4,407 studies (99.8%). Fixed
  with combined report-template + scanner-fingerprint grouping —
  `src/data.py::build_group_ids()`/`build_scanner_fingerprints()` — plus
  a `worse_of_two`/`per_label_gate` second-gauge selection rule
  (`src/evaluate.py`) and `CV_FOLDS` 5→4 (better gold-per-fold balance).
- **A1a′** (validate published label sets): a forum-referenced published
  LLM label set scores 0.878–0.893 macro-AUC vs. our 58 gold studies —
  a decisive win over our own labeler's 0.686. **A1a (port a lexicon)
  and A1b (build our own LLM pass) are skipped as unnecessary.**
- **A3** (preprocessing): reused a published slot-attention pixel cache
  (stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache) instead of
  rebuilding our own DICOM pipeline, given this project's own
  preprocessing track record (see RESOURCES.md). Validated against the
  real cache on Kaggle: `cache_meta.json` matches the source's build
  config exactly, all 4,407 studies + all 58 gold present across the 4
  train shards, and a visual check of decoded images across all 6 slots
  confirmed correctly-oriented real knee MRI. Graduated
  `src/data.py::load_slot_cache_shard()` and
  `src/features.py::select_group()`.

## Next steps

- [ ] **A2** — 6 physical slots (plane × fluid-sensitivity × fat-sat)
      with per-finding attention, as an A/B against a large random
      slice bag. Which anchor group(s) (`select_group`'s `group_index`)
      to feed the model is an open sub-question, deliberately not
      pre-committed in A3 — see `src/features.py::select_group()`'s
      docstring.
- [ ] Tier B items once A2 lands: drop `pos_weight`, EMA at 0.997,
      re-open augmentation, checkpoint/rank ensembling, RadImageNet
      domain pretraining (not backbone size — measured null, see the
      artifact).
