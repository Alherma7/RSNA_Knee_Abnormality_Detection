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
- **A2 v1** (slot-attention model): DINOv2-small backbone
  (`vit_small_patch14_dinov2.lvd142m`, fine-tuned, last 6 transformer
  blocks unfrozen) with per-finding masked-softmax attention over the 6
  A3 slots — full design in
  `docs/superpowers/specs/2026-08-26-a2-slot-attention-model-design.md`.
  Also fixed a real gap: `load_training_labels()` still called the old
  regex labeler (0.686) for weak studies instead of the A1a′-adopted
  published set (0.8927) — A1a′ decided that swap but never wired it in.
  Trained for real on Kaggle (fold 0 of 4 only, `group_index=1` centre
  anchor, none of the Tier B items below): **0.7689 gold macro-AUC** —
  beats Fase 5's old CV (0.5711) and the real LB reference (0.596) by a
  wide margin, matches/slightly exceeds Reference A's own full-pipeline
  OOF (0.7675). One real, unresolved caveat:
  `medial_meniscus_tear` scored below random (0.458) on this fold's 17
  gold validation studies — flagged, not yet explained, worth
  re-checking once the full 4-fold CV pools more gold per finding.
  Graduated `src/dataset.py::SlotCacheDataset` and
  `src/model.py::build_backbone()`/`build_multiplane_model()`/
  `SlotAttentionModel`. `src/train.py::run()` (the submission pipeline)
  is still `NotImplementedError` — out of scope for graduating this
  architecture.
- **A2 v1, pooled 4-fold CV**: fold 0 alone (0.7689) was optimistic — an
  easier split. Trained folds 1-3 for real on Kaggle (fold 1: 0.7480,
  fold 2: 0.8334, fold 3: 0.8282) and pooled all 4 folds' OOF predictions
  over the full 58 gold studies: **0.7512 macro-AUC — the trustworthy A2
  v1 baseline going forward**, superseding the single-fold number. Both
  fold-0-only caveats resolved as hypothesized small-sample artifacts:
  `medial_meniscus_tear` 0.458 (below random, 17 gold) → **0.6635** pooled
  (58 gold); `oa_lateral_compartment` undefined (single class, 17 gold) →
  **0.6325** pooled. No new `src/` graduation — this only recomputes the
  evaluation of the already-graduated A2 v1 architecture on more gold
  data; weakest pooled findings are `mcl_injury` (0.6145) and `synovitis`
  (0.6559).
- **A4** (submission pipeline): this is a Code Competition — the A3
  pre-built pixel cache can't cover the hidden test set (built before it
  existed), so `notebooks/07v1_a2_submission_inference.ipynb` decodes
  DICOM live instead, porting the same stevenleehans source A3 already
  cited, validated against the pre-built train cache before trusting it
  on hidden data (full design:
  `docs/superpowers/specs/2026-08-27-a4-submission-pipeline-design.md`).
  A real bug was found and fixed via the validation gate failing on
  real Kaggle data: `decode_study_slots()` normalized over only the
  centre anchor's 3 slices instead of the source's pooled 9-slice
  window — same pixels selected, wrong normalization statistics. Fixed;
  gate now matches the pre-built cache bit-for-bit on 13/15 (87%) gold
  studies, with a ~13% residual accepted as a known, uninvestigable-
  further limitation (no code defect found on inspection — most likely
  a decode-library/cache-build-environment difference for a small
  subset of studies). Submitted to the real competition 2026-08-27:
  **real leaderboard 0.834** — well above A2 v1's local pooled CV
  (0.7512) and Fase 5's old real-LB reference (0.596), confirming the
  slot-attention architecture generalizes to hidden data. Still a gap to
  public top scores (0.89–0.95+, see Current plan above), but the
  biggest jump of any single step so far.
- **A2 v2** (multi-group slot attention): A2 v1's pooled 4-fold CV found
  5 findings clustered at 0.61–0.66, well below the other 7 (0.79–0.88).
  A 2026-08-28 investigation (live Kaggle forum, public notebooks, two
  purpose-built visual diagnostics run on Kaggle) narrowed this to A3's
  cache sampling 3 anchor groups per slot but A2 v1 using only the
  centre one (`group_index=1`) — full design/evidence/gate:
  `docs/superpowers/specs/2026-08-28-a2v2-multigroup-slot-attention-design.md`.
  Replaced centre-only with all 3 anchor groups on all 6 slots (18
  pseudo-slots, `expand_slot_groups()`) — needed **zero**
  `src/model.py` changes (`n_slots` was already a plain constructor
  parameter), only a new data-layer reshape. Two-tier real gate, both
  run on Kaggle: **fold-0 pilot** (`notebooks/09v1_a2v2_multigroup_baseline.ipynb`,
  0.7956 gold macro vs. A2 v1's 0.7689 fold-0 baseline — macro check
  passed, but the paired `medial_meniscus_tear`/`lateral_meniscus_tear`
  directional read was inconclusive: +0.1392 vs. -0.0005) → user chose
  to scale to the remaining 3 folds anyway (an explicit judgement call,
  not automatic) → **pooled 4-fold gate**
  (`notebooks/10v1_a2v2_pooled_4fold_cv.ipynb`, fold 0 reused via
  checkpoint, folds 1-3 trained fresh): **pooled gold macro-AUC 0.8009**
  — beats A2 v1's pooled 0.7512 baseline by **+0.0497**, and all 4/4
  weak-cluster findings moved concordantly positive (`mcl_injury`
  0.6145→0.6599, `oa_lateral_compartment` 0.6325→0.7969 — the largest
  mover, and the finding with no fold-0 baseline at all,
  `medial_meniscus_tear` 0.6635→0.7007, `lateral_meniscus_tear`
  0.6584→0.6907 — the one whose fold-0 read was flat, resolved once
  pooled). Both spec section 5.2 checks passed clearly, not marginally
  → **graduated**: `src/features.py::expand_slot_groups()` and
  `src/dataset.py::SlotCacheDataset`'s new `expand_groups=True`
  parameter (default `False`, A2 v1's exact behaviour unaffected), plus
  the unit tests spec section 6 called for. **New A2 v2 baseline going
  forward: 0.8009 pooled gold macro-AUC.**
- **A2 v2 submission pipeline** (`notebooks/11v1_a2v2_submission_inference.ipynb`):
  ports A4 (`07v1`, real LB 0.834 with A2 v1's 6-slot checkpoints) to
  ensemble the 4 new A2 v2 checkpoints instead. `decode_study_slots()`
  now keeps all 3 anchor groups per slot (previously discarded 2 of 3
  after normalization) and reshapes via a hand-kept
  `expand_slot_groups()` before the model call — normalization math
  itself unchanged from `07v1`. Section 2's validation gate now compares
  the full 3-group decode against the pre-built cache (strictly harder
  than `07v1`'s centre-only check) and still matched **13/15 (86.7%)**
  gold studies bit-for-bit — essentially unchanged from `07v1`'s own 87%
  despite comparing 3x the pixels, real evidence the port introduced no
  regression (both mismatches are the same `lat=unknown` residual
  category as `07v1`'s, diffs spread evenly across all 3 groups rather
  than concentrated in a way a group-order bug would produce). Submitted
  to the real competition 2026-08-29: **real leaderboard 0.849** — up
  from A2 v1/A4's 0.834 (+0.015), confirming the local pooled-CV gain
  (0.7512→0.8009) replicates on real hidden test data, not just the
  58-gold local gate. **New real LB baseline: 0.849.**
- **Checkpoint ensembling probe** (`notebooks/12v1_ensemble_gold_validation.ipynb`,
  negative result): blended A2 v1's 4 fold checkpoints with A2 v2's 4
  (per-study, each architecture's own OOF fold checkpoint only — not an
  8-way blend), inference-only, no retraining, scored on the 58 gold
  studies. Best variant (`weighted_70_30_toward_a2v2`) reached 0.8047 vs.
  A2 v2 alone's 0.8009 — **+0.0038, roughly 1/13th this project's own
  ~0.03 pooled-macro noise floor**, and the per-finding pattern doesn't
  match a real ensembling mechanism (biggest gain on `acl_injury`, an
  already-strong finding; biggest losses on `oa_lateral_compartment` and
  `synovitis`, the two findings independently flagged elsewhere as this
  pipeline's most volatile/noisiest). **Not ported to the submission
  pipeline, no submission spent.**
- **A3 v2 window-widening pilot** (`notebooks/13v1_a3v2_window_cache_build.ipynb`
  + `notebooks/14v1_a2v3_window_only_fold0_pilot.ipynb`, inconclusive —
  not adopted): rebuilt the A3 pixel cache using the reference kernel's
  own wider per-plane `PLANE_WINDOW` defaults (Sagittal/Axial 0.10-0.90,
  Coronal 0.15-0.85) instead of the pinned narrow `RSNA_WINDOW=0.35,0.65`,
  everything else (18-pseudo-slot architecture, hyperparameters)
  unchanged from A2 v2. `13v1`'s cache build: both pre-build gates clean
  (fingerprint diff showed only the intended `window` field change,
  probe rate 7.0 slot-series/s inside the healthy range), 8/8 shards, 0
  decode failures. `14v1`'s fold-0 pilot, dual best-epoch/SWA checkpoint
  tracking applied for the first time
  (`feedback_checkpoint_selection_noise.md`): best-epoch gold macro-AUC
  **0.7726** vs. A2 v2's fold-0 baseline 0.7956 (delta -0.0230, within
  the ±0.03 noise tolerance — macro check passed), SWA gold macro
  **0.7677** (selection-noise gap between the two: +0.0049, this
  project's first real on-data measurement of that gap). Per the
  pre-agreed I4 gate (macro passes AND at least one of
  medial/lateral meniscus tear clears a ~0.10 delta): `medial_meniscus_tear`
  0.4028 (delta -0.1942, worse), `lateral_meniscus_tear` 0.7121 (delta
  +0.0601, positive but under the bar) — neither cleared it;
  `mcl_injury` (+0.1000) reported as context only, not gating.
  **DECISION: STOP — inconclusive, not disproved.** Not scaled to 4
  folds, nothing graduated to `src/`.

## Next steps

- [x] Scale A2 from 1 fold to the full 4-fold CV — done above, resolved
      the `medial_meniscus_tear`/`oa_lateral_compartment` fold-0
      artifacts.
- [x] Get a real leaderboard number for A2 v1's architecture (A4,
      submission pipeline) — done above: **real LB 0.834**.
- [x] Investigate the weakest pooled findings (`mcl_injury` 0.6145,
      `synovitis` 0.6559) and the anchor-group question together (A2
      v2) — done above: multi-group slot attention graduated, new
      pooled baseline **0.8009**. `synovitis` (label-noise, not
      architecture — see the 2026-08-28 investigation) wasn't targeted
      by this change and moved anyway (0.6559→0.7873), plausibly a
      side effect of more slice coverage per finding generally.
- [x] Port A4's submission pipeline to A2 v2's 18-pseudo-slot inputs —
      done above (`11v1`): **real LB 0.849**, new baseline.
- [x] Checkpoint/rank ensembling (A2 v1 + A2 v2 blend) — tried, negative
      result, see above (`12v1`). Not pursued further.
- [x] `RSNA_WINDOW` widening (A3 v2/A2 v3) — tried, see above
      (`13v1`/`14v1`): inconclusive, not adopted, not scaled to 4 folds.
- [ ] Tier B items still open: drop `pos_weight`, EMA at 0.997, re-open
      augmentation, RadImageNet domain pretraining (not backbone size —
      measured null, see the artifact), compartment-aware attention.
- [ ] The 224px→336px resolution fix (deferred out of A2 v2's scope,
      needs a full A3 cache rebuild from raw DICOM) remains open — the
      other half of the original weak-finding diagnosis, not addressed
      by the anchor-group change or the window-widening pilot.
- [ ] ROI/spatial-attention pooling around the joint line (from
      Berat Kirbiyik's forum ablation — masked-softmax-then-pool may be
      losing thin/small findings to global pooling) — a more specific
      lead for the meniscus/MCL/lateral-OA cluster than resolution or
      window width alone.
