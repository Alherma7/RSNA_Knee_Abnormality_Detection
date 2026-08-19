# Fase 5 (training) — gold+weak training test

Status: design, not yet executed. Revised 2026-08-19 after an Opus-model
review against README.md/RESOURCES.md's recorded findings (see Section 9).
Date: 2026-08-19
Related: README.md (Fase 5 infra, already graduated), src/data.py::load_training_labels,
src/labelers.py::label_reports/report_group_key, docs/superpowers/specs/2026-08-17-fase4-baseline-cnn-design.md,
notebooks/04_baseline_cnn.ipynb, notebooks/04b_gold_weak_groupkfold.ipynb

## 1. Objective

The Fase 4 corrected baseline (single sagittal plane, EfficientNet-B0,
58 gold studies only) reached macro ROC-AUC 0.532 (std 0.021, 3-fold),
with severe overfitting (train AUC 0.95-0.98) diagnosed as a training-set
*size* problem, not an architecture problem (see README's 2026-08-18 MIL
entry: tripling views per study via multi-plane pooling gave no clear
win — 0.530 vs 0.532 — while train AUC stayed just as high). Fase 5's
label/split infrastructure (`load_training_labels()`,
`report_group_key()`) is already graduated and tested, but nothing has
been trained on the 4,349 weak-labeled studies yet — this design covers
that missing piece: does adding them break the 0.53 ceiling?

Per this project's own graduation gate (`structuring-ml-projects`, step
6), the Fase 4 image pipeline (`normalize_laterality`,
`normalize_physical_scale`, `build_25d_triplet`, etc.) has only been
validated against the gold-only baseline — not against a run that
actually uses weak data — so it stays in the notebook until *this*
experiment shows a win, rather than graduating on the strength of the
gold-only result alone.

## 2. Decisions locked in during brainstorming (2026-08-19)

| Decision | Choice | Why |
|---|---|---|
| Weak label 0.5 (labeler abstention) | Soft target in `BCEWithLogitsLoss` | Simpler than per-column masking; lets the model express uncertainty directly rather than the loss silently dropping ~90%-silence-rate findings like `oa_lateral_compartment` from most weak rows. |
| Validation set per fold | Gold rows only (`fold==k & is_gold`) | Weak graded labels come from a labeler with only 0.686 AUC against gold (Fase 3) — validating against them would measure agreement with a noisy labeler, not truth. Keeps this run's macro-AUC directly comparable to Fase 4's 0.532 (also gold-only). |
| Plane(s) | Single sagittal plane only (same as Fase 4) | The 2026-08-18 MIL experiment already showed multi-plane pooling doesn't help at this data scale and confirmed the bottleneck is study *count*. Changing planes AND adding data at once would confound which change caused any AUC delta. |
| Scale | All 4,349 weak studies (no pilot subset) | User's explicit call — go straight to the full set rather than a 500-1000 pilot. |
| Preprocessing/training split | Two notebooks: preprocess to a Kaggle Dataset, then a separate training notebook mounts it | Kaggle sessions have a GPU time budget; decoding+normalizing 4,349 DICOM series is expensive enough that it should happen once, not on every training retry (hyperparameter change, bug fix). |
| Validation aggregation | Pooled out-of-fold (OOF): concatenate each fold's gold-only held-out predictions into one 58-row table, compute **one** `macro_roc_auc` over it | 5-fold `GroupKFold` (not stratified) leaves gold split 16/11/14/8/9 (Fase 5 infra); the smallest folds risk a single-class column for a rare finding (MCL, 9 positives total), which raises in `macro_roc_auc` by design. Pooling avoids that failure mode and gives one number, still comparable in magnitude to Fase 4's 0.532. |
| Gate comparison | Run a **gold-only control arm** under this exact protocol (5-fold `GroupKFold`, pooled OOF), not just against Fase 4's recorded 0.532 | Fase 4's 0.532 came from a *different* protocol (3-fold `MultilabelStratifiedKFold`, mean-of-folds), so comparing the gold+weak run directly against it would conflate a protocol change with the data change. The 58 gold studies are already preprocessed, so this control is nearly free — see Section 5. |
| Validation seeds | 3 seeds, report mean+std of the pooled OOF macro-AUC (both arms) | Same standard the 2026-08-18 MIL experiment used (3 seeds x 3 folds), after finding epoch-to-epoch/fold-to-fold noise was the same order of magnitude as the effects being measured (fold 1 alone ran 0.59-0.63 vs. 0.48-0.55 elsewhere). A single seed here risks the same misread. |

## 3. Data pipeline

### 3.1 Series selection

Identical rule to Fase 4 Section 4.1, applied to all 4,407 studies (not
just the 58 gold): prefer `Fluid_Sensitive=True` sagittal series, tie-break
by most slices, then by lexicographically smallest `SeriesInstanceUID`.
This is a pure function of `train_series.csv` — no gold-specific
assumption — but has only been exercised against the 58 gold studies so
far; Notebook A's first job is confirming it resolves cleanly (exactly
one series) across the full 4,407, and logging how often each tie-break
branch fires at this larger scale.

### 3.2 Preprocessing (Notebook A — "05a_weak_dicom_preprocess.ipynb")

Reuses the corrected Fase 4 pipeline unchanged: order slices by
`SliceLocation`, apply the laterality list-order fix (sagittal:
reverse slice order by `Laterality`, not pixel flip — see Fase 4 audit
finding (1)), `normalize_physical_scale` (target 0.35mm/pixel),
`center_crop_or_pad` (`CROP_MM=130.0`), one center 2.5D triplet per
study (`GAP_MM=4.0`, same physical-mm gap logic as Fase 4 Section 4.2),
intensity normalization via `apply_voi_lut` with the percentile
fallback (Fase 4 audit fix (3)).

For each of the 4,407 studies: decode, preprocess, save as one `.npy`
file (float32, shape `(3, 371, 371)`) to `/kaggle/working/triplets/`.
Estimated size: ~1.65MB/study x 4,407 ≈ 7.3GB — fits comfortably in a
Kaggle Dataset. On completion, `Save Version` and publish as a private
Kaggle Dataset (e.g. `rsna-knee-weak-sagittal-triplets`).

Log any per-study preprocessing failures (missing sagittal series,
decode error) with the `StudyInstanceUID` rather than letting one bad
study kill the whole run — report a failure count/list at the end
rather than crashing partway through 4,407 studies.

### 3.3 Training (Notebook B — "05b_gold_weak_training.ipynb")

Mounts the Notebook-A Dataset as input. Loads `load_training_labels()`
for the label/fold table (already graduated in `src/data.py`). For each
of the 5 folds: train on all rows (gold+weak) not in that fold, predict
on that fold's **gold-only** rows, collect those predictions. After all
5 folds, concatenate the 5 held-out gold prediction sets (58 rows total)
and compute one `macro_roc_auc`/`per_finding_roc_auc` (`src/evaluate.py`,
already implemented/tested, reused as-is — no change needed for OOF
pooling, since it just receives whatever rows are passed in).

## 4. Model and training regimen

Unchanged from the Fase 4 winning configuration — reusing the same
config is deliberate (Section 2): EfficientNet-B0 (`timm`,
ImageNet-pretrained), differential LR (backbone 1e-5, head 1e-3),
dropout 0.5, weight_decay 1e-2, fixed epoch budget (8, no
checkpoint-selection-by-val — the Fase 4 audit's metric-leakage fix),
`torch.manual_seed(42)`, no augmentation (Fase 4's augmentation
experiment was a confirmed negative result, not revisited here since
this experiment already isolates a different variable).

**Loss, extended for soft targets:** `BCEWithLogitsLoss` with each row's
raw target (0.0/0.5/1.0) used as-is — abstentions still pull the
prediction toward 0.5 rather than being excluded from the loss (Section
2 decision). `pos_weight` per finding is `n_negative/n_positive` (Fase
4 §6's PyTorch formula — an earlier draft of this spec had the ratio
inverted, caught in review), computed **only over that finding's
hard-labeled rows** (`target != 0.5`) in the training fold:
`n_positive = (target == 1.0).sum()`, `n_negative = (target == 0.0).sum()`.
Computing it over all rows (counting each abstention as 0.5 positive +
0.5 negative) would dilute the correction exactly where it matters most:
`oa_lateral_compartment` abstains on ~90% of reports (Fase 3), so most of
its weak rows would sit at target 0.5 and its `pos_weight` would collapse
toward 1.0 regardless of the column's real class balance among the rows
that do carry information. Restricting the count to hard-labeled rows
keeps the imbalance correction meaningful without changing which rows
contribute to the loss itself (that's still every row, per Section 2).

**Batch composition:** gold and weak rows are mixed within the same
training fold with no reweighting between them (58 gold vs. ~3,479 weak
per fold, roughly 60:1) — noted as an open risk (Section 7), not a
decision to solve upfront; if the weak-label noise (labeler AUC 0.686)
turns out to swamp the gold signal, a follow-up experiment could
upweight gold rows or downweight low-confidence weak rows, but that's
premature before seeing whether the unweighted version helps or hurts
at all.

**Diagnostic logging (added in review):** log pooled **train**
macro-AUC (over the training-fold rows actually used, hard-labeled cells
only for a well-defined score) alongside the pooled val macro-AUC, for
every seed/fold. The training-set size grows ~75x versus Fase 4 (58 to
~4,400 studies per fold) while the epoch budget stays at 8 — that changes
the total number of gradient steps by roughly the same factor at an
unchanged learning rate, which could shift the model from Fase 4's
severe-overfitting regime (train AUC 0.95-0.98) toward a different
regime entirely. Without logging train AUC there is no way to tell
whether a result (win or not) reflects the weak data's signal/noise
tradeoff or simply a step-count/LR mismatch — this is exactly the
diagnostic that drove both the Fase 4 and MIL conclusions, so it isn't
optional here. Epoch budget itself is **not** re-tuned in this run
(Section 7) — logging first, decide whether to revisit it in a follow-up.

## 5. Validation

5-fold `GroupKFold` over all 4,407 studies, grouped by
`report_group_key()` — already computed by `load_training_labels()`, no
change needed. Per fold: train on non-fold rows (gold+weak), predict on
the fold's gold-only rows. Pool all 5 folds' held-out gold predictions
(58 rows) and compute one `macro_roc_auc` (Section 2 rationale). Repeat
for 3 seeds; report mean+std of the pooled macro-AUC across seeds
(Section 2).

**Control arm (added in review):** run the identical protocol — same 5
folds (grouped by `report_group_key()`, gold rows only decide which fold
each gold study lands in, since weak rows aren't present in this arm),
same pooled-OOF aggregation, same 3 seeds — training on **gold-only**
rows instead of gold+weak. This reuses the 58 already-preprocessed gold
triplets (no new preprocessing) and isolates the protocol change (3-fold
stratified/mean-of-folds -> 5-fold GroupKFold/pooled-OOF) from the data
change (58 -> ~4,400 training studies), so the two are never conflated
into one number.

**Gate:** the gold+weak arm must beat **this run's own gold-only control
arm** (not Fase 4's raw 0.532, which used a different protocol) to be
considered a win. Fase 4's 0.532 is reported alongside as historical
context, not as the gate threshold itself. Logged either way (win or
negative result) per `structuring-ml-projects`'s convention. If it
doesn't win, this stays in the notebook; nothing graduates to
`src/features.py`/`src/data.py`'s DICOM-loading stubs on the strength of
this result alone.

## 6. Testing strategy

No new `src/` functions are introduced by this experiment — it reuses
already-graduated/tested code (`load_training_labels`,
`report_group_key`, `macro_roc_auc`, `per_finding_roc_auc`) plus
notebook-only preprocessing/training code, per this project's
notebook-before-src rule. If this experiment wins and the image pipeline
graduates afterward, that graduation (and its tests) is a separate,
follow-up step — same as Fase 4 left graduation pending user review.

## 7. Explicitly out of scope

- Re-sweeping Fase 4's hyperparameters (`GAP_MM`, `TARGET_MM_PER_PIXEL`,
  `CROP_MM`, dropout, weight_decay, epoch budget) — reused as-is to
  isolate the more-data variable (Section 2). Of these, `CROP_MM=130.0`
  has a real source (pilkwang, measured to cover 99.6% of FOV);
  `TARGET_MM_PER_PIXEL=0.35` does not (Fase 4 §11 already flagged it as
  an unsourced value chosen at n=58) — a candidate lever for a later
  iteration given Fase 1 measured native spacing as fine as 0.137mm/px,
  but out of scope here.
- EMA of model weights — the 2026-08-18 MIL experiment's largest positive
  delta on record (+0.019, 0.549 vs. 0.530), but tuned there
  (`decay=0.9`) specifically for ~40 gradient steps total at n=58; at
  ~4,400 studies/fold the standard `decay=0.999` would apply instead, and
  introducing it now would confound whether any AUC change came from more
  data or from EMA. Deferred to a follow-up on top of this experiment's
  result, same treatment Fase 4 gave augmentation (only added after the
  un-augmented baseline was in).
- Multi-plane pooling (already tested 2026-08-18, no clear win; would
  also confound this experiment's variable).
- Gold/weak reweighting or confidence-based weak-row filtering — noted
  as a follow-up if the unweighted mix underperforms (Section 4).
- Graduating any pipeline code to `src/` — contingent on this
  experiment's result (Section 5 gate).

## 8. Open risks (carried forward, not blocking)

- ~60:1 weak:gold ratio per fold with no reweighting (Section 4) — if
  weak-label noise dominates, the effect could go either direction and
  won't be diagnosable from the macro-AUC number alone; per-finding AUC
  breakdown (`per_finding_roc_auc`) should be inspected alongside the
  macro number to see whether specific findings (e.g. the weaker-labeled
  `oa_lateral_compartment`, 0.553 labeler AUC per Fase 3) drag the result
  down disproportionately.
- Preprocessing 4,407 studies' DICOM (vs. 58 in Fase 4) is untested at
  this scale — series-selection edge cases (ties, fallbacks) and DICOM
  format edge cases (transfer syntax, corrupt files) that never appeared
  in the 58-study sample could appear here; Section 3.2's per-study
  failure logging is the mitigation, not a guarantee nothing needs a
  follow-up fix mid-run.
- Kaggle GPU session time budget for training on ~4,400 studies x 8
  epochs x 5 folds x 3 seeds x 2 arms (gold+weak and the gold-only
  control, Section 5) is unmeasured and now a bigger unknown than in the
  first draft of this spec — Notebook B may need to be split further
  (e.g. one fold per session, or the control arm run separately since it
  only needs the 58 gold triplets) if a single session's time limit is
  insufficient; not designed here since the actual per-epoch cost at this
  scale is unknown until Notebook A's preprocessing is done and a first
  training epoch is timed.

## 9. Review pass (2026-08-19)

Before building anything, this spec was reviewed by a separate Opus-model
agent against README.md/RESOURCES.md's actual recorded findings (MRNet/
RadImageNet backbone-capacity conclusion, the 2026-08-18 MIL experiment's
methodology and result, and the `dangnh0611` RSNA breast cancer 1st-place
writeup's "soft positive label" idea). Findings and resolutions:

- **`pos_weight` formula was inverted** in the first draft
  (`n_positive/n_negative` instead of PyTorch's documented
  `n_negative/n_positive`) — a real bug, not a design choice; fixed in
  Section 4.
- **Gate compared two protocols, not one variable** — the first draft's
  gate compared this run's 5-fold/pooled-OOF number directly against
  Fase 4's 0.532 from a 3-fold/mean-of-folds protocol, conflating a
  protocol change with the data change under test. Fixed by adding the
  gold-only control arm (Section 5), run under this experiment's exact
  protocol.
- **Single seed risked repeating the MIL experiment's own lesson** —
  fixed by adopting the same 3-seed standard already used there
  (Section 2, Section 5).
- **`pos_weight` over all cells (incl. abstentions) dilutes the
  correction on high-abstention columns** — fixed by restricting the
  count to hard-labeled cells only, without changing which rows
  contribute to the loss itself (Section 4).
- **Backbone/resolution capacity** (RadImageNet finding, `dangnh0611`'s
  much larger backbone) — evaluated and *not* changed: the project's own
  MRNet/RadImageNet conclusion was that backbone size wasn't the
  bottleneck at n=58 specifically because overfitting dominated there:
  re-sweeping backbone/resolution at the same time as adding ~75x more
  training data would leave no way to attribute any AUC change to either
  variable. The new train-AUC logging requirement (Section 4) is the
  mechanism for deciding, after this run, whether that conclusion still
  holds at the new data scale and backbone capacity is worth revisiting
  in a follow-up.
- **EMA** — evaluated and deferred, not adopted now, for the same
  confounding-variable reason (Section 7).
- **`dangnh0611`'s "soft positive label" (0.8-0.9 for a positive with
  weaker evidence)** — not adopted as a direct substitute for this
  project's 0.5-abstain convention: that writeup's soft label shrinks a
  *known positive* toward 1.0 based on evidence strength, while this
  project's 0.5 represents the labeler declining to make a call at all —
  different semantics, not a like-for-like swap. The pos_weight fix
  above addresses the specific interaction the review raised (abstentions
  diluting imbalance correction) without adopting the mammography
  writeup's technique wholesale.
