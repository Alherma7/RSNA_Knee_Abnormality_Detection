# Fase 5 (training) — gold+weak training test Implementation Plan

> **For agentic workers:** This plan is a lightweight cell-by-cell
> checklist, not a code-bearing task plan — deviation from
> `superpowers:writing-plans`'s default format, agreed with the user
> 2026-08-19, because this project's established convention (see
> `feedback_notebooks_built_cell_by_cell` memory) is that Kaggle notebook
> code is written and validated one cell at a time against real Kaggle
> output, never pre-authored. Execute this plan by proposing one cell at
> a time to the user in a Kaggle session, waiting for real run output
> before moving to the next checklist item. Do NOT use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` batch execution for the notebook cells —
> those assume pre-written code steps this plan deliberately doesn't
> contain. Check off each box only after the user has run that cell on
> Kaggle and the output matches (or the discrepancy has been resolved
> and logged, same as every prior phase in this project).

**Goal:** Answer whether adding the 4,349 weak-labeled studies to
training breaks the Fase 4 gold-only 0.532 macro-ROC-AUC ceiling, via
two Kaggle notebooks (DICOM preprocessing, then training+evaluation).

**Architecture:** Notebook A preprocesses all 4,407 studies' sagittal
DICOM into cached 2.5D triplets, published as a Kaggle Dataset. Notebook
B mounts that dataset, builds the gold+weak label/fold table, and runs
two evaluation arms (gold-only control, gold+weak) x 3 seeds x 5 folds,
each arm producing one pooled out-of-fold macro-ROC-AUC over the 58 gold
studies.

**Tech Stack:** Kaggle Notebooks (GPU), PyTorch, `timm` (EfficientNet-B0),
`pydicom`, `pandas`, `scikit-learn` (`GroupKFold`, `roc_auc_score`) — same
stack as Fase 4, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-19-fase5-weak-training-design.md`
(and `docs/superpowers/specs/2026-08-17-fase4-baseline-cnn-design.md` for
the preprocessing/model config this plan reuses unchanged).

## Global Constraints

- Reuse Fase 4's winning config verbatim, no re-sweeping: `GAP_MM=4.0`,
  `TARGET_MM_PER_PIXEL=0.35`, `CROP_MM=130.0`, EfficientNet-B0,
  differential LR (backbone `1e-5`, head `1e-3`), dropout `0.5`,
  `weight_decay=1e-2`, 8 fixed epochs, no augmentation, no EMA (spec §4, §7).
- `pos_weight = n_negative/n_positive` per finding, counted over
  **hard-labeled rows only** (`target != 0.5`) in the training fold (spec
  §4 — this was a bug in the spec's first draft, now fixed; get this
  right the first time in code).
- Soft targets (0.0/0.5/1.0) go into `BCEWithLogitsLoss` as-is — never
  masked out of the loss (spec §2, §4).
- Validation predicts/scores **gold rows only**, per fold (spec §2, §5).
- 3 seeds x 5 folds x 2 arms (gold-only control, gold+weak) — 30 total
  training runs; pool each arm's 5-fold gold-only OOF predictions per
  seed before computing `macro_roc_auc` (spec §5).
- Gate: gold+weak arm's pooled OOF macro-AUC (mean over 3 seeds) must
  beat the gold-only control arm's (spec §5) — not Fase 4's raw 0.532,
  which used a different CV protocol.
- Log pooled train macro-AUC alongside val, every seed/fold (spec §4).
- No `src/`/`tests/` changes in this phase (spec §6) — everything stays
  notebook-only until the gate is decided.
- Notebooks are self-contained (no `from src import ...`) — Kaggle
  doesn't mount this repo, same convention as notebooks 00-04b. Where
  logic already exists in `src/` (e.g. `load_training_labels`,
  `report_group_key`, `label_reports`, `macro_roc_auc`), port it into the
  notebook by copying the function body, and sanity-check the result
  against the numbers already verified for that `src/` function (listed
  per task below) rather than re-deriving the logic from scratch.

---

## Notebook A — `05a_weak_dicom_preprocess.ipynb`

Produces one `.npy` 2.5D triplet per study (all 4,407 — gold and weak
both, so Notebook B's control arm can reuse them with zero extra
preprocessing) and publishes them as a private Kaggle Dataset.

- [x] **Cell 1 — Imports + config.** `pydicom`, `numpy`, `pandas`, path
      constants for the Kaggle-mounted competition data
      (`/kaggle/input/competitions/rsna-knee-abnormality-detection`, per
      Fase 1) and the output dir (`/kaggle/working/triplets/`).
      Hyperparameter constants from Global Constraints
      (`CROP_MM`, `GAP_MM`, `TARGET_MM_PER_PIXEL`).

- [x] **Cell 2 — Load and sanity-check `train.csv`/`train_series.csv`.**
      Confirm row counts match what Fase 1/Fase 5 already measured: 4,407
      studies in `train.csv`, all with non-null `Report`; every study has
      >=1 sagittal series in `train_series.csv`. This is a scale check,
      not new discovery — Fase 1 already measured "all 4,407 training
      studies have all 3 planes present."

- [x] **Cell 3 — Series selection function (spec §3.1).** Port the exact
      Fase 4 §4.1 rule (prefer `Fluid_Sensitive=True` sagittal, tie-break
      by most slices, then lexicographically smallest
      `SeriesInstanceUID`) from `notebooks/04_baseline_cnn.ipynb` — same
      function, not a rewrite.

- [x] **Cell 4 — Run series selection over all 4,407 studies.** Log how
      often each branch fires (exact match / tie-break / no-`Fluid_Sensitive`
      fallback) as counts and percentages. Compare the shape of the
      distribution against Fase 4's 58-study numbers (56/58 had
      `Fluid_Sensitive=True` sagittal available, 8/58 needed the
      most-slices tie-break, 2/58 needed the fallback) — flag in the
      notebook (markdown cell) if the full-scale percentages look wildly
      different, since that would mean the rule behaves differently
      outside the 58-study sample it was designed against.

- [x] **Cell 5 — Slice ordering + laterality fix.** Reconstructed from
      README's audit description (2026-08-19: the actual corrected
      notebook cells from the 2026-08-18 audit were never synced back to
      this repo — only the pre-audit `04_baseline_cnn.ipynb` and the
      README prose survive locally; user confirmed reconstructing from
      the README description rather than pulling from Kaggle).
      **Real finding during this cell (systematic-debugging root-cause
      pass, 2026-08-19):** unlike the 58 gold studies (0/58 missing,
      confirmed in the 2026-08-18 audit), 2,218/4,407 (50.3%) selected
      sagittal series have no usable DICOM `Laterality` value.
      `ImageLaterality` (the alternate per-image tag) doesn't exist in
      this data; neither `train.csv` nor `train_series.csv` carries a
      laterality column. Root cause: a genuine large-scale metadata gap
      in the raw weak-study DICOM, not an artifact of the 30-study Fase 1
      sample or a wrong-tag lookup — partly (19% of the missing cases,
      426 studies, also touching 4/58 gold) explained by an
      anonymization-looking placeholder (`SeriesDescription ==
      "DummySeriesDesc!"`), the rest genuinely unpopulated. Resolution
      (user decision): a `SeriesDescription` text-fallback parser
      (`L-`/`R-`/`LT.`/`RT.` prefixes, `LEFT`/`RIGHT` as a word) recovers
      308 more (spot-checked clean, 15/15 correct) — final split: 2,188
      dicom_tag (49.7%), 308 series_description (7.0%), 1,911 unknown
      (43.4%, trained un-reversed, accepted as structured noise on the
      5/12 laterality-dependent findings for those studies rather than
      shrinking the weak set — spec's "full 4,349, no pilot" decision
      stands). `laterality_final`/`source` must be persisted per study
      (Cell 9/10) so Notebook B can break down per-finding AUC by
      known-vs-unknown laterality later if needed. Port
      `load_series_slices`-equivalent logic: order by `SliceLocation`,
      apply the sagittal laterality list-order reversal keyed on
      `Laterality` (Fase 4 audit finding (1) — reverse slice **order**,
      not pixel flip, for sagittal).

- [x] **Cell 6 — `normalize_physical_scale` + `center_crop_or_pad`.**
      Port from Fase 4: resize to `TARGET_MM_PER_PIXEL=0.35` using
      `PixelSpacing`, then crop/pad to `CROP_MM=130.0` (371x371px).

- [x] **Cell 7 — Intensity normalization.** Port `apply_voi_lut`-based
      normalization with the percentile-fallback path (Fase 4 audit fix
      (3)), rescale to [0,1].

- [x] **Cell 8 — 2.5D triplet construction.** Port `mm_to_slice_gap` +
      `build_25d_triplet` (physical-mm gap, `GAP_MM=4.0`, per Fase 4
      §4.2's revised mm-based logic, not raw slice-index gap).

- [x] **Cell 9 — Per-study preprocessing function with failure
      handling.** Wrap Cells 5-8 into one function taking a
      `StudyInstanceUID`, returning either the `(3, 371, 371)` float32
      triplet or raising — caller (Cell 10) catches and logs
      `StudyInstanceUID` + exception message rather than aborting the
      whole run (spec §3.2 requirement). Smoke-tested end-to-end on the
      sample study: `(3, 371, 371)` float32, range `[0.0, 0.9845]`.

- [x] **Cell 10 — Run the full loop.** For all 4,407 studies: preprocess,
      save `.npy` to `/kaggle/working/triplets/{StudyInstanceUID}.npy`.
      Collect failures into a list; print the failure count and the first
      ~20 failing `StudyInstanceUID`s + error messages at the end (not
      mid-run interruptions). **Result: 4,407/4,407 processed, 0 errors,
      ~1,808s (~30 min).** `preprocessing_metadata.csv` saved alongside
      (`laterality_final`/`laterality_source` per study, from Cell 5's
      resolution).

- [x] **Cell 11 — Sanity checks.** File count in `/kaggle/working/triplets/`
      equals `4407 - len(failures)`; spot-check shapes are uniformly
      `(3, 371, 371)`; visually plot 2-3 triplets (mix of gold and weak,
      at least one from a study flagged positive by `train.csv` or the
      labeler) the same way Fase 4's post-result diagnostic did, to catch
      a gross pipeline break before spending Kaggle Dataset publish time
      on broken output. **Confirmed:** 4,407/4,407 files, uniform
      `(3, 371, 371)` shape on the 200-file sample; both plotted studies
      (1 gold-positive, `laterality=R`; 1 weak, `laterality=unknown`)
      show a well-centered sagittal knee joint, good contrast, no
      cropping/corruption artifacts.

- [x] **Cell 12 — Publish as a Kaggle Dataset.** The in-Kaggle "Notebook
      Output Files" attach path (`Save & Run All` -> Output page -> "New
      Dataset") didn't surface in the UI; resolved via manual
      download/re-upload instead: zipped `triplets/` in-kernel
      (`shutil.make_archive`) after a `Quick Save` (uses the current
      interactive session's state, no 30-min full re-run), downloaded
      via `kaggle kernels output` (user-run, not Claude — per this
      project's no-Kaggle-CLI-download rule), re-uploaded as a new
      private Dataset named **`triplets_knee`** on kaggle.com/datasets.
      This is the dataset slug Notebook B mounts (Cell 1).

---

## Notebook B — `05b_gold_weak_training.ipynb`

Mounts Notebook A's dataset, builds labels/folds, runs the 2-arm x
3-seed x 5-fold evaluation, reports the gate result.

- [x] **Cell 1 — Imports + mount.** `torch`, `timm`, `sklearn.model_selection.GroupKFold`,
      `sklearn.metrics.roc_auc_score`. Mount Notebook A's published
      dataset as input; mount the competition dataset for `train.csv`
      (labels/report text — already available in the same Kaggle
      environment, per Fase 1's `ON_KAGGLE` path). **Actual mount path
      (discovered, not assumed):** `/kaggle/input/datasets/alherma7/triplets-knee/`
      — datasets nest under `datasets/<owner>/<slug>/` the same way
      competitions nest under `competitions/<slug>/` (Fase 1's finding),
      confirmed by listing rather than guessed. `triplets_knee`'s
      Kaggle-assigned slug is `triplets-knee` (underscore -> hyphen).
      Contents confirmed flat (no wrapper folder, matches
      `shutil.make_archive`'s default): 4,407/4,407 `.npy` files directly
      under the dataset root. **Gap found:** `preprocessing_metadata.csv`
      never made it into the zip (it was written to `/kaggle/working/`,
      one level above `OUTPUT_DIR`, which is all `shutil.make_archive`
      packaged) — not blocking, since the laterality fix is already
      baked into the saved pixel data; only needed for the optional
      Cell 10 known-vs-unknown breakdown, deferred until/unless that
      cell is reached (recompute cheaply from the competition DICOM
      headers in Notebook B rather than re-uploading).

- [x] **Cell 2 — Port the label/fold table logic.** Copy the bodies of
      `src/labelers.py::label_reports`/`report_group_key` and
      `src/data.py::load_training_labels` into the notebook (self-contained,
      per Global Constraints). Build the table, then sanity-check against
      the numbers already verified when this logic was graduated
      (`notebooks/04b_gold_weak_groupkfold.ipynb`, 2026-08-18): 4,407
      rows total, 58 `is_gold=True`, 54 duplicate report-template groups /
      206 studies with 0 groups split across folds, gold distributed
      16/11/14/8/9 across the 5 folds. **Result:** all invariants matched
      exactly (4,407 / 58 / 54 groups / 206 studies / 0 groups split
      across folds) except the exact gold-per-fold counts, which came out
      13/14/8/11/12 instead of 16/11/14/8/9 — same total, no degenerate
      fold, still 0 leakage; most likely a `scikit-learn` version
      difference between this Kaggle environment and wherever `04b` ran
      (`GroupKFold`'s internal greedy balancing is deterministic given
      identical input, so a differing-but-valid partition points to an
      environment difference, not a logic bug) — not investigated further
      since it doesn't affect any property the experiment actually
      depends on.

- [x] **Cell 3 — Dataset/DataLoader.** Wraps Notebook A's `.npy` triplets
      + the Cell 2 label table into a PyTorch `Dataset` (returns
      `(triplet_tensor, target_vector, is_gold, StudyInstanceUID)` per
      item) and a `DataLoader`. Studies present in the label table but
      missing a triplet (Notebook A Cell 10 failures) are excluded here,
      logged with a count. **Result:** perfect 4,407/4,407 intersection
      (0 missing either direction — consistent with Notebook A's 0
      preprocessing failures). Smoke test: `(3, 371, 371)` float32 input,
      `(12,)` float32 target with the expected `{0.0, 0.5, 1.0}` values.

- [x] **Cell 4 — Model.** Port the Fase 4 EfficientNet-B0 model
      definition (`timm`, ImageNet-pretrained, 12-logit linear head) and
      the differential-LR parameter-group split (backbone vs. head) from
      `notebooks/04_baseline_cnn.ipynb`. **Confirmed:** logits shape
      `(1, 12)`, backbone 4,007,548 params (lr=1e-5), head 15,372 params
      (lr=1e-3, matches `1280*12+12` for EfficientNet-B0's 1280 features).

- [ ] **Cell 5 — Loss with corrected `pos_weight`.** `BCEWithLogitsLoss`
      taking soft targets directly; `pos_weight` computed per finding
      **per training fold** as `n_negative/n_positive` over hard-labeled
      rows only (`target != 0.5`) — per Global Constraints, this is the
      one place the spec's first draft had a bug, so write a quick manual
      check in this cell (e.g. print `pos_weight` for `oa_lateral_compartment`
      and confirm it's not collapsing toward 1.0 the way the buggy
      all-cells version would).

- [x] **Cell 6 — Single (arm, seed, fold) training function.** **Note:**
      `batch_size` not specified by the spec's Global Constraints —
      `batch_size=8` from Fase 4 would mean ~440 steps/epoch on the
      gold+weak arm's ~4,400 training studies, impractically slow.
      Smoke-tested (batch_size=32 at the time) with `arm="gold_only"`,
      1 epoch: `val_ids=13` (matches this run's fold-0 gold count),
      `val_preds.shape=(13,12)`, `train_auc=0.5320`.
      **GPU memory probe (added after this cell, real Kaggle 2xT4):**
      peak memory scales linearly with batch size on a single T4
      (~0.253GB/unit: 32->8.09GB, 40->10.09GB, 48->12.07GB, 56->14.10GB,
      64->OOM even on a clean GPU — confirmed it's real activation
      memory at 371x371, not cached clutter from earlier cells).
      **Final choice: `batch_size=48`** (12.07GB / 15.36GB = 78.6%,
      ~3.3GB headroom for real-data DataLoader overhead vs. the random-
      tensor probe). Used for all of Cell 7's 30 runs. Given
      `arm` (`"gold_only"` or `"gold_weak"`), `seed`, `fold`: set
      `torch.manual_seed(seed)`, build the train split (`fold != k`,
      filtered to gold-only rows if `arm == "gold_only"`), train for the
      fixed 8-epoch budget (no val-based checkpoint selection — Fase 4
      audit fix (2)), then predict on that fold's gold-only rows for both
      the final-epoch train set and the val set. Returns pooled
      train-predictions and val-predictions for this fold.

- [~] **Cell 7 — Outer loop.** For each of the 2 arms x 3 seeds x 5
      folds (30 runs): call Cell 6, collect val predictions into a
      per-(arm, seed) pooled table (58 rows once all 5 folds are done)
      and train predictions similarly. This is the long-running cell —
      note in a markdown cell before it that Kaggle session time should
      be checked here (spec §8 open risk); split into multiple cells
      per-arm if a single session can't fit all 30 runs.
      **gold_only done (2 attempts):** first attempt used
      `batch_size=48` (same as planned for `gold_weak`) — produced
      severe underfitting (`train_auc` 0.62-0.66 vs. Fase 4's expected
      0.95-0.98), root-caused to too few gradient steps/epoch at that
      batch size against gold_only's tiny ~44-50-study folds (~1
      batch/epoch instead of Fase 4's ~5-6). Fixed by using
      `batch_size=8` (matches Fase 4) for `gold_only` specifically,
      keeping `batch_size=48` for `gold_weak` (large enough folds that
      it doesn't collapse step density) — re-run confirmed `train_auc`
      0.9809/0.9793/0.9793 across the 3 seeds, in the expected
      memorization regime. 110s total. Pooled OOF macro-AUC:
      **0.5286 (std 0.0113)** across the 3 seeds — closely matches
      Fase 4's historical 0.532 (std 0.021) despite the different CV
      protocol, a strong external-validity check on the reconstructed
      pipeline before committing to the expensive arm.
      **`gold_weak` (batch_size=48) attempted interactively, lost to a
      Kaggle inactivity timeout** at seed=43/fold=2/epoch=4 (~3,065s in)
      — the interactive session gets killed after browser/tab
      inactivity, independent of whether the kernel is still busy
      computing; `seed=42`'s completed result (`train_auc=0.7871`,
      notably lower than `gold_only`'s 0.98 — consistent with less
      overfitting on more data, the experiment's core hypothesis) was
      lost since nothing was persisted outside session memory.
      **Decision (2026-08-19): switch to `Save & Run All` (background
      commit) for the actual full run**, instead of continuing
      interactively — a committed run executes on Kaggle's
      infrastructure independent of browser activity, bound only by the
      session's GPU-hour quota (9-12h), not by inactivity. Next session:
      consolidate Cells 1-8 (all individually validated above, including
      both `batch_size` fixes) into the final notebook and commit it;
      expect a full re-run from scratch (~110s gold_only + ~1.8h
      gold_weak, based on measured per-seed timing).

- [ ] **Cell 8 — Compute metrics.** For each (arm, seed): `macro_roc_auc`
      and `per_finding_roc_auc` on the pooled val table (58 gold rows)
      and separately on the pooled train table. Aggregate mean+std of
      the val macro-AUC across the 3 seeds, per arm.

- [ ] **Cell 9 — Gate decision.** Print/tabulate: gold-only control arm's
      mean+std, gold+weak arm's mean+std, Fase 4's historical 0.532 (for
      context only). State explicitly whether gold+weak beat the
      control arm (the actual gate, per spec §5) — win or not, this is
      the number that goes back into `README.md`.

- [ ] **Cell 10 — Per-finding breakdown + train/val gap.** Table of
      per-finding AUC for both arms, plus the train-vs-val macro-AUC gap
      per arm (Global Constraints' diagnostic-logging requirement) —
      inspect whether specific findings (e.g. `oa_lateral_compartment`,
      weakest-labeled per Fase 3) drag the gold+weak arm down
      disproportionately (spec §8 open risk), and whether the train/val
      gap narrowed relative to Fase 4's 0.95-0.98 train-AUC regime.

---

## Closing task (after both notebooks run successfully)

- [ ] **Update `README.md` Progress and `RESOURCES.md` (if applicable)
      with the real result** — win or negative result, per this
      project's logging convention (`structuring-ml-projects` step 7).
      Include: gold-only control arm's number, gold+weak arm's number,
      per-finding breakdown highlights, and the train/val AUC gap
      finding. If it's a win, open the follow-up conversation about
      graduating the image pipeline to `src/` (spec §1, §5) — a separate
      brainstorming pass, not part of this plan.
