# A4 — submission pipeline (live DICOM decode + ensembled A2 v1 inference)

**Status:** approved by user 2026-08-27, ready for implementation plan.
**Plan item:** new (not in the original A0→Tier B priority order — added
2026-08-27 because no real leaderboard submission has happened since
Fase 5's calibration submission scored 0.596, while local CV has since
moved from 0.5711 to A2 v1's pooled 0.7512; a real LB number is the only
way to confirm that improvement generalizes to hidden data).
**Depends on:** A2 v1 (done, pooled 4-fold gold macro-AUC 0.7512, all 4
fold checkpoints in `models/`), A0b (`geometric_slice_order()` — informs
this design, but see §3: the notebook ports the source's `order_slices()`
rather than reusing ours), A3 (slot definitions/config, reused here —
but NOT the pre-built cache itself, see §1).

## 1. Why this needs new work, not just reuse

This competition is a **Code Competition**: the submitted notebook is
re-run by Kaggle against the real, hidden test set at scoring time —
`internet disabled`, `<=9h run-time`, `submission.csv` as the notebook's
`Output` (confirmed by the user from Fase 5's retired `06 - Submission
inference` notebook header, the one that actually scored the real LB
0.596). Locally visible `data/raw/test.csv`/`test_series.csv` are a
3-study stub; the real hidden studies only exist inside the scored
rerun.

A3 deliberately reused a pre-built pixel cache (stevenleehans's Kaggle
Dataset) instead of writing our own DICOM decode, because rebuilding a
decode pipeline for **training** data was avoidable extra risk (see
[[feedback-prefer-reuse-over-rebuild-preprocessing]]). That shortcut
does not exist for **inference**: the pre-built cache is a static
Dataset built before the competition's test set was ever hidden, so by
construction it cannot contain the studies Kaggle will actually score
against. Live DICOM decode inside the submission notebook, at rerun
time, is unavoidable — this spec designs that decode step, reusing as
much of the already-cited source logic as possible rather than
re-deriving it from scratch.

## 2. Architecture

```
test_series/ (real, hidden until scoring)
        |
        v
  decode_study_slots()   <- NEW, ported from the A3 cache source
        |                    (per-study, synchronous — no threading/
        |                     sharding, that infra is corpus-build-only)
        v
  (6, 3, 224, 224) uint8 + (6,) presence mask   <- same shape contract
        |                                          as SlotCacheDataset's
        |                                          select_group() output
        v
  SlotAttentionModel x 4 (fold 0-3 checkpoints)  <- pretrained=False,
        |                                            our own fine-tuned
        |                                            weights only
        v
  mean(sigmoid(logits)) across 4 folds
        |
        v
  submission.csv  (sample_submission.csv template, per-study fallback
                    to constant 0.5 on any decode/inference failure)
```

## 3. Live decode: `decode_study_slots()`

Ports, adapts to single-study synchronous use (dropping the source's
`ThreadPoolExecutor`/chunking/resume machinery, which exists only for
corpus-wide batch builds — not needed for a per-study inference call),
and validates against the pre-built train cache before trusting on
hidden data:

- **Header probe** — `probe()` + `HDR_TAGS` (cell `cell-13`): one
  `stop_before_pixels=True` read of each series' **middle** slice,
  merged onto that study's `*_series.csv` rows. This runs *first* and is
  not optional: the CSVs carry only `StudyInstanceUID`,
  `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`,
  `Anatomical_Plane` — none of the `SeriesDescription`/`SequenceName`/
  `ScanOptions`/`ScanningSequence`/`RepetitionTime`/`EchoTime` fields the
  slot classification needs, nor the `Laterality`/`ImageLaterality`/
  `StudyDescription`/`BodyPartExamined` fields the laterality resolution
  needs. Without it every series classifies as
  `fatsat=False`/`weight=UNK`/`fluid=False` and the 6-slot recovery
  collapses to at most 2 wrong slots. It also supplies `n_slices` (the
  slot tie-break) for free.
- **Slot recovery** — `annotate()` + `pick_slots()`
  (`data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`,
  cells `cell-10`/`cell-12`): recovers fat-suppression (`ScanOptions`
  token match + `SeriesDescription`/`SequenceName` regex, careful about
  GE's `SAT_GEMS` false-positive) and pulse-sequence weighting
  (`RepetitionTime`/`EchoTime`/`ScanningSequence` heuristic) from DICOM
  headers, then picks one series per named slot
  (`config.SLOT_NAMES`) per the same plane × fluid-sensitivity ×
  fat-suppression rule as the cache — including the T1 scarce-slot
  fallback (any non-fat-sat series in-plane if the exact match is
  missing).
- **Slice ordering** — ports the source's own `order_slices()` (project
  `ImagePositionPatient` onto the `ImageOrientationPatient` normal, with
  a `SliceLocation` → `InstanceNumber` → filename fallback chain). It
  deliberately does **not** reuse `src/data.py::geometric_slice_order()`
  (A0b), despite the family resemblance: the source's chain is
  **all-or-nothing per series** (geometry only if *every* file in the
  series has a position tag, otherwise the whole series falls through
  wholesale to the next method), while `src/data.py`'s version picks its
  fallback key **per file**, which in degenerate series mixes
  geometry/`SliceLocation`/`InstanceNumber` within one series and yields
  a different order. The pre-built A3 cache this decode is validated
  against was built with the all-or-nothing chain, so matching it
  exactly is what parity requires — the notebook's copy is intentionally
  the source's behaviour, not ours, and must not be "fixed" into
  matching `src/data.py`.
- **Pixel decode** — `read_slot()` (cell `cell-14`): `group=3`
  physically-adjacent slices around the centre anchor
  (`config.SLOT_CACHE_GROUP_SIZE`/`GROUP_INDEX=1`, matching A2 v1's
  fixed `group_index=1`), `RescaleSlope`/`RescaleIntercept` applied,
  MONOCHROME1 inversion, physical crop via `PixelSpacing` to
  `config.SLOT_CACHE_CROP_MM=130.0`mm (pad-if-short-FOV left off, same
  as the cache's default), 1st–99th percentile normalization, bilinear
  resize to `config.SLOT_CACHE_IMG_SIZE=224`, output uint8.
- **Laterality normalization** — `normalise_laterality()` (cell
  `cell-14`): the side is resolved **once per study** (the source's
  `resolve_laterality(g)` runs over a `StudyInstanceUID` groupby group,
  never per series — the `Laterality` tag is missing on roughly half the
  series, so per-series resolution would give one study a mixed flip
  convention across its own slots), from `Laterality` tag →
  `ImageLaterality` → a multilingual side-word search across
  `SeriesDescription` + `StudyDescription` + `BodyPartExamined` that
  returns *unknown* on contradiction; the geometric fallback stays OFF,
  matching the source's own default. That resolved side maps every knee
  onto a left-knee convention
  — right knees get a horizontal flip (coronal/axial) or reversed slice
  order (sagittal); unresolved laterality is left alone (a wrong flip is
  worse than no flip, the presence mask already tells the model how much
  to trust a slot). **This step did not exist anywhere in this project's
  own code before now** — A2/A3 never needed it because the pre-built
  cache already had it baked in. Skipping it here would silently train
  the head to expect one convention while inference feeds the other —
  the single highest-risk step in this spec, and exactly the kind of bug
  Fase 5's old (reconstructed-from-prose) laterality logic flagged as
  unvalidated. Porting the real source code instead of reconstructing it
  removes that specific risk class.

**Validation gate (must pass before trusting this on hidden data):** run
`decode_study_slots()` on a handful of real gold **train** studies
(DICOM available locally-on-Kaggle same as always) and diff the result
against those same studies' already-known-correct pixels in the
pre-built train cache (`load_slot_cache_shard()` + the shard's
`studies.csv` for the row lookup). Expect an exact or near-exact uint8
match — any material mismatch means the port diverged from the source
and must be fixed before it's trusted on real hidden test data, which
offers no such cross-check.

## 4. Model and ensembling

- `SlotAttentionModel(pretrained=False)` — loads each of the 4 fold
  checkpoints (`models/a2_v1_fold{0,1,2,3}_best.pt`) in turn, no
  ImageNet download (blocked anyway, see §6).
- Per study: `mean(sigmoid(logits))` across the 4 folds — same pattern
  as Fase 5's own 5-fold ensemble in the retired `06` submission
  notebook. All 4 folds already exist and are the real pooled-4-fold-CV
  checkpoints (0.7512 OOF) — ensembling them is close to free and is
  standard practice for k-fold-trained checkpoints.
- Checkpoints delivered to the submission notebook as **4 separate
  Kaggle Datasets, one per fold** (`CHECKPOINT_PATHS`, same convention
  `06v2` already used for folds 0-2 — the user's own checkpoint uploads
  are per-fold, not bundled). `FINDINGS`/`OFFICIAL_LABEL_COLUMNS`/
  `CV_FOLDS`/`GROUP_INDEX` are hardcoded directly in the notebook
  (matching `05v2`/`06v2`'s own convention) rather than read from a
  bundled `metadata.json` — revised 2026-08-27 after the original
  one-Dataset-plus-metadata.json design (mirroring the retired `06`
  notebook's `CHECKPOINTS_MOUNT/metadata.json` pattern) turned out not
  to match how the user had actually uploaded the checkpoints.

## 5. Submission assembly and error handling

Same defensive pattern as the retired `06` notebook (already proved
itself on a real scored run):
- Start from `sample_submission.csv`, fill every row with the constant
  `0.5` fallback first.
- Per study: try decode → inference; on **any** exception (decode
  failure, missing slots, model error), log the `StudyInstanceUID` +
  exception and leave that row at the 0.5 fallback — never abort the
  whole run over one bad study.
- Before writing: assert column order equals
  `["StudyInstanceUID"] + list(config.OFFICIAL_LABEL_COLUMNS.values())`,
  assert row count equals `len(test.csv)`, assert `StudyInstanceUID` is
  unique. Write to `/kaggle/working/submission.csv`.

## 6. Open risk, not blocking design: `timm` with no internet

Training notebooks `pip install -q timm` — that will fail in the scored
rerun (internet disabled). `pretrained=False` avoids needing to
*download weights* (we load our own fine-tuned `state_dict` regardless),
but the **package itself** still needs to be importable offline. Two
fallbacks if it isn't preinstalled on Kaggle's static image: (a) a
Kaggle "Utility Script"/wheel Dataset attached to the submission
notebook, installed via `pip install --no-index --find-links=...`, or
(b) vendoring just the `timm` classes actually used
(`vit_small_patch14_dinov2` construction) to drop the dependency
entirely. **Verify directly on Kaggle before the real submission run**
(try `import timm` in a notebook with internet toggled off) — this is
not resolvable from outside Kaggle, and does not change anything else in
this design, so it is not blocking the implementation plan.

## 7. New notebook and `src/` disposition

- New: `notebooks/07v1_a2_submission_inference.ipynb` — self-contained
  (no `import src`, same convention as `05v2`/`06v2`, since Kaggle
  cannot import this repo), covering §2–§5 end to end, plus the §3
  validation gate as its own early section (run before spending the
  ~9h budget on real hidden-test decode).
- `src/train.py::run()` stays `NotImplementedError` — Kaggle's
  self-contained-notebook constraint means a locally-runnable pipeline
  script was never going to be what actually executes the submission,
  the same reason A2 v1's real training lives in a notebook, not
  `src/train.py`. Update its docstring to say so explicitly (point at
  `notebooks/07v1_...` instead of implying `run()` itself is the
  missing piece), so a future reader doesn't mistake this for
  unfinished work.
- **Graduates to `src/` only if the §3 validation gate passes**: the
  slot-decode function (exact module — `src/data.py` alongside
  `geometric_slice_order()`, or a new `src/preprocess.py` — is an
  implementation-plan decision, not fixed here), with unit tests over
  its **pure, DICOM-free sub-pieces** on synthetic fixtures — the
  `annotate()` regex classification (fat-suppression token matching
  incl. the `SAT_GEMS` trap, T1/T2/PD weighting rules) and
  `normalise_laterality()`'s flip logic (coronal/axial flip vs. sagittal
  reversal, unresolved-laterality no-op) — the same split A0b/A3 already
  used between locally-testable pure logic and Kaggle-only real-DICOM
  validation. The full per-study decode itself is validated by the §3
  gate on Kaggle, not by a local unit test (no real DICOM fixtures
  available locally).

## 8. Testing strategy

**Unit tests** (hermetic, synthetic fixtures):
- `annotate()`'s fat-suppression/weighting classification on hand-built
  header rows, including the `SAT_GEMS` (spatial, not fat, saturation)
  and `t2_de3d_we_tra` (underscore-as-separator) traps the source's own
  docstring calls out.
- `normalise_laterality()`: `R` + coronal/axial → horizontal flip; `R` +
  sagittal → reversed slice order; `L`/unresolved → no-op, on small
  synthetic tensors.
- Submission-assembly guards: column-order/row-count/uniqueness asserts
  each fail on a deliberately broken synthetic `DataFrame`, and the
  per-study fallback-to-0.5 path is exercised on a synthetic failure.

**Kaggle validation** (`notebooks/07v1_...`, needs GPU + DICOM tree, not
runnable locally):
1. §3's validation gate: live-decode a handful of real gold train
   studies, diff against the pre-built cache's pixels for those same
   studies.
2. `timm`-offline check (§6) before committing to an approach for it.
3. End-to-end dry run against the 3-study `test.csv` stub (real DICOM,
   real model, real `submission.csv` — cheap, fast, exercises the whole
   pipeline before spending real GPU/decode time on the actual scored
   rerun).
4. Only after the user reviews this notebook's real output does it get
   submitted to the competition for real.

## 9. Open items carried forward, not blocking this spec

- `timm`-offline resolution (§6) — verify on Kaggle, pick (a) or (b).
- Real hidden test set size and exact rerun time budget are unknown
  beyond "<=9h" — the design does not assume a size; if decode time
  becomes a real constraint once the true count is known, that's a
  follow-up optimization (e.g. batching decode across studies), not a
  blocker to building this pipeline now.
- Ensembling all 4 folds is fixed here as the default — comparing
  against a single best-fold checkpoint, if ever wanted, is a cheap
  later experiment, not part of this spec.

## 10. Known accepted residual: ~13% of studies diverge from the cache on one slot

Found running the §3 gate for real on Kaggle, 2026-08-27, after fixing a
real systematic bug (percentile normalization computed over only the
centre anchor's 3 slices instead of the source's pooled 9-slice window —
found and fixed the same day; that bug was 100%-of-studies and is
confirmed resolved). On a full sweep of every gold study present in
cache shard 0 (15 studies), 13/15 (87%) now match the pre-built cache
bit-for-bit exactly. The remaining 2/15 diverge on one slot each
(max|diff| 167–249 on a 0–255 scale). A full per-slice DICOM inspection
of one case found no code defect: clean headers, no decode failures or
exceptions, consistent `RescaleSlope`/`RescaleIntercept`, and the
anchor/crop/percentile math verified correct by hand against the source
for that exact case. Both affected studies had `laterality=unknown`,
but that is not the differentiator — 4 of the 6 `unknown`-laterality
studies in the same 15-study sample matched exactly.

Most likely explanation: a decode-library or cache-build-environment
difference (e.g. a `pydicom`/dependency version difference between now
and whenever the cache was originally built) affecting a small,
unpredictable subset of studies — not something reachable from this
notebook's own code, which already reproduces the source's algorithm
exactly. **Accepted as a known limitation, user decision 2026-08-27**,
for this calibration submission (the goal is one real leaderboard number
to sanity-check local CV, not a pixel-perfect reproduction of every
study) — same category of decision as Fase 5's own accepted intensity-
residual limitation, see
[[feedback-match-debugging-effort-to-stakes]]. The §3 gate's pass
threshold was changed from requiring 100% match to `>=75%` (well below
the observed 87%, so it still catches a real regression, e.g. a return
to the pre-fix ~0%) rather than re-litigating this residual on every
run. Revisit only if the real leaderboard score comes back implausibly
bad.
