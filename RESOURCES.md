# Resources

One entry per source. `Why` connects the source to a specific decision
or function in this repo.

## Papers / frameworks

- **Data Programming: Creating Large Training Sets, Quickly** (Ratner
  et al., NeurIPS 2016) / Snorkel (snorkel.org, tutorial "Intro to
  labeling functions")
  Why: weak supervision — combines `src/labelers.py::label_report()`'s
  labeling functions into a probabilistic label for studies without an
  official one.
- **MONAI documentation** (docs.monai.io)
  Why: DICOM reading and medical-image transforms (`src/data.py`),
  avoids reinventing resampling/windowing.
- **pydicom**
  Why: raw DICOM header/metadata inspection ahead of any framework.
- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"**
  (JMLR 2012)
  Why: justifies random search over grid search for a multi-hyperparameter
  space (backbone + pooling), once that tuning stage is reached.
- **Bien et al., "Deep-learning-assisted diagnosis for knee magnetic
  resonance imaging: Development and retrospective validation of MRNet"**
  (PLOS Medicine, 2018) / stanfordmlgroup.github.io/projects/mrnet
  Why: the most direct academic precedent for this problem (classifying
  multiple knee findings from MRI with little data). Investigated
  2026-08-17 after Fase 4's weak result: MRNet aggregates ALL slices per
  series (max-pooling) across all 3 planes (axial+coronal+sagittal), not
  a single central slice of a single plane like our baseline — and
  reaches AUC 0.937 (abnormality)/0.965 (ACL)/0.847 (meniscus) with a
  simpler backbone (AlexNet) than ours (EfficientNet-B0). Confirmed
  Fase 4's limitation was view/information count per study, not backbone
  size — supported prioritizing multi-plane pooling over backbone swaps
  (later independently re-confirmed at this project's own scale, see the
  MIL experiment entry below, and again by discussion 735304's ablation
  in the 2026-08-25 reorientation).
- **Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research
  Dataset for Effective Transfer Learning"** (Radiology: Artificial
  Intelligence, 2022)
  Why: investigated 2026-08-17 alongside MRNet. Pretraining on 1.35M
  radiology images (CT/MRI/US) instead of ImageNet gives +4.5-4.8% AUC on
  small MRI datasets, including ACL/meniscus tear classification
  specifically — the gain concentrates exactly in this project's
  low-data regime. A candidate pretrained backbone (Tier B item B5 in
  the current plan), more grounded than simply enlarging the backbone
  (which a same-competition ablation later measured as a null — see
  discussion 735304 entry above).
- **Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks"** (ICML 2019)
  Why: the backbone chosen for the Fase 4 baseline
  (`src/model.py::build_backbone`, EfficientNet-B0). Verified 2026-08-17
  that neither Dive into Deep Learning nor Hands-On Machine Learning
  teaches EfficientNet in depth (D2L only mentions it in passing as a
  Neural Architecture Search example, citing this same paper; Hands-On
  ML doesn't mention it) — the EfficientNet-B0 choice itself cites the
  original paper, not those books.
- **Dive into Deep Learning, sections 14.1 (Image Augmentation) and 14.2
  (Fine-Tuning)** — verified against the book's actual text, 2026-08-17,
  not cited from memory
  Why: (1) 14.2 confirms fine-tuning the backbone instead of freezing it
  for small datasets (exactly the 58-gold-study case), and specifies the
  real technique — differential learning rate: a small rate for
  pretrained parameters, a larger rate for the new (randomly
  initialized) output layer — more precise than "fine-tune everything
  equally"; (2) 14.1 states literally "Flipping the image left and right
  usually does not change the category of the object" — the book itself
  assumes horizontal flip is safe only when the label doesn't depend on
  orientation, confirming why Fase 4 does NOT use horizontal flip as
  augmentation (`normalize_laterality` already fixes which side is
  medial/lateral for 5 of the 12 findings; a random horizontal flip would
  undo that normalization).
- **Hands-On Machine Learning with Scikit-Learn and TensorFlow, ch. 3
  ("Classification"), "Multilabel Classification" section** (p. 100) —
  verified against the actual text, 2026-08-17
  Why: confirms averaging a binary classification metric across several
  labels (macro-average) as a valid technique for a multilabel problem,
  naming "weighted" (by support) as the alternative — supports
  `src/evaluate.py::macro_roc_auc` (macro, unweighted, fixed by the
  competition's own rules) as a documented design choice, not an
  arbitrary one.
- **PyTorch docs, `torch.nn.BCEWithLogitsLoss`** (docs.pytorch.org,
  fetched 2026-08-17)
  Why: grounds `pos_weight` (Fase 4, later dropped — see Tier B item B2
  in the current plan) for per-finding class imbalance (15.5%-60.3%
  prevalence). Official formula: `pos_weight = n_negative / n_positive`
  for the class; the docs illustrate the parameter with a multi-label
  binary scenario (`c` = number of classes, `c>1` for multi-label) —
  matches this project's exact problem shape (12 findings, one logit
  per finding). `p_c > 1` raises recall, `p_c < 1` raises precision.
- **Dive into Deep Learning, sections 3.6.3 (Cross-Validation), 5.5.3-
  5.5.4 (Early Stopping / Classical Regularization), and 5.6 (Dropout)**
  — verified 2026-08-17
  Why: (1) 3.6.3 confirms K-fold cross-validation as the standard
  solution precisely when "training data is scarce" and there isn't
  enough to hold out a separate validation set — the 58-gold-study case;
  (2) 5.5.4 notes something non-obvious: weight decay alone (L2
  regularization) is often insufficient in deep networks to stop them
  interpolating the dataset, and its benefits "may only make sense in
  combination with the early stopping criterion" — implying weight decay
  and early stopping should be used together, not weight decay alone;
  (3) 5.6 confirms dropout as a standard regularization technique, with
  0.5 as the book's own illustrative value (an MLP example, not a CNN —
  only a reasonable starting point, not a prescriptive value for this
  problem).

- **Sechidis, Tsoumakas & Vlahavas, "On the Stratification of Multi-Label
  Data"** (ECML PKDD 2011) / `iterative-stratification` package
  (`MultilabelStratifiedKFold`) — verified 2026-08-17 (package
  description on PyPI confirms it implements this exact paper)
  Why: self-review finding on the Fase 4 spec — a plain random `KFold`
  over 58 studies doesn't guarantee a stable per-fold count of MCL's 9
  positives (the concern that motivated 3 folds over 5 in the first
  place), so the CV split needs to stratify across all 12 finding
  columns at once, not rely on fold count alone.

- **Kaggle "Dataset Description"** (competition Data tab, pasted by the
  user 2026-08-16)
  Why: confirms `src/config.py::FINDINGS` (12/12 match) and adds
  `OFFICIAL_LABEL_COLUMNS`; confirms the twelve label columns and
  `Report` live in `train.csv` itself (no separate `train_labels.csv`/
  `reports.csv`, contradicting what `src/data.py` assumed before this
  check); confirms `test_series.csv`/`test_series/` exist with the same
  schema as train (`src/data.py::load_series_metadata`,
  `load_dicom_series` now take a `split` argument); flagged DICOMs as
  mixed transfer syntax (JPEG Lossless, JPEG 2000, Implicit/Explicit VR
  Little Endian) stripped to an allowlist of 86 metadata tags, and
  hedged that `Fluid_Sensitive`/`Fat_Suppression` are "not necessarily
  equivalent for every case" — see the `notebooks/01_eda_dicom.ipynb`
  entry below for how these were actually checked against the data
  (the doc's hedge on Fluid_Sensitive/Fat_Suppression turned out overly
  cautious; the allowlist/transfer-syntax flags were confirmed real).
- **notebooks/01_eda_dicom.ipynb kernel run** (Kaggle, 2026-08-16, user-run
  against the mounted competition dataset)
  Why: ground truth for `src/data.py`/`src/features.py` docstrings,
  superseding both the reference notebooks and the official Dataset
  Description's hedged language where they conflict — this is a direct
  measurement on the actual data, not a description of it. Findings:
  (1) Fluid_Sensitive == Fat_Suppression on 100% of all 24,371
  train_series.csv rows (0 disagreements) — confirms pilkwang's
  original claim, contradicting the official doc's hedge; (2) all 4,407
  training studies have all 3 planes present; (3) InstanceNumber,
  ImagePositionPatient, SliceLocation, SliceThickness,
  SpacingBetweenSlices, and Laterality all survive the 86-tag allowlist
  (`src/features.py::normalize_laterality` can read `Laterality`
  directly instead of inferring it); (4) filename order vs.
  InstanceNumber rho ranged -0.24 to 0.69 across 10 sample series,
  confirming filename order is unreliable (matches both reference
  notebooks' rho ~ 0.01 claim); (5) a real bug caught mid-EDA: indexing
  `ImagePositionPatient[2]` assumes an axial-like z-varying axis, which
  is wrong for sagittal/coronal series (produced a constant column,
  `scipy` `ConstantInputWarning`) — use `SliceLocation` instead, the
  scalar DICOM already computes by projecting onto the slice normal;
  rho(InstanceNumber, SliceLocation) was exactly +-1.000 on all 10
  series re-run with the fix — same rank order, but the sign flips
  between series (5 of 10 were -1), so sort by one field
  (`SliceLocation`) consistently rather than mixing the two expecting a
  shared direction;
  (6) only one transfer syntax (Explicit VR Little Endian,
  `1.2.840.10008.1.2.1`) observed across 165 sample series, 0 pixel
  decode failures — the official doc's mixed-syntax warning wasn't hit
  in this sample, so don't assume it can't happen elsewhere in the
  ~0.5 TB corpus; keep `pylibjpeg`/`gdcm` available defensively when
  writing `load_dicom_series`; (7) the actual mounted path on a live
  Kaggle kernel is `/kaggle/input/competitions/<slug>/`, not
  `/kaggle/input/<slug>/` — `src/config.py::_KAGGLE_INPUT_DIR` and the
  notebook's `RAW_DIR` corrected accordingly; (8) train.csv's null
  pattern for the 12 label columns is strictly all-or-nothing across
  all 4,407 rows — exactly 58 rows have all 12 populated, 4,349 have
  none, zero rows partial — confirming `src/data.py::load_gold_labels`;
  `Report` is non-null on every row; (9) PixelSpacing (165 series, 30
  studies) ranges 0.137-0.703 mm/pixel (5.14x max/min), confirming
  `src/features.py::normalize_physical_scale` is needed — row and
  column spacing identical on every series (isotropic), and spacing is
  similar across planes (0.33-0.35 mm mean each), so the variation is
  per-study/protocol, not per-plane; SliceThickness 0.6-5.0 mm;
  SpacingBetweenSlices missing on 14/165 series (fall back to
  SliceThickness or consecutive SliceLocation deltas).
- **notebooks/00_export_local_gold_subset.ipynb kernel run** (Kaggle,
  2026-08-17, user-run) + **notebooks/02_eda_reports.ipynb local run**
  (2026-08-17, `nbconvert --execute` against the exported
  `data/raw/train.csv`/`train_series.csv`)
  Why: the export notebook operationalizes the 2026-08-16 infra decision
  (download only a small local subset, not the full ~0.5 TB) — copied
  all 4 metadata CSVs in full (text-only, small) plus measured the gold
  DICOM subset at 7.01 GB (well within the ~515 GB free; not copied this
  run since Fase 2 only needed the CSVs). With those CSVs local, the
  report EDA notebook ran fully offline, no Kaggle round-trip needed.
  Findings, all measured directly against the real 4,407-row train.csv
  (not assumed from the reference notebooks): (1) report length is
  comparable between gold (n=58, mean 1305 chars / 179 words) and weak
  (n=4349, mean 1095 chars / 147 words) — gold runs ~19% longer on
  average but is not a qualitatively different distribution, so
  validating Fase 3's `label_report()` against gold is not obviously
  biased by length; (2) hard-wrap (prvsiyan's claim that "a large share
  of the corpus" arrives wrapped at a fixed column) is real but a
  minority pattern here — only 23.2% of the 4,407 reports have >=1
  line-pair flagged as a wrap candidate (median `frac_wrap_candidates`
  across the corpus is 0.0) — a correction in the same spirit as the
  Fluid_Sensitive/Fat_Suppression finding in Fase 1: the reference
  notebook's qualitative framing doesn't match the measured frequency
  here; (3) a real flaw in the wrap-detection heuristic itself (adapted
  from prvsiyan's `unwrap()`), caught by inspecting a flagged example: a
  bulleted report (`> finding one`, `> finding two`) gets marked as
  wrap-candidate on every line, because the check tests whether the next
  line's first character is uppercase and a bullet symbol (`>`) is
  neither upper- nor lowercase — `str.isupper()` on a non-letter char is
  `False`, so every bullet line after a non-punctuation-ending line
  reads as "continuation" even though each `>` line is a complete,
  independent finding; fix before reusing this logic in Fase 3 by
  stripping leading bullet/quote markers before the capitalization
  check; (4) script distribution (Unicode-range detection, no external
  library, ran on all 4,407 reports): 87.7% Latin (3,866), 7.3% Greek
  (321), 5.0% Cyrillic (220) — `langdetect` isn't installed locally
  (not in `requirements.txt`) so there is no finer per-language split
  within the Latin bucket yet, but manual inspection of duplicate
  templates (finding 5 below) directly confirmed Turkish, English, and
  Spanish inside it; contrasting the full 9-language breakdown against
  `config.py::REPORT_LANGUAGE_COUNT=9` is still open, pending
  `langdetect` (or an equivalent) being available in whichever
  environment runs this next; (5) 54 duplicate-report groups (identical
  text after NFKD-normalizing case/diacritics), covering 206 of 4,407
  studies (4.7%) — the largest is a 37-times-repeated Turkish
  "normal knee" template — and critically, **1 template is shared
  between a gold study and a weak study**, so `src/labelers.py
  ::report_group_key()` (Fase 5) is confirmed necessary to protect gold
  from a validation leak too, not only to protect weak-vs-weak splits;
  (6) vocabulary: 15,492 distinct tokens over 646,582 total (Unicode-
  aware tokenizer covering Latin/Greek/Cyrillic ranges) — the top 40 by
  frequency already mixes English/Spanish anatomy terms (medial,
  ligament, meniscus, tear, cartilage, patellofemoral, collateral,
  edema, tendon) with a Greek token (`του`) and a Cyrillic token (`на`),
  confirming in this project's own data (not just asserted from the
  reference notebooks) that Fase 3's labeling functions need to test all
  nine languages per clause rather than routing by a guessed language.
- **notebooks/03_labeler_validation.ipynb kernel run** (Kaggle and local,
  both 2026-08-17, identical results) — graduated `src/labelers.py
  ::label_report()`/`label_reports()`
  Why: applies the Data Programming/Snorkel weak-supervision approach
  (cited above) and the reference notebooks' assertion/negation/
  multilingual-pooling pattern to build the actual labeling function, and
  measures it against this project's real metric before wiring it into
  `src/`, per structuring-ml-projects's graduation rule. Result: macro
  ROC-AUC 0.688 against the 58-row gold subset (vs. 0.5 for a constant
  baseline), all 12 findings individually above 0.5. Findings from the
  validation process itself (not just the final number): (1) a first,
  naive version — "pathology cue wins over negation when both match in a
  clause" — scored `effusion` at AUC 0.438, *below random*, because for
  entity-only findings (effusion, synovitis, baker's cyst, bone
  contusion) the anatomy cue and the pathology cue are the same term, so
  "no effusion" matches both and the naive rule voted every negated
  mention positive; fixed by giving negation priority whenever both cues
  match the same clause, which raised macro ROC-AUC from 0.629 to 0.677;
  (2) the three osteoarthritis-compartment findings had the weakest
  initial coverage (`oa_medial_compartment` silence rate 89.6%) — instead
  of guessing why, inspected real gold-positive reports directly and
  found the pathology is stated with synonyms the initial lexicon didn't
  cover ("cartilage fissuring" not "chondral fissuring", "spurring" not
  just "osteophyte", "subchondral cystic change") — widening the lexicon
  with these observed terms raised macro ROC-AUC to 0.688 and brought
  every finding above the 0.5 baseline (weakest remaining:
  `oa_lateral_compartment` at 0.553, silence rate ~90%, not further
  chased to avoid overfitting the lexicon to this same 58-row set); (3)
  this first pass has cue coverage in English and Spanish only (the two
  languages dominating the corpus per the vocabulary finding above) —
  Greek/Cyrillic reports (~12% of the corpus) always abstain (0.5)
  rather than receiving a wrong confident label, a known and accepted gap
  for now, not a silent one. `report_group_key()` was not part of this
  graduation — still `NotImplementedError`, scoped for Fase 5.
- **Negation-scoping fix in `src/labelers.py`** (local, 2026-08-18,
  validated against the same 58-row gold set as the original graduation)
  Why: the 2026-08-18 audit (see the Fase 4 audit entry above) found
  that `label_report()`'s clause-level negation check let a negation cue
  anywhere in a clause veto a pathology assertion anywhere else in that
  same clause — e.g. "tear of the medial meniscus without effusion" got
  a wrong confident 0.0 for the tear, because "without" and "medial
  meniscus" shared a clause. Measured impact before fixing: at least 53
  studies across 6 findings, a conservative lower bound over the full
  weak corpus (barely visible on the small, high-abstention gold gate,
  but a real training-label contaminant at the 4,349-study scale this
  fix is for). Fixed with a new `_negation_applies()` check: a negation
  cue only counts against `finding` if the text it plausibly governs
  (after a pre-nominal cue like "without"/"no"/"sin", or before a
  predicate cue like "... is normal/intact") itself names `finding`'s
  own anatomy or pathology term. An earlier version of the fix also
  split clauses on every comma to separate compound sentences like "The
  ACL is torn but the PCL is intact" — this worked for that pattern but
  broke a different, more common one in this same corpus: naming a
  structure once and continuing to describe it across a comma without
  repeating the name ("menisco medial ... conservada, sin signos de
  rotura"; "no effusion, synovitis, or bone contusion identified" — a
  shared negation over a coordinated list). Measured directly: comma-
  splitting dropped gold macro ROC-AUC from 0.688 to 0.674, a real
  regression, reverted in favor of splitting only on the unambiguous
  `but`/`pero`/`aunque`/`although` boundary plus the local-argument
  check. Final result: gold macro ROC-AUC 0.686 (vs. 0.688 originally —
  negligible difference; 10 of 11 changed cells went from a correct
  negative to an abstention, not to a wrong answer, and 1 was a genuine
  fix). One case is a documented, accepted gap rather than a silent one:
  two fully independent clauses joined only by a bare comma (no
  conjunction, no repeated anatomy word) aren't reliably scoped apart,
  since fixing that specific pattern would reintroduce the coordinated-
  list regression. 6 new tests in `tests/test_labelers.py` (16 total),
  including one that documents that accepted gap explicitly rather than
  asserting a result the fix doesn't actually guarantee.
- **`report_group_key()` implementation and validation** (local,
  2026-08-18, real data)
  Why: implements the function this project had cited as a Fase 5
  prerequisite since Fase 3 (was `NotImplementedError`). Hashes a
  normalization that collapses ALL whitespace (spaces/tabs/newlines
  alike), not `_normalize()` (which deliberately keeps newlines for
  `_clauses()`'s sentence splitting) — reusing `_normalize()` was tried
  first and measured to under-count real duplicate templates that only
  differ by line-wrap position: 50 groups/192 studies instead of the
  real 54/206 against `data/raw/train.csv`. The corrected version
  reproduces notebooks/02_eda_reports.ipynb section E exactly: 54
  duplicate groups, 206 studies, 1 template shared between a gold and a
  weak study. 4 new tests in `tests/test_labelers.py`.
- **scikit-learn, `sklearn.model_selection.GroupKFold`** (scikit-learn
  docs) — adopted 2026-08-18 for the Fase 5 combined-dataset split
  Why: plain group-only splitting (not `MultilabelStratifiedKFold`,
  cited above for the Fase 4 58-gold CV) grouped by
  `report_group_key()`, so a report template never spans train/val —
  confirmed necessary in notebooks/02_eda_reports.ipynb section E (1
  template shared between a gold and a weak study). No stratification is
  attempted at this scale: the small-sample class-imbalance risk that
  motivated `MultilabelStratifiedKFold` for 58 gold studies (as few as 9
  MCL positives) is far less pressing across 4,407 studies, and
  stratifying a mix of hard 0/1 gold labels and graded 0.5-abstention
  weak labels isn't well-defined without picking an arbitrary threshold
  first. Validated in notebooks/04b_gold_weak_groupkfold.ipynb against
  the real train.csv: `GroupKFold(n_splits=CV_FOLDS)` (`CV_FOLDS=5`,
  `src/config.py`, unused since it was added until now) over all 4,407
  studies gives 0 report-template groups split across folds and a
  reasonable (not perfectly even, since ungrouped) gold-per-fold spread
  of 16/11/14/8/9.
- **`src.data.load_training_labels()` graduation** (local, 2026-08-18,
  validated in notebooks/04b_gold_weak_groupkfold.ipynb against the real
  train.csv)
  Why: builds the one-row-per-study training table this project needs
  for Fase 8's `src/train.py::run()` — official 0/1 labels for the 58
  gold studies (never re-derived from the labeler), graded weak labels
  from `label_reports()` for the other 4,349, plus the `GroupKFold` fold
  assignment above. Graduating this also finally implemented
  `load_reports()`/`load_gold_labels()`, both `NotImplementedError`
  since the project skeleton. Verified against real data: 4,407 rows out
  of 4,407 studies, 0 discrepancies between the gold rows here and
  `train.csv`'s own official columns. 4 new tests in
  `tests/test_data.py`, using small synthetic CSVs (not the real
  train.csv) so they stay hermetic — the real-data numbers are already
  proven in the notebook and don't need re-proving in a unit test.
- **notebooks/04_baseline_cnn.ipynb kernel run** (Kaggle, 2026-08-17,
  built and run cell-by-cell — the notebook itself was retired in the
  2026-08-25 cleanup, see git history before that commit for the file)
  Why: first real CNN baseline on the 58 gold studies. Applies Tan & Le
  2019 (EfficientNet-B0), D2L §14.2 (differential-LR fine-tuning) and
  §5.5-5.6 (dropout/weight-decay/early-stopping technique), PyTorch docs
  (`pos_weight`), and Sechidis et al. 2011
  (`MultilabelStratifiedKFold`) — all already cited above — to an actual
  training run rather than just a design. Findings from running it, not
  from the design alone: (1) inter-slice spacing varies 13.75x across
  the 58 selected sagittal series (0.4-5.5mm), discovered while
  investigating a legitimate 320-slice high-resolution outlier series
  (`SAG 3D_VIEW_PD_SPAIR_HR L`) — the spec's original slice-index `gap`
  was revised to a physical-mm `gap` for this reason; (2) a fixed
  physical crop (`CROP_MM=130.0`, pilkwang's measured value) turned out
  to be necessary in addition to `normalize_physical_scale`, since that
  alone still leaves a different pixel shape per study; (3) the first
  un-augmented, unregularized run showed real overfitting (train macro
  AUC 0.698 vs. val 0.492 on one fold) — confirming augmentation should
  be introduced in response to evidence, not assumed upfront, per the
  user's explicit call; (4) a `weight_decay` sweep (0, 1e-4, 1e-3, 1e-2)
  on that fold found `1e-2` best (val AUC 0.588); (5) augmentation
  (small rotation, intensity/contrast jitter, small translation — no
  horizontal flip, since `normalize_laterality` already canonicalizes 5
  of the 12 findings' medial/lateral axis) was then tried at multiple
  epoch budgets and combined/isolated from `weight_decay`, and never
  beat `weight_decay=1e-2` alone — logged as a negative result on this
  dataset size, not a permanent conclusion about the technique; (6)
  final 3-fold CV with the winning config (`weight_decay=1e-2`, no
  augmentation): macro ROC-AUC 0.595 / 0.515 / 0.612 per fold, mean
  0.574 (std 0.052), beating the 0.5 baseline — but every fold's train
  macro AUC reaches 0.95-0.99 (fold 2: 0.999), so this is a real but
  fragile generalization signal from a near-memorized training set, not
  evidence of a robust model at this sample size. **Superseded
  2026-08-18** — see the audit entry below; this 0.574 included ~0.042
  of metric-selection optimism plus two real preprocessing bugs.
- **Opus general-purpose audit of notebooks 00-05 + `src/` + `tests/`**
  (2026-08-18, read-only, dispatched before graduating anything from
  Fase 4)
  Why: found 4 P0 bugs in the Fase 4 CNN pipeline, independent of the
  "single slice, single plane" architectural limitation already
  identified via MRNet (above). (1) `normalize_laterality` flipped the
  wrong axis for sagittal images (`np.fliplr` mirrors the in-plane
  anterior/posterior axis; medial/lateral is the out-of-plane axis for
  a sagittal series) — actively corrupting orientation consistency
  rather than fixing it. (2) the reported 0.574 was `max(val AUC over
  epochs)` on the same fold used for early stopping — measured
  optimism +0.042 from the weight_decay sweep's own per-epoch traces,
  comparable to the entire margin over the 0.5 baseline. (3) no
  intensity normalization — raw DICOM `uint16` fed directly into an
  ImageNet-pretrained backbone. (4) the augmentation experiment was
  invalidated by (3): `ColorJitter` on unnormalized values in the
  hundreds saturates via torchvision's internal float clamp
  (`bound=1.0`), so the "augmentation doesn't help" conclusion had no
  valid evidence behind it.
- **Fix verification runs** (Kaggle, 2026-08-18, real data, cell-by-cell
  per the project's build convention) — closed out the audit above
  Why: (1) confirmed the laterality bug and derived the correct fix by
  reading `ImageOrientationPatient`/`ImagePositionPatient` on 8 real
  gold studies (mixed L/R): scanner orientation is identical between L
  and R, but ascending `SliceLocation` means lateral→medial for a right
  knee and medial→lateral for a left knee — opposite directions — so
  the fix reverses the **slice list order** per `Laterality`, not
  pixels; (2) `pydicom.pixels.apply_voi_lut` applied for intensity
  normalization (`WindowCenter`/`WindowWidth` confirmed present on the
  sample study), percentile 0.5-99.5 fallback if absent, rescaled to
  [0,1]; reprocessed all 58 gold studies with both fixes: 0 errors,
  uniform (3, 371, 371) shape; (3) retrained with a fixed 8-epoch
  budget, no val-based checkpoint selection, last-epoch score reported,
  `torch.manual_seed(42)` — macro ROC-AUC 3-fold **0.532 (std 0.021)**,
  still beats 0.5 but by a thinner, more credible margin; train AUC
  still reaches 0.95-0.98 per fold even with intensity normalized,
  confirming the ceiling is architectural, not these bugs; (4)
  augmentation retried on the fixed pipeline (plus fixing that
  `RandomAffine`/`ColorJitter` on a whole batched tensor draws one
  random transform for all 8 images instead of one per image): macro
  ROC-AUC 0.516 (std 0.035) — still does not beat no-augmentation,
  this time a trustworthy negative result. Fase 4 closed at 0.532;
  nothing graduated to `src/` yet.
- **MIL multi-plane pooling experiment** (Kaggle, 2026-08-18, real data,
  cell-by-cell) — tested whether MRNet's/the breast-cancer writeup's
  multi-view finding (both cited above) transfers to this project's 58
  gold studies before committing to it for Fase 6
  Why: found a third laterality axis the Fase 4 fix didn't cover — in
  coronal/axial (unlike sagittal), medial/lateral *is* an in-plane axis
  (image column = patient x, verified via `ImageOrientationPatient` on
  16 real studies, 8 per plane, mixed L/R), so the correct fix there is
  a pixel flip (`np.fliplr` gated on `Laterality == "L"`), the opposite
  of the sagittal fix. Built a MIL model (1 triplet/plane x 3 planes,
  shared backbone, max-pool over plane features before the head) plus a
  manually-implemented EMA (decay=0.9, deliberately low — vision
  defaults of 0.999+ assume thousands of training steps; this dataset
  gives only ~40 total, so a high decay would leave the shadow weights
  almost frozen at initialization) and evaluated over 3 seeds x 3 folds
  instead of one run, given the Fase 4 correction already showed
  epoch-to-epoch noise of the same order as the effects being measured.
  Result: no clear win — macro ROC-AUC 0.530 (std 0.048, no EMA) vs.
  0.532 for the single-plane baseline (statistically indistinguishable);
  0.549 (std 0.041) with EMA, a plausible but barely-confirmable small
  gain from 9 runs. More informative than the mean: fold 1 scored
  systematically higher than folds 0/2 across all 3 seeds (0.59-0.63 vs.
  0.48-0.55) — at n=58 with 3 folds, which studies land in which fold
  dominates over any architecture change tested. Train AUC still reached
  0.94-0.96 with MIL, same as single-plane — tripling the views per
  study didn't reduce overfitting or raise val AUC, so the bottleneck is
  training-study count (38-40/fold), not view count per study. MRNet
  combines multi-plane pooling *and* every slice in each series *and*
  (most likely) many more training studies than this project's 58 gold
  — replicating only the architectural piece without more data has
  limited return. Own-data confirmation (not just the cited papers) that
  Fase 5 (adding the 4,349 weak-labeled studies) is higher-value than
  further architecture tuning against the same 58 gold studies. Lives
  only in `notebooks/04_baseline_cnn.ipynb` Kaggle cell outputs
  (2026-08-18); nothing graduated to `src/model.py`.
- **pydicom, `pydicom.pixels.apply_voi_lut`** (docs.pydicom.org) —
  adopted 2026-08-18 while fixing finding (3) above
  Why: implements the DICOM standard VOI LUT windowing (PS3.3
  C.11.2.1.2.1) directly from a dataset's `WindowCenter`/`WindowWidth`,
  used instead of hand-rolling the windowing math. Looked at
  `dangnh0611/kaggle_rsna_breast_cancer`'s own `src/utils/windowing.py`
  (1st-place RSNA breast cancer solution, github.com/dangnh0611/
  kaggle_rsna_breast_cancer — shallow-cloned locally only to review,
  then removed: 62MB with vendored YOLOX/timm/albumentations and its
  own nested `.git`, too heavy to keep versioned here unlike the two
  small knee reference notebooks) first — it reimplements the same
  PS3.3 formula by hand, which confirmed VOI LUT windowing as the
  standard approach for DICOM intensity normalization, but pydicom's
  built-in function is used in this repo's code instead of copying
  that implementation. That repo's own laterality handling was also
  checked (`grep -i flip` across its `src/utils/`) and found to do no
  pixel-level laterality normalization at all — no direct precedent
  there for this project's sagittal medial/lateral bug (finding 1
  above). The author's own writeup (Kaggle discussion 392449, "1st
  place solution", pasted by the user 2026-08-18) has more transferable
  lessons than the raw code at our scale — see the entry below.
- **dangnh0611, "1st place solution" writeup** (RSNA Screening
  Mammography Breast Cancer Detection, Kaggle discussion 392449,
  pasted in full by the user 2026-08-18) — read for technique ideas,
  not code (the underlying pipeline: YOLOX ROI detection, 4x
  Convnext-small at 2048x1024, 5 external datasets, multi-day
  multi-GPU training, is far beyond this project's scale of 58 gold
  studies on a single Kaggle GPU)
  Why: (1) section 3.6/4.4's OOF table is a second, independent,
  empirical confirmation (after MRNet, cited above) that aggregating
  multiple views per subject matters more than backbone choice: single
  cropped-image AUC 0.873-0.896 vs. groupby-mean/max AUC 0.892-0.943 on
  the same folds — same direction as MRNet's finding, different
  competition and modality, reinforces prioritizing multi-plane pooling
  in Fase 6 over more Fase 4 tuning; (2) "soft positive label" (using
  0.8-0.9 instead of a hard 1.0 for a positive whose evidence is
  per-study rather than per-image, i.e. some images of a positive
  breast don't show the cancer) is the same asymmetric-confidence
  problem this project already has in `src/labelers.py::label_report()`
  (weak labels are graded floats near 0.5 when the labeling function
  abstains, not forced to a hard 0/1) — the writeup is independent
  validation that this graded-label design is the right shape for
  weak/uncertain supervision, not just this project's own choice;
  (3) their EfficientNet-without-EMA experiments show the same failure
  mode seen in this project's Fase 4 (`overfits quickly... AUC drops
  with longer training` under a high positive ratio) — Convnext with
  drop_rate=0.5 + EMA + a *lower* positive upsampling ratio trained
  more stably; worth trying in a future Fase 4/6 revision if the
  architecture-limited baseline is revisited (model EMA is not
  currently used anywhere in `src/model.py`); (4) section 3.1 tracks
  multiple metrics because "the competition metric is not stable and
  hard to track" at their scale — the same instability this project
  measured directly in the Fase 4 audit (2026-08-18, epoch-to-epoch val
  AUC noise of the same order as the effect being measured), independent
  confirmation that a single point-estimate AUC is not enough evidence
  on a small validation fold; (5) confirms `apply_voi_lut(prefer_lut =
  False)` (this project's Fase 4 fix, above) as a real technique used on
  one of their external datasets (VinDr-Mammo), not something invented
  for this project.

## Comparable projects

- **pilkwang/rsna-knee-baseline-v1** (Kaggle, downloaded to
  `data/raw/_reference_kernels/rsna-knee-baseline-v1.ipynb`)
  Why: defines the exact metric (unweighted mean of 12 AUCs), the list
  of the 12 findings (`src/config.py::FINDINGS`), the physical
  slice-order gotcha (rho ~0.01 if sorted by filename —
  `src/data.py::load_dicom_series`), and that
  `Fluid_Sensitive`/`Fat_Suppression` in `train_series.csv` agree on
  every row (not two independent axes in this dataset).
- **prvsiyan/rsna-knee-read-the-report-then-the-knee** (Kaggle,
  downloaded to
  `data/raw/_reference_kernels/rsna-knee-read-the-report-then-the-knee.ipynb`)
  Why: independently confirms the metric and the 12 findings; grounds
  per-finding attention pooling over plane/sequence "slots"
  (`src/model.py`) instead of mean pooling; grounds fine-tuning the
  encoder instead of freezing it; documents the "shared reports" leak
  (identical template reports across studies) that requires grouping
  folds by report-text hash (`src/labelers.py::report_group_key`); uses
  rank-blend instead of averaging raw probabilities for the final
  ensemble (`src/evaluate.py::rank_blend`).
- **yashbishnoi98/rsna-knee-infer-v1** (Kaggle, cited inside
  prvsiyan/rsna-knee-read-the-report-then-the-knee)
  Why: a high-score reference (~0.903) training on all report-only
  studies (not just the 58 gold) — supported the original case for
  using weak labels at scale, not just gold.
- **blacklions/report-teacher-anatomy-aware-hierarchical-multimod**
  (Kaggle, cited inside
  prvsiyan/rsna-knee-read-the-report-then-the-knee)
  Why: example of a per-finding specialist (Synovitis) with an
  RTA-HMIL ensemble — a reference if one specific finding lags behind
  the rest after the A2 baseline.
- **"Reference A" — `rsna-knee-data-structure-eda-baseline.ipynb`**
  (external notebook found by the user, 2026-08-24, downloaded to
  `data/raw/_reference_kernels/rsna-knee-data-structure-eda-baseline.ipynb`)
  Why: measured OOF 0.7675 with a graded 9-language assertion/negation/
  severity lexicon (0.814 agreement vs. our gold, vs. our own 0.686) and
  a 6-slot (plane × fluid-sensitivity × fat-sat) per-finding attention
  architecture — the source design for A1a and A2. Its own full-pipeline
  score (0.7675) trails the field (0.89–0.95+), so its design choices are
  treated as one candidate among several, not a default — see the
  strategy artifact linked from README.md for the full comparison.
- **This competition's own discussion, post 735304 and ~15 related
  threads** (Kaggle, pasted by the user 2026-08-24/25) — the single
  highest-weight source behind the 2026-08-25 reorientation
  Why: real, same-competition, same-metric evidence that outranks every
  cross-competition source above. Established: real single-model scores
  up to 0.951 (some from "LLM-only labels", 0.926–0.929); the host's
  official per-finding severity thresholds (Dr. Jacob Kazam) — the
  structural reason no text-only labeler clears ~0.82–0.88; an official
  ruling (Po-Hao "Howard" Chen) that LLM-based report labeling is
  permitted; Oleksii Zhukov's DICOM-metadata probe measuring a 0.0534
  macro-AUC drop from random to scanner-grouped folds (motivated A0's
  scanner-fingerprint grouping fix); stevenleehans's measured findings —
  slice ordering only ~5% filename-reliable (A0b), per-label noise
  diagnostics (`src/evaluate.py::per_label_gate`), and a labeling
  deep-dive reaching 0.878–0.893 agreement vs. our own gold (A1a′, see
  below). Full detail and per-item sourcing: the strategy artifact
  linked from README.md.
- **Published LLM label set, found 2026-08-25** (Kaggle Dataset,
  downloaded locally by the user to `data/raw/_published_labels/`,
  exact author unconfirmed — numbers reproduce stevenleehans's
  forum-reported 0.878/0.8873 exactly, so likely their
  `RSNA Knee Report Labels` dataset or an identical fork of it)
  Why: scored 0.878–0.893 macro-AUC vs. our 58 gold studies
  (`notebooks/03v2_published_label_validation.ipynb`, A1a′) — a
  decisive win over our own regex labeler's 0.686, making A1a/A1b
  unnecessary. Adopted as the training label source going forward.
- **stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache** (Kaggle
  notebook, downloaded to
  `data/raw/_reference_kernels/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb`)
  Why: **A3's preprocessing source** — chosen 2026-08-26 to reuse rather
  than rebuild (see
  [[feedback-prefer-reuse-over-rebuild-preprocessing]]). Its `Output` is
  the attachable slot cache (`notebooks/04v2_slot_cache_integration.ipynb`):
  6 named slots recovered from raw DICOM headers (plane × fluid-sensitivity
  × fat-suppression, independently confirming
  `Fluid_Sensitive`/`Fat_Suppression` collapse to one bit in
  `train_series.csv`), `crop_mm=130` (not the historical 160, which sits
  at this corpus's median FOV and silently disabled the crop on ~61% of
  studies), 9 slices/slot as 3 physically-adjacent-slice anchors
  (`group=3, n_group=3`, `take_group` indexing — mirrored by
  `select_group` in the notebook above), and a narrowed sampling window
  (`RSNA_WINDOW=0.35,0.65`) built specifically to de-confound an earlier
  same-notebook finding that 3 centre slices beat 9 window-spread slices
  (`exp-016`, +0.0086 AUC) — the "3 vs 9" comparison, that source's own
  code comments note, is a real result but confounded on *which* window
  was used, not settled for this narrower-window build.
- **stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache** — reused again for A4
  Why: A4's live test-time decode (`notebooks/07v1_a2_submission_inference.ipynb`,
  `src/preprocess.py`) ports the same source's `annotate()`/`pick_slots()`/
  `read_slot()`/`normalise_laterality()` (cells cell-10/cell-12/cell-14),
  adapted for single-study synchronous use — the pre-built cache itself
  cannot cover the hidden test set in a Code Competition (built before
  the test set existed), so the decode step is re-run live at scoring
  time instead, validated against the pre-built train cache's pixels
  before being trusted (`notebooks/07v1_...` section 2).
- **timm's `vit_small_patch14_dinov2.lvd142m`** (Hugging Face model card,
  huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
  Why: A2's backbone (`notebooks/05v2_slot_attention_baseline.ipynb`) —
  confirmed 2026-08-26 this is the exact tagged identifier timm needs
  (the bare name without `.lvd142m` does not resolve); native pretrained
  resolution is 518×518, interpolated down to our cache's 224×224 via
  `img_size=224`.
- **This competition's live discussion, several threads found 2026-08-28**
  (Kaggle, read via browser — not downloaded, per
  [[feedback-no-kaggle-downloads]] — see
  [[reference-kaggle-forum-search-method]]): starkhushi's
  "Why everyone's Synovitis AUC is stuck around 0.6-0.7" (discussion/737566,
  weak-label agreement ~0.79 on Synovitis, independently matching our own
  0.7903, and a RadImageNet-pretrained encoder reaching 0.78 vs. 0.62-0.72
  for ImageNet-pretrained models); Beyonder's unanswered host question on
  Synovitis/Effusion grading criteria (discussion/737155); Berat Kirbiyik's
  "What we've ruled out at 0.79 OOF" (discussion/737597, controlled
  ablations on a near-identical slot-attention architecture: EMA/mixup/
  longer training within noise, asymmetric loss badly hurt, centre-adjacent
  slices beat spread slices, and the same large/discrete-vs-thin/small
  error pattern this project measured); Oscar Yáñez Feijóo's "What is the
  real bottleneck" thread (discussion/735826, community consensus on
  label quality/image pipeline over backbone capacity, and real reports of
  gold labels contradicting their own report text).
- **wguesdon/rsna-knee-dinov2-at-meniscus-resolution** (Kaggle notebook,
  read via browser 2026-08-28, not downloaded — public score 0.815, solo)
  Why: a rigorous Nyquist-sampling argument for DINOv2 ViT-S/14 (a `d`-mm
  feature needs pixel pitch `<= d/2` to survive resizing), verified both
  analytically and against the frozen backbone empirically. Applied to
  our own A3 cache (224px/130mm crop, 0.580 mm/px, ~8.1mm/DINOv2-patch-
  token): resolves the upper end of the 1-3mm meniscal-tear range but not
  the smallest tears. 336px measured as the resolution sweet spot for this
  backbone (448px tested, no further gain) — a candidate re-render setting
  for A3 if the meniscus-tear architecture path is chosen. The notebook's
  own end-to-end score (0.815) is below our A4 real LB (0.834), so this is
  a diagnostic mechanism to borrow, not an end-to-end recipe.
- **dreaddevelopment/knee-mri-twelve-findings-from-a-single-model**
  (Kaggle notebook, read via browser 2026-08-28, not downloaded — public
  score 0.924, single non-ensembled model, 0.9167 gold macro-AUC on our
  same 58 gold studies held out of its training)
  Why: real code (not just discussion prose) for two structural
  differences from our A2 v1: (1) 64 slices/study across 5 slots (vs. our
  18, 6 slots × 3 slices, centre anchor only) and slices sampled evenly
  across 6-94% of the stack rather than only from the centre; (2) direct,
  measured claim in the notebook's own words that centre-only slice
  selection "measurably cost accuracy" specifically on the collateral
  ligaments and lateral meniscus — the outer slices, not the centre ones,
  is where they sit. Directly implicates A3's `group_index=1` (centre-
  anchor-only) choice as a plausible real contributor to this project's
  own low `mcl_injury`/`lateral_meniscus_tear` pooled AUC, beyond the
  small-gold-sample noise already identified — see
  [[project-rsna-phase-status]] for the full reasoning. 336px images,
  140mm crop (close to our 130mm) — independently arrives at the same
  336px resolution sweet spot as the notebook above, via a different
  (CNN/CoAtNet, not ViT) architecture.
- **tonylica/rsna-knee-dino-radimagenet-rank-ensemble** (Kaggle notebook,
  read via browser 2026-08-28, not downloaded — public score 0.92,
  35-checkpoint community mega-ensemble)
  Why: confirms RadImageNet-pretrained heads (10 of 35 ensemble members,
  sharing one RadImageNet ResNet-50 encoder) are a real, still-current,
  actively-used technique among top scorers — corroborates the synovitis
  RadImageNet lead from starkhushi's discussion post above with real code,
  not just one person's anecdote. Heavily forked/community-built (not a
  clean from-scratch reference), and doesn't isolate RadImageNet's
  per-finding contribution on its own.

## Library books (Desktop/LIBROS/)

- **Real-World Machine Learning**, ch. 4 and 8 — classifier evaluation;
  applied NLP as a labeler baseline.
- **Building Machine Learning Systems**, ch. 6–7 — classic text feature
  engineering, a baseline before Snorkel.
- **Dive into Deep Learning**, ch. 8, 13, 14 — fine-tuning modern CNNs,
  efficient compute, practical computer vision.
- **Hands-On Machine Learning**, ch. 3, 13 — classification metrics
  (ROC), CNNs.
- **Advanced Machine Learning with Python**, ch. 8 — ensembles and
  model robustness.
- **AI Engineering**, ch. 8–9 — data-quality framework and inference
  optimization (adapted from LLMs to this batch pipeline).
- **Bishop, PRML**, ch. 1.5 — decision theory behind AUC.
