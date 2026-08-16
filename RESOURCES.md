# Resources

Una entrada por fuente. `Why` conecta la fuente con una decision o
funcion concreta de este repo.

## Papers / frameworks

- **Data Programming: Creating Large Training Sets, Quickly** (Ratner
  et al., NeurIPS 2016) / Snorkel (snorkel.org, tutorial "Intro to
  labeling functions")
  Why: weak supervision — combina las funciones de etiquetado de
  `src/labelers.py::label_report()` (Fase 3) en una etiqueta
  probabilistica, para los estudios sin etiqueta oficial.
- **MONAI documentation** (docs.monai.io)
  Why: lectura DICOM y transforms de imagen medica (Fase 1,
  `src/data.py`), evita reinventar resampling/windowing.
- **pydicom**
  Why: inspeccion de headers/metadata DICOM cruda antes de cualquier
  framework, para `notebooks/01_eda_dicom.ipynb`.
- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"**
  (JMLR 2012)
  Why: justifica random search sobre grid search en la Fase 6 (espacio
  multi-hiperparametro del backbone + pooling).

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

## Comparable projects

- **pilkwang/rsna-knee-baseline-v1** (Kaggle, descargado en
  `data/raw/_reference_kernels/rsna-knee-baseline-v1.ipynb`)
  Why: define la metrica exacta (media no ponderada de 12 AUC), la
  lista de los 12 hallazgos (`src/config.py::FINDINGS`), el gotcha de
  orden fisico de cortes (rho ~ 0.01 si se ordena por nombre de
  archivo — `src/data.py::load_dicom_series`), y que
  `Fluid_Sensitive`/`Fat_Suppression` en `train_series.csv` coinciden
  en todas las filas (no son dos ejes independientes en este dataset).
- **prvsiyan/rsna-knee-read-the-report-then-the-knee** (Kaggle,
  descargado en
  `data/raw/_reference_kernels/rsna-knee-read-the-report-then-the-knee.ipynb`)
  Why: confirma independientemente la metrica y los 12 hallazgos;
  fundamenta el pooling por atencion por hallazgo sobre "slots" de
  plano/secuencia (`src/model.py`) en vez de mean pooling; fundamenta
  fine-tunear el encoder en vez de congelarlo; documenta el leak de
  "shared reports" (informes plantilla identicos entre estudios) que
  exige agrupar los folds por hash del texto del informe
  (`src/labelers.py::report_group_key`, Fase 5); usa rank-blend en vez
  de promediar probabilidades para el ensemble final
  (`src/evaluate.py::rank_blend`).
- **yashbishnoi98/rsna-knee-infer-v1** (Kaggle, citado dentro de
  prvsiyan/rsna-knee-read-the-report-then-the-knee)
  Why: referencia de score alto (~0.903) entrenando con todos los
  estudios solo-reporte (no solo los 58 gold) — apoya la Fase 5
  (usar weak labels a escala, no solo gold).
- **blacklions/report-teacher-anatomy-aware-hierarchical-multimod**
  (Kaggle, citado dentro de
  prvsiyan/rsna-knee-read-the-report-then-the-knee)
  Why: ejemplo de especialista por hallazgo (Synovitis) con ensemble
  RTA-HMIL — referencia para la Fase 6 si un hallazgo concreto queda
  por debajo del resto tras el baseline.

## Libros de la biblioteca (Desktop/LIBROS/)

- **Real-World Machine Learning**, cap. 4 y 8 — evaluacion de
  clasificadores; NLP aplicado como baseline del labeler.
- **Building Machine Learning Systems**, cap. 6-7 — text feature
  engineering clasico, baseline previo a Snorkel.
- **Dive into Deep Learning**, cap. 8, 13, 14 — fine-tuning de CNNs
  modernas, computo eficiente, computer vision practico.
- **Hands-On Machine Learning**, cap. 3, 13 — metricas de
  clasificacion (ROC), CNNs.
- **Advanced Machine Learning with Python**, cap. 8 — ensembles y
  robustez del modelo.
- **AI Engineering**, cap. 8-9 — marco de calidad de datos y
  optimizacion de inferencia (adaptado desde LLMs a esta pipeline por
  lotes).
- **Bishop, PRML**, cap. 1.5 — teoria de decision detras de AUC.
