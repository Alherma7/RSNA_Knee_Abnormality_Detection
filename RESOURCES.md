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
