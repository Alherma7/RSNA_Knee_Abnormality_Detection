# RSNA Knee Abnormality Detection
<img width="559" height="280" alt="header" src="https://github.com/user-attachments/assets/1d5f97a3-e23f-4966-9797-75a20c2ff4fa" />

Detectar 12 hallazgos clinicos en RM de rodilla multiplanar, entrenando
con un subconjunto pequeno de estudios con etiqueta oficial ("gold") y
el resto solo con el informe radiologico en texto libre. Metrica:
**Macro ROC-AUC** (media no ponderada de las 12 AUC por hallazgo).

Ver `RESOURCES.md` para la fuente de cada tecnica usada, y el plan
completo (`plan.html` / artifact) para el detalle fase por fase.

Forma de trabajo de este proyecto: el codigo se valida en los
notebooks de `notebooks/` antes de pasar a `src/`. Ningun modulo de
`src/` se considera terminado hasta que su funcion tiene: (a) docstring
con fuente citada, (b) validacion contra la metrica de este proyecto,
(c) una mejora medida frente a un baseline, y (d) un test en `tests/`.

## Hecho clave que define el diseno

De 4,407 estudios de entrenamiento, solo un subconjunto pequeno (~58)
tiene etiqueta oficial; el resto solo tiene el informe. Ademas,
`train.csv` tiene columna `Report` pero `test.csv` no — el texto solo
existe en entrenamiento. Consecuencia directa: el texto sirve para
generar etiquetas de entrenamiento (weak supervision), nunca como
entrada del modelo en inferencia.

## Progress

- [x] Esqueleto de proyecto creado (`src/`, `tests/`, `notebooks/`,
      `data/`, `models/`, `outputs/`).
- [x] `src/evaluate.py::macro_roc_auc()` implementado y con tests
      pasando (`tests/test_evaluate.py`, 5/5 verde).
- [x] Dos notebooks de referencia descargados a
      `data/raw/_reference_kernels/` (pilkwang/rsna-knee-baseline-v1,
      prvsiyan/rsna-knee-read-the-report-then-the-knee) y revisados
      para fundamentar `RESOURCES.md` y los docstrings de `src/`.

## Next steps

- [ ] Fase 0 — descargar el dataset de la competicion a `data/raw/` y
      verificar `src/config.py::FINDINGS` contra el data dictionary
      oficial (la lista actual viene de los notebooks de referencia,
      no del archivo original).
- [ ] Fase 1 — `notebooks/01_eda_dicom.ipynb`: estructura de
      planos/series, orden fisico real de los cortes, espaciado de
      voxel.
- [ ] Fase 2 — `notebooks/02_eda_reports.ipynb`: vocabulario, idiomas,
      confirmar ausencia de `Report` en test.csv.
- [ ] Fase 3 — `notebooks/03_labeler_validation.ipynb`: implementar
      `src/labelers.py::label_report()`, validar contra el subset gold.
- [ ] Fase 4 — `notebooks/04_baseline_cnn.ipynb`: baseline 2.5D CNN
      entrenado solo con gold.
- [ ] Fase 5 — mezclar gold+weak, `GroupKFold` por hash del informe
      (`src/labelers.py::report_group_key`), no solo por estudio.
- [ ] Fase 6 — `notebooks/05_ensemble_calibration.ipynb`: ensembling
      multi-plano/backbone con `src/evaluate.py::rank_blend()`.
- [ ] Fase 7 — optimizacion de inferencia para el limite de 9h sin
      internet (pesos empaquetados como Kaggle Dataset propio).
- [ ] Fase 8 — `src/train.py::run()` end-to-end, `outputs/submission.csv`.
