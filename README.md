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

De 4,407 estudios de entrenamiento, solo un subconjunto pequeno (58,
confirmado exacto — ver Fase 1) tiene etiqueta oficial; el resto solo
tiene el informe. Ademas,
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
- [x] Decision de infraestructura (2026-08-16): el dataset completo
      pesa ~0.5 TB y no cabe con comodidad en el disco local (~515 GB
      libres). El trabajo pesado (Fases 1+: lectura de DICOM,
      entrenamiento) se hace en **Kaggle Notebooks**, contra el
      dataset ya montado ahi, en vez de descargarlo entero en local.
      Solo se baja localmente un subconjunto pequeno (el gold) para
      prototipar y para los tests. `src/config.py::DATA_RAW_DIR`
      detecta el entorno (`ON_KAGGLE`) y apunta a
      `/kaggle/input/competitions/rsna-knee-abnormality-detection` alli
      (confirmado en un kernel real 2026-08-16 — Kaggle monta las
      competiciones bajo `competitions/<slug>/`, no `<slug>/` a secas),
      o a `data/raw/` en local. Sincronizacion notebooks<->Kaggle: manual
      por ahora (sin `kaggle kernels push/pull` automatizado). El
      usuario gestiona el lado de Kaggle (reglas, descargas, kernels)
      el mismo — Claude no ejecuta `kaggle download` en este proyecto.
- [x] Fase 0 — `src/config.py::FINDINGS` verificado contra la
      "Dataset Description" oficial de Kaggle (pegada por el usuario
      2026-08-16): los 12 hallazgos coinciden en contenido y orden.
      De paso, la descripcion oficial corrigio dos supuestos: (1) no
      existe `train_labels.csv` ni `reports.csv` — las 12 etiquetas y
      `Report` viven en `train.csv`; (2) `test_series.csv`/
      `test_series/` existen con el mismo schema que train (necesario
      para inferencia). Su tercer aviso (que `Fluid_Sensitive`/
      `Fat_Suppression` no siempre coinciden) result'o ser demasiado
      cauto frente a los datos reales — ver Fase 1 abajo.
- [x] Fase 1 — `notebooks/01_eda_dicom.ipynb` (secciones A-F,
      completa), corrido en Kaggle 2026-08-16 contra el dataset
      montado. Hallazgos:
      `train.csv` tiene exactamente 58 filas con las 12 etiquetas
      pobladas y 4,349 con ninguna — patron todo-o-nada, cero parcial;
      `Report` no nulo en las 4,407 filas; `Report` ausente en
      `test.csv`; los 3 planos siempre presentes en los 4,407 estudios
      de train; `Fluid_Sensitive` == `Fat_Suppression` en el 100% de
      las 24,371 filas de `train_series.csv` (0 desacuerdos — contradice
      el aviso de la Dataset Description oficial, confirma al notebook
      pilkwang); `InstanceNumber`/`ImagePositionPatient`/`SliceLocation`/
      `Laterality` sobreviven el allowlist de 86 tags; orden por
      filename confirmado no fiable (rho -0.24 a 0.69 en 10 series);
      **bug real cazado en la propia exploracion**: indexar
      `ImagePositionPatient[2]` asume eje axial y falla en planos
      sagital/coronal — usar `SliceLocation` en su lugar, que da
      rho(InstanceNumber, SliceLocation) = ±1.000 exacto en las 10
      series (mismo orden, pero el signo cambia entre series — ordenar
      por un solo campo de forma consistente); unica transfer syntax
      vista en la muestra (30 estudios): Explicit VR Little Endian sin
      comprimir, 0 fallos de decodificacion; ruta real de montaje en
      Kaggle es `/kaggle/input/competitions/<slug>/`, no `<slug>/` a
      secas (corregido en `src/config.py`); `PixelSpacing` (165 series,
      30 estudios) va de 0.137 a 0.703 mm/pixel (ratio 5.14x max/min),
      confirmando que hace falta `normalize_physical_scale` — el
      espaciado es similar entre planos (0.33-0.35 mm de media cada
      uno), la variacion es por estudio/protocolo, no por plano; fila y
      columna con el mismo spacing siempre (isotropico); `SliceThickness`
      0.6-5.0 mm; `SpacingBetweenSlices` ausente en 14/165 series (usar
      `SliceThickness` o deltas de `SliceLocation` como respaldo).
      Detalle completo en `RESOURCES.md`. Ajustados `src/data.py`,
      `src/features.py` (`normalize_laterality`,
      `normalize_physical_scale`) y `src/config.py` en consecuencia.

## Next steps

- [ ] Fase 2 — `notebooks/02_eda_reports.ipynb`: vocabulario, idiomas.
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
