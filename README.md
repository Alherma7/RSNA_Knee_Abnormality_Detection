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
- [x] `notebooks/00_export_local_gold_subset.ipynb` corrido en Kaggle
      (2026-08-17): 58/58 estudios gold identificados; CSVs exportados
      (`train.csv` 5.7 MB, `train_series.csv` 3.5 MB); DICOM gold miden
      7.01 GB (no copiados esta vez — `PROCEED_WITH_DICOM_COPY=False`,
      solo hacia falta texto para la Fase 2; caben de sobra en los
      ~515 GB libres si una fase posterior los necesita). Zip de 2.2 MB
      extraido en `data/raw/` local — operacionaliza la decision de
      infraestructura del 2026-08-16 (bajar solo el gold localmente).
- [x] Fase 2 — `notebooks/02_eda_reports.ipynb` corrido en local
      (2026-08-17, contra `data/raw/` real, via `nbconvert --execute`)
      tras el export de arriba. Hallazgos:
      longitud comparable entre gold (n=58, media 1305 caract./179
      palabras) y weak (n=4349, media 1095 caract./147 palabras) — gold
      ~19% mas largo en promedio, no una distribucion distinta, asi que
      validar Fase 3 contra el gold no es obviamente sesgado por
      longitud; hard-wrap presente pero minoritario — solo 23.2% de los
      informes tiene >=1 linea candidata (frac_wrap_candidates mediana
      0.0), bastante menos que el "gran parte del corpus" que sugiere
      prvsiyan — y el heuristico de deteccion (copiado del notebook de
      referencia) tiene un falso positivo real: en reportes con vinetas
      (`> hallazgo`), el simbolo `>` no cuenta como mayuscula y el check
      lo marca como wrap aunque sean lineas independientes — a corregir
      si se construye `unwrap()` en la Fase 3, no antes; script dominante
      por informe: 87.7% Latino (3,866), 7.3% Griego (321), 5.0%
      Cirilico (220) — `langdetect` no esta instalado localmente asi que
      no hay desglose fino de idiomas dentro del bloque Latino, pero
      inspeccion manual confirmo al menos turco, ingles y espanol ahi
      (contrastar `REPORT_LANGUAGE_COUNT=9` de `config.py` sigue
      pendiente hasta tener ese desglose); 54 grupos de informes
      duplicados (texto identico tras normalizar), 206 estudios (4.7%)
      — la plantilla mas repetida es turca ("rodilla normal", 37 veces)
      — **1 plantilla se comparte entre gold y weak**, confirmando que
      `report_group_key` (Fase 5) importa tambien para no filtrar
      gold<->weak, no solo weak<->weak; vocabulario: 15,492 tokens
      distintos, 646,582 tokens totales — el top-40 ya mezcla terminos
      anatomicos en ingles/espanol (medial, ligament, meniscus, tear,
      cartilage, patellofemoral...) con tokens griegos (`του`) y
      cirilicos (`на`), confirmando que las labeling functions de la
      Fase 3 deben poner a prueba las nueve lenguas a la vez, no
      enrutar por idioma detectado. Detalle completo en `RESOURCES.md`.

- [x] Fase 3 — `src/labelers.py::label_report()`/`label_reports()`
      graduados a `src/` (2026-08-17), validados en
      `notebooks/03_labeler_validation.ipynb` contra el gold real (corrido
      en Kaggle y en local, mismo resultado en ambos): macro ROC-AUC
      **0.688** vs. 0.5 del baseline constante, los 12 hallazgos por
      encima de 0.5 individualmente (el mas debil:
      `oa_lateral_compartment`, 0.553, silence rate ~90%). Proceso real,
      no lineal: una primera version ingenua ("patologia gana sobre
      negacion") puntuo `effusion` en AUC 0.438 (peor que azar) porque
      "no effusion" contiene la palabra "effusion" y disparaba ambos
      cues a la vez — corregido dando prioridad a la negacion. El
      lexicon de los 3 compartimentos de OA se amplio tras inspeccionar
      informes gold reales (no adivinar), agregando sinonimos vistos de
      verdad ("cartilage fissuring", "spurring", "subchondral cystic
      change"). Cobertura de esta primera version: solo ingles y
      espanol (las 2 lenguas dominantes, Fase 2 seccion F); griego/
      cirilico (~12% de los informes) caen en abstencion (0.5), no en
      una respuesta erronea con confianza — pendiente de ampliar.
      10 tests nuevos en `tests/test_labelers.py`, incluidos dos
      regresiones directas de los bugs encontrados (prioridad de
      negacion, y el fix de vinetas de `unwrap()` de la Fase 2).
      Limitacion abierta: n=58 es un set de validacion pequeno (min. 9
      positivos en MCL) — seguir ajustando el lexicon contra este mismo
      set arriesga sobreajustarlo. `report_group_key()` queda pendiente
      para la Fase 5.

- [x] Fase 4 — `notebooks/04_baseline_cnn.ipynb` construido y corrido
      celda a celda en Kaggle (2026-08-17, spec en
      `docs/superpowers/specs/2026-08-17-fase4-baseline-cnn-design.md`):
      baseline 2.5D CNN (EfficientNet-B0, solo plano sagital) entrenado
      con las 58 filas gold, 3-fold CV estratificado
      (`MultilabelStratifiedKFold`). **Resultado: macro ROC-AUC 0.574
      (std 0.052)** vs. 0.5 del baseline constante — gana, los 3 folds
      individualmente por encima de 0.5 (0.595, 0.515, 0.612). Config
      ganadora: gap fisico entre cortes GAP_MM=4.0 (no en indices —
      el espaciado real varia 13.75x entre series, 0.4-5.5mm),
      TARGET_MM_PER_PIXEL=0.35, crop fisico CROP_MM=130.0 (pilkwang),
      learning rate diferencial (backbone 1e-5, cabeza 1e-3),
      dropout=0.5, weight_decay=1e-2, early stopping sobre val macro
      AUC, **sin augmentation**. Proceso real: el primer run sin
      regularizar mostro overfitting severo (train AUC 0.698 vs. val
      0.492); un barrido de weight_decay (0 a 1e-2) mejoro el val AUC
      progresivamente; augmentation (rotacion/jitter/traslacion, sin
      flip horizontal por `normalize_laterality`) se probo despues, en
      varias combinaciones, y **no gano** contra weight_decay solo —
      resultado negativo honesto, no descartado, candidato a revisar en
      Fase 5 con mas datos. Caveat honesto: train AUC llega a 0.95-0.99
      en los 3 folds (memorizacion casi total de ~38-40 estudios por
      fold) — que aun asi generalice a 0.574 es una senal real pero
      fragil, no un modelo robusto.
      **Diagnostico post-resultado (2026-08-17):** el usuario pregunto
      si 0.574 con esa fragilidad podia deberse a un bug de etiquetado/
      EDA en vez de a una limitacion real del modelo. Verificado: (1)
      0/58 discrepancias entre `gold_labels` y las columnas originales
      de `train.csv` — descarta un bug de desalineacion imagen-etiqueta;
      (2) inspeccion visual de 2 estudios positivos (ACL + otros
      hallazgos) confirmo que el crop de 130mm captura la articulacion
      correctamente, centrada, sin recortes obvios — aunque el muestreo
      para el chequeo fallo en encontrar un estudio "sin ningun
      hallazgo" (los 58 gold tienen al menos 1 positivo), asi que no
      hubo contraste visual positivo-vs-negativo real. Con ambos bugs
      descartados, la explicacion mas probable es que el baseline es
      deliberadamente limitado: un solo corte central de un solo plano
      por estudio, sin contexto multi-plano — si el hallazgo no cae en
      ese corte exacto, es invisible para el modelo. Pendiente: revision
      del usuario del notebook completo antes de graduar nada a `src/`.

- [x] Investigado (2026-08-17) si un backbone mas potente resuelve el
      techo visto en la Fase 4, o si es un problema de cuanta
      informacion se usa por estudio. Conclusion, con fuentes (ver
      RESOURCES.md, entradas MRNet y RadImageNet): **no es el tamano
      del backbone**. MRNet (Bien et al. 2018) logra AUC 0.937-0.965
      con un backbone mas simple (AlexNet) que el nuestro
      (EfficientNet-B0), agregando TODOS los cortes de las 3 series
      (axial+coronal+sagital) por max-pooling, no un solo corte central
      de un solo plano como nuestro baseline. Los notebooks de
      referencia coinciden: tratan el pooling multi-plano por atencion
      como necesario, no opcional, y advierten contra un DINOv2
      *congelado* (no es "el mejor modelo" per se, esta acotado por
      vocabulario de imagen natural) — el backbone es secundario frente
      a fine-tunear y usar mas vistas. RadImageNet (Mei et al. 2022) es
      un candidato mas fundamentado que agrandar el backbone, si se
      prueba cambiar de backbone (+4.5-4.8% AUC en MRI pequenas,
      incluida rotura de ACL/menisco). Confirma priorizar el pooling
      multi-plano de `src/model.py` (Fase 6) sobre cambiar de backbone
      en la Fase 4.

## Next steps

- [ ] Revisar `notebooks/04_baseline_cnn.ipynb` completo y decidir que
      (si algo) gradua a `src/data.py`/`src/features.py`/`src/model.py`,
      con sus tests correspondientes.
- [ ] Fase 5 — mezclar gold+weak, `GroupKFold` por hash del informe
      (`src/labelers.py::report_group_key`), no solo por estudio.
- [ ] Fase 6 — `notebooks/05_ensemble_calibration.ipynb`: ensembling
      multi-plano/backbone con `src/evaluate.py::rank_blend()`.
- [ ] Fase 7 — optimizacion de inferencia para el limite de 9h sin
      internet (pesos empaquetados como Kaggle Dataset propio).
- [ ] Fase 8 — `src/train.py::run()` end-to-end, `outputs/submission.csv`.
