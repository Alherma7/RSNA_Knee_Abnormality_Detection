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
      descartados, la hipotesis en ese momento fue que el baseline era
      deliberadamente limitado (un solo corte, un solo plano) — la
      auditoria de abajo encontro que ademas habia bugs reales sin
      relacion con esa limitacion arquitectonica.

- **Auditoria independiente y correccion de la Fase 4 (2026-08-18):**
      un agente (Opus, modo lectura) audito notebooks 00-05, `src/` y
      `tests/` en busca de errores antes de dar el 0.574 por bueno.
      Encontro 4 bugs P0 en el pipeline de imagen, los 3 primeros
      corregidos y re-verificados contra los 58 estudios gold reales en
      Kaggle:
      (1) `normalize_laterality` volteaba el eje equivocado — en un
      corte sagital, medial/lateral es el eje *fuera* de plano (a lo
      largo de la serie), no uno de los dos ejes del propio corte, asi
      que `np.fliplr` no normalizaba nada real y en cambio espejaba
      anterior/posterior para toda rodilla derecha. Verificado con datos
      reales (`ImageOrientationPatient` + `ImagePositionPatient` de 8
      estudios gold, mezcla L/R): la orientacion del escaner es identica
      entre L y R, pero `SliceLocation` ascendente significa
      lateral→medial para R y medial→lateral para L — direcciones
      opuestas. Fix real: invertir el **orden de la lista de cortes**
      (no los pixeles) segun `Laterality`, en `load_series_slices`.
      (2) el 0.574 reportado era el **maximo** de val AUC a lo largo de
      las epocas del mismo fold que decidia el early stopping —
      seleccion y evaluacion sobre los mismos datos. Medido directamente
      de las trazas del barrido de `weight_decay`: optimismo medio
      +0.042, del mismo orden que el margen reportado sobre el azar
      (+0.074). Fix: presupuesto de epocas fijado de antemano (8, sin
      re-elegir por fold), sin seleccion de checkpoint por val, se
      reporta la **ultima** epoca; `torch.manual_seed(42)` anadido.
      (3) sin normalizacion de intensidad — `uint16` crudo de DICOM
      directo a un EfficientNet-B0 preentrenado en ImageNet, sin
      reescalar. Fix: `pydicom.pixels.apply_voi_lut` (VOI LUT real del
      DICOM, `WindowCenter`/`WindowWidth` confirmados presentes) con
      fallback a percentiles 0.5-99.5 si faltaran, seguido de
      reescalado a [0,1].
      (4) el experimento de augmentation de la Fase 4 original estaba
      roto por el mismo motivo que (3): `ColorJitter` sobre un tensor
      con valores en cientos saturaba casi todos los pixeles a 1.0 via
      el clamp interno de torchvision — la conclusion "augmentation no
      ayuda" no estaba respaldada. Con el pipeline ya corregido se
      volvio a probar (arreglando de paso que `RandomAffine`/
      `ColorJitter` sobre un batch entero sacaban un solo sorteo
      aleatorio para las 8 imagenes en vez de uno por imagen): macro
      ROC-AUC 0.516 (std 0.035) vs. 0.532 sin augmentation — **augmentation
      sigue sin ayudar, esta vez confirmado sin el bug de por medio**.
      Las 58 series se reprocesaron con los fixes (1)+(3): 58/58 sin
      errores, shape uniforme (3, 371, 371). Re-entrenamiento con el fix
      (2) aplicado: macro ROC-AUC 3-fold **0.532 (std 0.021)** — sigue
      ganando a 0.5, por un margen mas delgado y creible que el 0.574
      original (~0.042 de esa diferencia era sesgo de seleccion, casi
      exactamente lo medido). El overfitting severo persiste igual
      (train AUC 0.95-0.98 en los 3 folds) pese a la intensidad ya
      normalizada, confirmando que el techo es arquitectonico (un solo
      corte de un solo plano por estudio), no los bugs corregidos aqui.
      **La Fase 4 se cierra con este resultado.** Nada de esto esta
      graduado a `src/` todavia — `src/features.py::normalize_laterality`
      sigue siendo `NotImplementedError`; la graduacion depende de que
      el usuario valide el notebook completo de punta a punta.
      Nota aparte: durante esta auditoria se puso en duda brevemente la
      autenticidad de `data/raw/_reference_kernels/{rsna-knee-baseline-v1,
      rsna-knee-read-the-report-then-the-knee}.ipynb` (ambos anadidos en
      el commit inicial del repo, antes de la fecha en que README decia
      que se habian descargado, y con texto identico entre si pese a ser
      supuestamente de autores distintos) — el usuario confirmo que son
      reales, se mantiene la cita tal cual en RESOURCES.md.

- [x] Investigado (2026-08-17) si un backbone mas potente resuelve el
      techo visto en la Fase 4 (con el numero de entonces, 0.574 —
      corregido a 0.532 el 2026-08-18, ver entrada de auditoria arriba;
      la conclusion de esta investigacion no cambia), o si es un
      problema de cuanta informacion se usa por estudio. Conclusion, con
      fuentes (ver
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

- [x] **Probado el pooling multi-plano MIL (2026-08-18)**, adelantando
      parte de la Fase 6 para verificar con datos propios (no solo citas)
      si la prediccion de MRNet/RadImageNet se cumple a esta escala.
      Diseno: 1 tripete 2.5D por plano (sagital+coronal+axial, no todos
      los cortes como MRNet) por estudio, backbone compartido, max-pool
      de features entre los 3 planos antes de la cabeza. De paso
      corregido un tercer eje de lateralidad no cubierto por el fix de
      sagital: en coronal y axial, medial/lateral SI es un eje dentro
      del plano (columna de imagen = eje x del paciente, verificado con
      `ImageOrientationPatient` real en 16 estudios, 8 por plano,
      mezcla L/R) — ahi el fix correcto es voltear pixeles (`np.fliplr`
      condicionado a `Laterality == "L"`), lo contrario que en sagital.
      Se sumo tambien EMA de pesos del modelo (decay=0.9, mas bajo que
      el 0.9-0.9999 tipico de vision porque solo hay ~40 pasos de
      entrenamiento en total con este dataset — un decay alto dejaria
      el EMA casi congelado en la inicializacion) y evaluacion sobre 3
      seeds x 3 folds en vez de 1 sola corrida, para no confundir ruido
      de fold con senal real (la Fase 4 corregida ya habia mostrado
      ruido intra-fold del mismo orden que el efecto medido).
      **Resultado: sin ganancia clara.** Macro ROC-AUC medio (9
      corridas): 0.530 (std 0.048) sin EMA vs. 0.532 (1 sola corrida)
      del baseline de un plano — estadisticamente indistinguible. Con
      EMA: 0.549 (std 0.041), una mejora pequena y verosimil pero al
      limite de lo que 9 corridas pueden confirmar con fiabilidad.
      Hallazgo mas revelador que el numero en si: **el fold 1 sale
      sistematicamente mas alto que fold 0/2 en las 3 seeds** (0.59-0.63
      vs. 0.48-0.55) — que estudios caen en que fold pesa mas que
      cualquier cambio de arquitectura probado, con n=58 y 3 folds. El
      overfitting tampoco cambio (train AUC 0.94-0.96 igual con MIL que
      con un plano): si triplicar las vistas por estudio no reduce el
      overfitting ni sube el val AUC, el cuello de botella no es cuantas
      vistas ve el modelo por estudio, es cuantos estudios tiene para
      entrenar (38-40 por fold). MRNet junta multi-plano *y* todos los
      cortes de la serie *y* (con toda probabilidad) muchos mas estudios
      de entrenamiento que nuestros 58 gold — replicar solo la parte
      arquitectonica sin mas datos tiene retorno limitado. Confirma con
      evidencia propia (no solo la cita) priorizar la Fase 5 (sumar los
      4,349 estudios weak) sobre seguir afinando arquitectura contra los
      mismos 58 gold. Codigo de este experimento vive solo en
      `notebooks/04_baseline_cnn.ipynb` (celdas corridas en Kaggle,
      2026-08-18) — nada graduado a `src/model.py` todavia, dado el
      resultado neutro.

- [x] **Arreglado el bug de negacion del labeler (2026-08-18)**, antes
      de escalarlo a los 4,349 estudios weak en la Fase 5. Dos fixes
      relacionados en `src/labelers.py`: (1) `_clauses()` ahora separa
      frases aunque no haya espacio tras `.;!?` (15.4% de los 4,407
      informes, 677, corren hallazgos pegados tipo "acl tear.mcl
      normal."); (2) nueva `_negation_applies()` — una negacion solo
      veta `finding` si su argumento local (el texto que gobierna, antes
      o despues del cue segun sea predicado o pre-nominal) menciona la
      propia anatomia o patologia de ese hallazgo, no solo "esta en
      algun sitio de la misma frase". Proceso real, no lineal: un primer
      intento trocaba tambien por coma para separar "ACL is torn but PCL
      is intact" en dos piezas independientes — funcionaba para ese caso
      pero rompia el patron contrario, muy comun en este corpus en los 2
      idiomas: nombrar la estructura una vez y seguir describiendola tras
      una coma sin repetir el nombre ("menisco medial ... conservada, sin
      signos de rotura", "no effusion, synovitis, or bone contusion
      identified") — medido: macro ROC-AUC en gold bajo de 0.688 a 0.674
      con el split por coma, revertido. La version final solo trocea en
      `but`/`pero`/`aunque`/`although` (limites seguros) y usa
      `_negation_applies()` para el resto. Resultado en gold: **0.686**
      (vs. 0.688 original, diferencia despreciable — 10 de 11 celdas que
      cambiaron pasaron de "negativo correcto" a "abstencion", no a un
      error; 1 se corrigio de verdad). El bug real (falso negativo
      confiado en frases compuestas) apenas se nota en gold por la alta
      tasa de abstencion, pero es justo lo que habria contaminado las
      etiquetas de los 4,349 weak sin este fix. 6 tests nuevos en
      `tests/test_labelers.py` (16 en total), incluida una limitacion
      documentada y aceptada (no arreglada): dos clausulas totalmente
      independientes unidas solo por coma sin conjuncion ("radial tear of
      the medial meniscus, the lateral meniscus is normal") no se separan
      de forma fiable — arreglarlo con split-por-coma reintroduce la
      regresion de las listas coordinadas.

- [x] **Fase 5 — mezclar gold+weak con GroupKFold (2026-08-18)**,
      validado en `notebooks/04b_gold_weak_groupkfold.ipynb` (corrido en
      local contra los 4,407 estudios reales, sin falta de Kaggle/GPU) y
      graduado a `src/`: `src/labelers.py::report_group_key()` (hash
      SHA-256 de una normalizacion que colapsa todo el whitespace,
      distinta de `_normalize()` porque esta ultima preserva saltos de
      linea a proposito para `_clauses()` — reusar `_normalize()` aqui
      subconto: 50 grupos/192 estudios en vez de los 54/206 reales,
      corregido) y `src/data.py::load_training_labels()` (ademas gradua
      de paso `load_reports()`/`load_gold_labels()`, que seguian siendo
      `NotImplementedError`). Verificado exacto contra `data/raw/`: 54
      grupos duplicados / 206 estudios / 1 plantilla compartida
      gold-weak (coincide al digito con la Fase 2); `GroupKFold(n_splits=
      CV_FOLDS)` sobre los 4,407 estudios agrupados por plantilla, 0
      grupos repartidos entre folds; gold distribuido 16/11/14/8/9 por
      fold (sin estratificar — a este tamano de muestra ya no hace falta
      el `MultilabelStratifiedKFold` que si hacia falta en el CV de solo
      58 estudios de la Fase 4, y estratificar una mezcla de 0/1 duros y
      0.5 de abstencion no esta bien definido de todas formas); tabla
      combinada con 0 discrepancias entre las filas gold y las columnas
      oficiales de `train.csv` (nunca se pasan por el labeler). 8 tests
      nuevos (4 en `tests/test_labelers.py` para `report_group_key`, 4 en
      `tests/test_data.py`, con CSVs sinteticos en vez de depender del
      `data/raw/train.csv` real para que sean hermeticos). `src/data.py`
      sigue con `load_dicom_series`/`build_dicom_cache`/
      `load_series_metadata` como `NotImplementedError` — la carga de
      DICOM de la Fase 4 sigue sin graduar, es un trabajo aparte.

- [x] **Submission de calibracion (2026-08-24)**: primera submission real
      a la competicion, para saber si el CV local de 58 gold significa
      algo en el leaderboard de verdad. `05c_gold_weak_checkpoint_train.ipynb`
      reentreno `gold_weak`/seed=42 (5 folds, mismos hiperparametros que
      `05b`) guardando checkpoints -- reprodujo el 0.5659 de `05b` exacto
      (delta +0.0000), confirmando que el reentrenamiento con guardado de
      pesos no introduce drift. `06_submission_inference.ipynb` hizo
      inferencia en vivo sobre el test oculto (preprocesado DICOM propio,
      ensemble de los 5 folds) -- el pipeline de preprocesado de
      `05a_weak_dicom_preprocess.ipynb` nunca se sincronizo a este repo y
      se confirmo perdido tambien en Kaggle, asi que las celdas de
      lateralidad/normalizacion de intensidad se reconstruyeron de la
      prosa del README y se validaron aparte
      (`06b_preprocessing_validation.ipynb`) contra los 58 triples gold
      ya conocidos de `triplets_knee`: la direccion del fix de
      lateralidad salio perfecta (58/58 en orden directo, no invertido)
      pero queda un residuo de intensidad sin explicar en ~10/58 estudios
      (MAE moderado, no correlacionado con lateralidad; 3 hipotesis
      descartadas -- corte, gap, PhotometricInterpretation/RescaleSlope)
      que se acepto como limitacion conocida en vez de seguir
      persiguiendolo (ver memoria `feedback_match_debugging_effort_to_stakes`).
      **Resultado: leaderboard real 0.596**, por encima del CV local de
      este mismo modelo (0.5659 en `05c`, 0.5711 media-3-seeds en `05b`)
      -- confirma que el CV de 58 gold es razonablemente informativo, no
      esta desconectado del metric real. Contraste importante encontrado
      en discussion de la competicion (post `735304`): top scorers
      publicos reportan 0.887-0.94+ con DINOv2/ensembles/mas datos por
      estudio/etiquetas via LLM -- brecha real y grande (0.596 vs
      0.89-0.94) que motiva revisar arquitectura, representacion de
      imagen (mas cortes/secuencias por estudio) y calidad de las weak
      labels antes de seguir iterando solo contra el CV local.

## Next steps

- [ ] Revisar `notebooks/04_baseline_cnn.ipynb` completo (ya con los
      fixes de lateralidad/intensidad/fuga de metrica del 2026-08-18) y
      decidir que (si algo) gradua a `src/data.py`/`src/features.py`/
      `src/model.py`, con sus tests correspondientes — incluyendo tests
      de regresion para los 2 bugs reales encontrados en la auditoria
      (orden de cortes por lateralidad, fuga de seleccion de metrica).
- [ ] Fase 6 — `notebooks/05_ensemble_calibration.ipynb`: ensembling
      multi-plano/backbone con `src/evaluate.py::rank_blend()`. El
      experimento MIL 3-plano del 2026-08-18 (ver arriba) no gano con
      solo 58 gold y 1 triplete/plano — si se retoma tras la Fase 5 (mas
      estudios de entrenamiento), probar tambien mas cortes por plano
      (varios tripletes o todos los cortes con max-pool), mas cerca del
      diseno real de MRNet, antes de descartar el pooling multi-plano
      por completo.
- [ ] Fase 7 — optimizacion de inferencia para el limite de 9h sin
      internet (pesos empaquetados como Kaggle Dataset propio).
- [ ] Fase 8 — `src/train.py::run()` end-to-end, `outputs/submission.csv`.
