# Análisis de errores del modelo final (fase 4)

Para decidir mejoras se usa **val**: la receta final (base 16, dropout 0,5, sin profundidad) entrenada solo con train, 5 semillas (`final_model.py val`). El test ya se gastó (`research/10`); sus errores se muestran solo como descripción para la model card. `scripts/checks/error_analysis_final.py` → `results/models/error_fp_tipo.csv`, `error_nosignal.csv`.

## 1. Tipos de falso positivo

- **Kepler:** motivos del Robovetter (`koi.csv`).
- **TESS:** comentarios de ExoFOP (`data/labels/exofop_toi.csv`, descargado el 2026-11-05) clasificados por palabras clave. Son etiquetas ruidosas: un tercio de los FP solo dice "TFOP FP" o "found in faint-star QLP search".
- **K2:** no publica motivo.

Tipos:

- **Binaria:** EB/SEB, forma en V, secundario, par/impar. Visible en la curva.
- **Contaminación:** NEB, centroide desplazado, coincidencia de efemérides. La señal es de otra estrella: es invisible en la curva de la estrella objetivo.
- **No tránsito:** artefactos, ruido.

| Val | n | Acierta FP | → PLANET | → NO_SIGNAL |
|---|---|---|---|---|
| Kepler, binaria | 209 | 0,88 | 0,09 | 0,03 |
| Kepler, solo contaminación | 91 | 0,60 | **0,29** | 0,11 |
| Kepler, no tránsito | 79 | 0,71 | 0,05 | **0,24** |
| TESS, binaria | 26 | 0,62 | **0,31** | 0,08 |
| TESS, solo contaminación | 29 | 0,66 | 0,14 | 0,21 |
| TESS, sin motivo reconocido | 32 | 0,53 | 0,19 | 0,28 |
| TESS, todos | 89 | 0,58 | 0,20 | 0,21 |
| K2, todos | 19 | 0,95 | 0,05 | 0 |

En test (descriptivo), TESS va peor: el 38 % de los FP se predicen como planeta (val: 20 %). Por tipo, contaminación 41 % y binaria 26 %.

- **Kepler:** la mitad de los FP que pasan por planeta son de contaminación. Es el **techo físico**: con solo la curva no se puede distinguir. Solo lo arreglaría el centroide (fase 2: del 66 % al 80 %), que no es la curva.
- **TESS:** las **binarias**, que sí se ven en la curva, se escapan como planeta un 31 % (Kepler: 9 %). Aquí hay margen. El modelo no recibe las vistas par/impar ni la del secundario, que es como AstroNet-Vetting y ExoMiner detectan binarias.
- **"No tránsito" → NO_SIGNAL:** es la frontera difusa entre esas dos clases ya vista en la fase 2. Es un error de etiquetado más que del modelo.

## 2. NO_SIGNAL frente a PLANET

La precisión de PLANET si hubiera r NO_SIGNAL por planeta se calcula como tasa de detección / (tasa de detección + r · tasa de NO_SIGNAL → PLANET). Ignora los FP.

| Val | NO_SIGNAL → PLANET | Planeta detectado | ROC-AUC PL vs NS | Precisión r = 1 | r = 10 | r = 100 |
|---|---|---|---|---|---|---|
| Kepler, catálogo | 1,1 % | 0,92 | 0,998 | 0,99 | 0,90 | 0,46 |
| K2, catálogo | 3,0 % | 0,77 | 0,959 | 0,96 | 0,72 | 0,20 |
| TESS, catálogo | 1,3 % | 0,81 | 0,976 | 0,98 | 0,86 | 0,38 |
| Kepler, BLS | **19,7 %** | 0,81 | 0,877 | 0,80 | 0,29 | 0,04 |
| K2, BLS | **42,9 %** | 0,60 | 0,673 | 0,58 | 0,12 | 0,01 |
| TESS, BLS | **35,3 %** | 0,75 | 0,796 | 0,68 | 0,17 | 0,02 |

- Con nuestros NO_SIGNAL (estrella tranquila plegada con una efeméride prestada), PLANET y NO_SIGNAL casi no se confunden. Aun así, con 100 NO_SIGNAL por planeta, más de la mitad de los "planetas" serían falsos.
- **Con negativos realistas, plegados en un pico de ruido que BLS confunde con un tránsito,** entre el 20 % y el 43 % pasan por planeta. Así es "no hay señal" en la práctica: falsas detecciones de la tubería. El modelo **no ha visto nunca este tipo de negativo al entrenar**, y en ese caso no es fiable.
- En K2, 4 de 17 planetas de val se predicen NO_SIGNAL (n pequeño).

## Opciones (decisión de Jairo pendiente)

1. **Negativos difíciles en el entrenamiento.** Ejecutar BLS en los NO_SIGNAL de train y añadir sus vistas, plegadas en el pico de ruido, como NO_SIGNAL. Ataca directamente el punto 2. Coste: BLS en ~5 800 estrellas, ~5 h (o ~2 h con 1 500 de Kepler).
2. **Vistas par/impar y secundario.** Ataca las binarias que se escapan como planeta, sobre todo en TESS. Par/impar ya está en las vistas; el secundario hay que construirlo.
3. **Umbral de PLANET** elegido en val para una precisión objetivo con un prior realista. Sin reentrenar; se cambia recall por precisión.

Cualquier mejora se mide en val. La cifra de test de `research/10` sigue siendo la oficial. Volver a usar el test exigiría declararlo como segunda evaluación.

## 3. Experimento: negativos difíciles (decisión de Jairo, 2026-11-05)

- BLS sobre los NO_SIGNAL de train (`build_bls_views.py --splits train --nosignal`, 12 procesos, ~3 h): 5 415 vistas válidas, plegadas en el pico de ruido.
- Se añaden como NO_SIGNAL a train: 11 732 + 5 415 ejemplos. Las dos copias de una estrella van al mismo lado de la parada temprana.
- Misma receta y 5 semillas (`scripts/models/hard_negatives.py`). Se compara en val con la línea base (`final_model.py val`).

| Val | Macro-F1 catálogo (base → negativos difíciles) | Macro-F1 BLS | NO_SIGNAL BLS → PLANET | NO_SIGNAL BLS → FP | Precisión PLANET r = 10 (BLS) |
|---|---|---|---|---|---|
| Kepler | 0,874 → **0,839** | 0,41 → **0,62** | 19,7 % → 7,7 % | 69 % → 3 % | 0,29 → 0,48 |
| K2 | 0,891 → 0,862 | 0,41 → 0,47 | 43 % → 24 % | 52 % → 14 % | 0,12 → 0,16 |
| TESS | 0,784 → **0,745** | 0,37 → **0,56** | 35 % → 22 % | 51 % → 8 % | 0,17 → 0,24 |

- **Gana robustez:** el ruido con forma de tránsito ya no pasa por FP (69 % → 3 % en Kepler) y pasa mucho menos por planeta. Con efemérides de BLS, el macro-F1 sube ~0,2 en Kepler y TESS.
- **Pierde en el caso de catálogo** (−0,03/−0,04 de macro-F1). El modelo aprende "tránsito débil o ruidoso → NO_SIGNAL", y ahí caen también FP débiles (FP de Kepler → NO_SIGNAL: 10 % → 18 %) y algunos planetas (Kepler 1,1 % → 4,5 %).
- **El orden mejora poco:** el ROC-AUC PLANET vs NO_SIGNAL con BLS pasa de 0,88 a 0,90 (Kepler) y de 0,80 a 0,84 (TESS). Sobre todo cambia a qué clase van los casos dudosos.
- Matiz: con vistas BLS, parte de los "planetas → NO_SIGNAL" son planetas a los que BLS dio un periodo erróneo (~30 %). Ahí NO_SIGNAL es la respuesta razonable.
- Los NO_SIGNAL fáciles (catálogo) no cambian: 1,1–1,3 % → PLANET en los dos modelos.

**Decisión de Jairo (2026-11-06):** para las señales con efeméride dada (TOIs, informe de un candidato) se usa el modelo base; el de negativos difíciles queda solo para la búsqueda sin efeméride. La frontera FP «no parece tránsito» / NO_SIGNAL se deja como está en la v1 y se documenta en el informe.

## 4. Triaje en modo búsqueda: CANDIDATO frente a SIN_SEÑAL (2026-11-06)

Prioridad fijada por Jairo: que sea fiable distinguir CANDIDATO (PLANET o FP: hay una señal real) de SIN_SEÑAL, solo con la curva.
- Los nombres son solo de presentación; en el código sigue `NO_SIGNAL`.
- Con la efeméride del catálogo ya es fiable: ROC-AUC 0,95–0,99.
- El problema está en modo búsqueda.

**Prueba barata** (`scripts/checks/triage_val.py` → `results/models/triage_val.csv`):
- Estrellas: las 1 403 de val con vista BLS.
- Puntuaciones comparadas: la SDE de BLS, la probabilidad de candidato de la CNN, y las dos combinadas con una regresión logística. La combinación se puntúa fuera de pliegue (5 folds).
- Métrica: recall de candidatos con el umbral que deja pasar el 1 % de los SIN_SEÑAL.

| Val, todas las misiones | ROC-AUC | Recall con 1 % de falsas alarmas | Recall con 5 % |
|---|---|---|---|
| SDE sola | 0,855 | 0,26 | 0,48 |
| CNN final | 0,788 | 0,11 | 0,36 |
| CNN con negativos difíciles | 0,809 | 0,31 | 0,47 |
| SDE + CNN final | 0,868 | 0,36 | 0,60 |
| **SDE + CNN con negativos difíciles** | **0,877** | **0,47** | **0,66** |

Por misión, para la mejor combinación (recall con 1 %): Kepler 0,48, TESS 0,41, K2 0,73. K2 tiene solo 21 SIN_SEÑAL, así que su 1 % no es fiable.

**Dónde está el límite:**
- Si BLS encuentra el periodo correcto (68 % de los candidatos), el recall con 1 % de falsas alarmas es **0,65**.
- Si BLS da otro periodo (32 %), baja a **0,11**.
- El clasificador funciona razonablemente bien. **El cuello de botella es la detección de BLS.**

## 5. Prueba: los 3 mejores picos de BLS (2026-11-06) — descartada

**Qué se hizo:**
- Cada estrella de val se pliega con sus 3 picos distintos de mayor SDE (`bls.find_ephemerides`, `scripts/data/build_bls_peaks.py`). El primer pico reproduce exactamente el BLS anterior.
- La CNN con negativos difíciles (pesos ahora guardados en `data/models/hard_seed*.pt`; el reentrenamiento da las mismas predicciones) puntúa cada vista.
- La regresión SDE + CNN se ajusta con el pico 1 de los otros folds y la estrella toma el máximo de sus picos.

**Resultados:**
- El periodo correcto aparece en el pico 1 en el 68 % de los candidatos y en alguno de los 3 en el 75 %.

| Val, todas las misiones | ROC-AUC | Recall con 1 % de falsas alarmas | Recall con 5 % |
|---|---|---|---|
| SDE + CNN, pico 1 | 0,878 | 0,474 | 0,667 |
| SDE + CNN, mejor de 3 picos | 0,873 | 0,470 | 0,680 |

**Conclusión:** sin mejora.
- Solo rescata un 7 % de periodos, y el ruido gana a la vez 3 oportunidades de parecer tránsito.
- Para el 25 % de los candidatos, ningún pico de BLS tiene el periodo correcto. **BLS está en su límite con estas curvas**: el siguiente paso tiene que ser un detector que no dependa de BLS.

**Recall de candidatos frente a ruido que pasa** (SDE + CNN, pico 1, val):

| Candidatos detectados | 80 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|
| Ruido que pasa, búsqueda (BLS) | 19 % | 46 % | 67 % | 95 % |
| Ruido que pasa, periodo de catálogo (CNN final) | 0,1 % | 2,7 % | 11 % | 44 % |

Objetivo de Jairo: falsas alarmas ≤ 1 % al final de la cadena (triaje + modelo final) sin perder candidatos. La fase 1 debe priorizar el recall.
