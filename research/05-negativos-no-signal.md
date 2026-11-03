# Negativos NO_SIGNAL y test de atajo

Tarea #32, 2026-11-01. Código: `src/exoplanet/negatives.py`, `scripts/data/build_negatives.py` (lista), `scripts/data/build_sample.py --negatives` (descarga y plegado), `scripts/checks/shortcut_negatives.py` (test de atajo). Tests: `tests/test_negatives.py`, `tests/test_curves.py`.

## Problema

Las vistas NO_SIGNAL del dataset de Hugging Face son ceros, así que los negativos se fabrican: estrellas sin candidato, descargadas y plegadas con la misma función que los positivos y con una efeméride al azar. El riesgo es que el modelo distinga NO_SIGNAL por **cómo se fabricó** (cantidad de datos, campaña, brillo) y no por la ausencia de tránsito.

## Diseño final

| | Kepler | K2 | TESS |
|---|---|---|---|
| Estrellas | NO_SIGNAL del dataset | sin candidato, fuera del dataset | sin TOI, fuera del dataset |
| Emparejamiento | split | 1:1 con cada positivo: misma campaña del segmento usado y Kp más cercano | 1:1: mismo sector y Tmag más cercano |
| Segmentos | ~100 d (D3) | 1 campaña | 1 sector |
| Efeméride (P, duración) | pareja real de positivos de train de la misión | pareja real de positivos de train de la misma campaña | ídem, mismo sector |
| Negativos | 1000 | 398 | 400 (4 sin curva en MAST) |

Época uniforme en [0, P). Cada negativo de K2/TESS hereda el split de su pareja. Diferencia de magnitud con la pareja: mediana 0,00, p90 ≤ 0,01.

**Reproducible** (2026-11-01): un generador por misión (`MISSION_SEED`) y orden fijo de estrellas y candidatos (ni MAST ni las descargas en paralelo lo garantizan). Dos ejecuciones seguidas de `build_negatives.py` dan el mismo CSV byte a byte. La versión anterior (979 en Kepler, de una muestra previa) queda en `negatives_before_reproducible.csv`.

## Cómo se llegó aquí (ROC-AUC NO_SIGNAL vs positivos sin mirar el tránsito, boosting)

Durante la tarea estas cifras se midieron en val+test. En la auditoría (2026-11-01) el script pasó a evaluar solo en val, sin tocar el test; los números de la última fila con ese criterio son K2 0,86, Kepler 0,60 y TESS 0,58. Que las decisiones se tomaran mirando también el test es una contaminación leve: son diagnósticos de los datos, no selección de modelos.

| Paso | Kepler | K2 | TESS | Atajo encontrado |
|---|---|---|---|---|
| Negativos al azar por campaña/sector, todos los segmentos | 0,64 | 0,90 | 0,83 | los positivos tienen más sectores (TESS: mediana 3 frente a 1) |
| 1 segmento en K2/TESS | 0,64 | 0,87 | 0,70 | mezcla de campañas distinta (K2) y ruido (TESS) |
| Emparejados por segmento y magnitud | 0,64 | 0,85 | 0,51 | periodo típico distinto por campaña (K2) |
| + efeméride por segmento y C10 corregida | 0,64 | 0,84 | 0,55 | — |

La campaña 10 de K2 se publica en dos mitades (archivos `c101`/`c102`) pero MAST la numera como 10 (`curves.mast_sequence`); sin eso, 18 positivos se quedaban sin pareja.

## Lo que ve el modelo

Boosting sobre la vista global **con el tránsito tapado** (20 % central), AUC con validación cruzada de 5 particiones
**dentro de train** (`results/sample/shortcut_negatives_views.csv`):

| | Kepler | K2 | TESS |
|---|---|---|---|
| PLANET vs NO_SIGNAL | 0,59 | 0,59 | 0,57 |
| FP vs NO_SIGNAL | 0,65 | 0,84 | 0,49 |

Con el sorteo anterior de negativos (no reproducible) PLANET daba 0,51 / 0,56 / 0,47 y FP 0,64 / 0,86 / 0,50: el AUC de PLANET varía ~0,05-0,10 solo por qué estrellas sin señal salen en el sorteo. Hay que leerlo como "0,5-0,6", no como un valor exacto.

- **FP**: separación sobre todo astrofísica (binarias y estrellas variables: eclipse secundario y variabilidad fuera del tránsito; en K2 el ruido fuera de tránsito y secundario da 0,80). Es señal que el modelo debe aprender.
- **PLANET (~0,57-0,59) y ruido en Kepler (~0,56)**: sesgo de detección, los planetas se encuentran más en estrellas tranquilas. Se acepta y se declara.

## Limitaciones y reglas para la fase 2

- Los metadatos del proceso (`n_points`, `data_days`, `cadence_days`, `n_transits`, `frac_empty_*`, `n_files`) aún separan en K2 (AUC CV ~0,67). **Nunca se usan como rasgos.**
- Evaluar también por clase y por misión, para ver si NO_SIGNAL se apoya en el ruido.
- No se emparejó por ruido medido (CDPP): quitaría el sesgo de detección pero haría los negativos menos representativos (decisión 2026-11-01).
- Con 1 sector, el % de vistas válidas de PLANET en TESS baja de 96,3 % a 87,8 % (planetas de periodo largo con pocos tránsitos).

Resultados: `results/sample/shortcut_negatives.csv` (final) y `*_multisegment.csv`, `*_matched_v1.csv` (pasos intermedios).

## Muestra completa (2026-11-03)

8 582 negativos 1:1 (Kepler 6 560, TESS 1 624, K2 398) con el mismo procedimiento; 11 de TESS sin curva en MAST. Vistas válidas: NO_SIGNAL Kepler 83,5 %, K2 91,2 %, TESS 89,3 %.

Test de atajo con la vista global sin tránsito (CV en train):

| | Kepler A → completa | K2 | TESS A → completa |
|---|---|---|---|
| PLANET vs NO_SIGNAL | 0,59 → 0,60 | 0,59 | 0,57 → 0,55 |
| FP vs NO_SIGNAL | 0,65 → 0,69 | 0,84 | 0,49 → 0,54 |

Metadatos y ruido (train → val, boosting): Kepler 0,64, TESS 0,59, K2 0,77 (K2 sin cambios). Con ~5 veces más datos, el atajo sigue siendo débil para PLANET y estable. Bordes: sin artefactos en NO_SIGNAL ni en PLANET (|z| > 3 ≤ 0,2 % en la vista global). Los FP de K2 dan 32 % en la global, que es física (eclipses secundarios cerca de la fase 0,5), no un artefacto.
