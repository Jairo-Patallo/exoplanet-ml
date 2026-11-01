# Negativos NO_SIGNAL y test de atajo

Tarea #32, 2026-11-01. Código: `src/exoplanet/negatives.py`, `scripts/build_negatives.py` (lista), `scripts/build_sample.py --negatives` (descarga y plegado), `scripts/shortcut_negatives.py` (test de atajo). Tests: `tests/test_negatives.py`, `tests/test_curves.py`.

## Problema

Las vistas NO_SIGNAL del dataset de Hugging Face son ceros, así que los negativos se fabrican: estrellas sin candidato, descargadas y plegadas con la misma función que los positivos y con una efeméride al azar. El riesgo es que el modelo distinga NO_SIGNAL por **cómo se fabricó** (cantidad de datos, campaña, brillo) y no por la ausencia de tránsito.

## Diseño final

| | Kepler | K2 | TESS |
|---|---|---|---|
| Estrellas | NO_SIGNAL del dataset | sin candidato, fuera del dataset | sin TOI, fuera del dataset |
| Emparejamiento | split | 1:1 con cada positivo: misma campaña del segmento usado y Kp más cercano | 1:1: mismo sector y Tmag más cercano |
| Segmentos | ~100 d (D3) | 1 campaña | 1 sector |
| Efeméride (P, duración) | pareja real de positivos de train de la misión | pareja real de positivos de train de la misma campaña | ídem, mismo sector |
| Negativos | 979 | 398 | 400 (9 sin curva en MAST) |

Época uniforme en [0, P). Cada negativo de K2/TESS hereda el split de su pareja. Diferencia de magnitud con la pareja: mediana 0,00, p90 ≤ 0,01.

## Cómo se llegó aquí (ROC-AUC NO_SIGNAL vs positivos sin mirar el tránsito, boosting, val+test)

| Paso | Kepler | K2 | TESS | Atajo encontrado |
|---|---|---|---|---|
| Negativos al azar por campaña/sector, todos los segmentos | 0,64 | 0,90 | 0,83 | los positivos tienen más sectores (TESS: mediana 3 frente a 1) |
| 1 segmento en K2/TESS | 0,64 | 0,87 | 0,70 | mezcla de campañas distinta (K2) y ruido (TESS) |
| Emparejados por segmento y magnitud | 0,64 | 0,85 | 0,51 | periodo típico distinto por campaña (K2) |
| + efeméride por segmento y C10 corregida | 0,64 | 0,84 | 0,55 | — |

La campaña 10 de K2 se publica en dos mitades (archivos `c101`/`c102`) pero MAST la numera como 10 (`curves.mast_sequence`); sin eso, 18 positivos se quedaban sin pareja.

## Lo que ve el modelo

Boosting sobre la vista global **con el tránsito tapado** (20 % central), AUC con validación cruzada de 5 particiones:

| | Kepler | K2 | TESS |
|---|---|---|---|
| PLANET vs NO_SIGNAL | 0,50 | 0,60 | 0,48 |
| FP vs NO_SIGNAL | 0,67 | 0,85 | 0,49 |

- **FP**: separación sobre todo astrofísica (binarias y estrellas variables: eclipse secundario y variabilidad fuera del tránsito; en K2 el ruido fuera de tránsito y secundario da 0,80). Es señal que el modelo debe aprender.
- **PLANET en K2 (0,60) y ruido en Kepler (0,59)**: sesgo de detección, los planetas se encuentran más en estrellas tranquilas. Se acepta y se declara.

## Limitaciones y reglas para la fase 2

- Los metadatos del proceso (`n_points`, `data_days`, `cadence_days`, `n_transits`, `frac_empty_*`, `n_files`) aún separan en K2 (AUC CV ~0,67). **Nunca se usan como rasgos.**
- Evaluar también por clase y por misión, para ver si NO_SIGNAL se apoya en el ruido.
- No se emparejó por ruido medido (CDPP): quitaría el sesgo de detección pero haría los negativos menos representativos (decisión 2026-11-01).
- Con 1 sector, el % de vistas válidas de PLANET en TESS baja de 96,3 % a 87,8 % (planetas de periodo largo con pocos tránsitos).

Resultados: `results/sample/shortcut_negatives.csv` (final) y `*_multisegment.csv`, `*_matched_v1.csv` (pasos intermedios).
