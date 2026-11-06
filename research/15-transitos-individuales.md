# Tránsitos individuales: ¿información que la mediana plegada borra? (fase 7, 2026-11-06)

Paso 1 de la propuesta P1 de la auditoría (`docs/auditoria-y-propuestas-2026-11-06.pdf`): antes de construir un Transformer sobre el conjunto de tránsitos, medir si los tránsitos por separado contienen información sobre planeta frente a FP que la CNN no usa. Criterio fijado en Notion antes de mirar.

## Método

`exoplanet.features.transit_features` (3 tests en `tests/test_features.py`). Mismos archivos y mismo aplanado que la vista de catálogo (1 sector en TESS).
- A cada tránsito se le ajusta la plantilla plegada de los **demás** tránsitos (forma fija, amplitud libre, ±1,5 duraciones). Así una forma en V o la cadencia de 30 min no crean diferencias falsas. La primera versión medía la media de los puntos centrales y en Kepler daba z_scatter de ~20 en binarias profundas solo por el muestreo de la V.
- **Ruido rojo:** la misma plantilla se ajusta en "tránsitos falsos" (fases 0,15–0,85, lejos del tránsito y del secundario). β = dispersión robusta de sus amplitudes en unidades de su error. Es la idea del block bootstrap de #9.
- Rasgos, todos ~N(0, 1) en un planeta sea cual sea el número de tránsitos, para no delatar ese número (metadato del proceso):
  - |z| par/impar;
  - z_scatter: χ² de las amplitudes pasado a z;
  - z_max: el tránsito más desviado.
- `scripts/data/build_transit_feats.py` → `results/sample/transit_feats.csv`: 6 650 PLANET/FP de train + val, 0 errores, β en el 91 %.
- `scripts/checks/transit_feats_check.py` → `results/models/transit_feats_check.csv`.
  - Puntuación de la CNN: logit de p_P/(p_P + p_FP).
  - Predicciones fuera de muestra: **LOMO** (train + val de la misión que el modelo no vio) y **val** con la receta final.
  - Ganancia: logística fuera de pliegue (5 folds), [CNN] frente a [CNN + rasgos], con IC bootstrap pareado.

## Calibración de los rasgos (ruido bien medido: mediana |z| 0,67, P(|z| > 3) 0,003)

| | n | Tránsitos (mediana) | β | Mediana \|z\| par/impar | P(\|z\| > 3) | Mediana z_scatter | P(z_scatter > 3) |
|---|---|---|---|---|---|---|---|
| Kepler PLANET | 1 650 | 16 | 1,10 | 0,93 | 0,052 | 1,07 | 0,19 |
| Kepler FP | 3 346 | 50 | 1,22 | 1,47 | 0,242 | 4,46 | 0,50 |
| K2 PLANET | 168 | 8,5 | 1,61 | 0,80 | 0,077 | 0,50 | 0,16 |
| K2 FP | 161 | 39 | 2,37 | 1,67 | 0,323 | 5,44 | 0,47 |
| TESS PLANET | 514 | 6 | 1,02 | 0,80 | 0,049 | 0,59 | 0,09 |
| TESS FP | 811 | 9 | 1,05 | 0,89 | 0,079 | 0,62 | 0,14 |

- **El test par/impar queda mucho mejor calibrado** que el de la vista global: en planetas, P(|z| > 3) es 0,05 frente a 0,145 en `research/02`. No llega a 0,003. El resto puede ser física real, como TTV (desplazan el tránsito respecto a la plantilla y bajan su amplitud) o cambios de contaminación entre trimestres en Kepler. Sirve para #9 y para el informe de la fase 8.
- En TESS, los FP casi no se distinguen de los planetas en estos rasgos (P(|z| > 3) 0,079 frente a 0,049). Con 6–9 tránsitos de un sector, cada tránsito tiene poca señal.

## Resultado (ROC-AUC planeta frente a FP)

| Predicciones | Misión | n | CNN | Solo rasgos | Ganancia CNN + rasgos [IC 95 %] | Rasgos donde la CNN dice PLANET |
|---|---|---|---|---|---|---|
| LOMO | TESS | 1 325 | 0,780 | 0,545 | **+0,000** [−0,005, +0,005] | **0,515** [0,472, 0,558] (n 702) |
| LOMO | Kepler | 4 996 | 0,897 | 0,676 | +0,001 [−0,000, +0,002] | 0,615 [0,589, 0,645] |
| LOMO | K2 | 329 | 0,955 | 0,671 | +0,003 [−0,005, +0,010] | (10 FP) |
| val | TESS | 148 | 0,856 | 0,518 | −0,023 [−0,049, +0,001] | 0,546 [0,384, 0,710] |
| val | Kepler | 560 | 0,953 | 0,673 | +0,000 [−0,003, +0,003] | 0,534 [0,423, 0,648] |

TESS por número de tránsitos (LOMO): ≤ 8 → rasgos 0,535; > 8 → 0,563 [0,506, 0,614].

**Criterio: no se cumple.** En TESS, la ganancia es +0,000 [−0,005, +0,005] y, entre las estrellas que la CNN llama planeta, los rasgos dan un AUC de 0,515, con un límite inferior por debajo de 0,5.

## Lectura

- En Kepler y K2, la dispersión entre tránsitos sí separa por sí sola (AUC ~0,67). Pero la CNN ya lo sabe: la ganancia es nula. Los FP con tránsitos desiguales son binarias que la forma plegada ya delata.
- En TESS, la información por tránsito está casi vacía: con un sector, los tránsitos individuales tienen poca SNR. Las binarias de TESS que pasan por planeta no son del tipo par/impar ni de profundidad variable.
- Que estos rasgos no ganen no demuestra que no haya ninguna información por tránsito. Pero los tres mecanismos principales (par/impar, dispersión, un tránsito anómalo) no aportan nada en TESS. La ganancia esperada de un modelo por tránsito (pasos 2–3 de la P1) es baja, y **no se recomienda seguir con la P1 para TESS**.
- La siguiente palanca con más sentido para las binarias de TESS es la forma del secundario y la forma local a mayor resolución (P2 de la auditoría). Decisión de Jairo.

## Cambio colateral

`scripts/data/build_bls_views.star_files` no encontraba los FITS de la campaña 10 de K2 (carpeta `c10`, archivo `c102`). Afectaba a 44 de 796 estrellas de K2; las 16 estrellas de K2 marcadas «sin archivos locales» en `qc_bls.csv` vienen de ahí. Corregido: 0 de 731 sin archivos. Las vistas BLS, las series y los picos de BLS ya generados no se han rehecho.
