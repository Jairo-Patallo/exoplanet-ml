# Predicciones congeladas v1 en los TOIs abiertos (fase 7, #85 y #87; 2026-11-07)

Primer paso de la validación prospectiva: predecir hoy los TOIs en PC y compararlos dentro de meses con las disposiciones nuevas del TFOPWG (#92). Las predicciones no se pueden cambiar después: si se mejora el modelo, se congela una v2 aparte, con su fecha.

## Qué se congeló

- **TOIs:** los 4 813 en PC de ExoFOP (`data/labels/exofop_toi.csv`, descargado el 2026-11-05). 76 no tienen periodo (un tránsito) y se omiten. Se procesan 4 737; **3 536 tienen vista válida**. 756 no tienen curva de 2 min de SPOC (solo imágenes completas, QLP) y 444 no son válidas (pocos tránsitos o huecos).
- **Modelo:** el final de la fase 4 (CNN solo curva sin profundidad, 5 semillas) con la temperatura de TESS (T = 1,07). Decisión 13: modelo base, no el de negativos difíciles.
- **Columnas** (`results/toi/predictions_v1.csv`):
  - las 3 probabilidades, la clase y P(planeta | señal) = p_P/(p_P + p_FP);
  - la deriva de la efeméride y la marca `efemeride_incierta`;
  - la puntuación de ExoMiner++ (Zenodo 10.5281/zenodo.15466293, CC-BY-4.0; misma TIC y periodo a ±1 %; 2 007 TOIs con puntuación);
  - las prioridades del TFOP (Master, SG1A, SG1B, SG2).
- **Manifiesto** (`results/toi/manifest_v1.json`): SHA-256 de los 5 pesos, la temperatura, las vistas (concatenadas), el qc, ExoFOP, ExoMiner++ y las predicciones. Commit `c0839cc`, árbol limpio.

## Hallazgo: la deriva de la efeméride

El error de una efeméride crece con las órbitas desde la época: σ_t = sqrt(σ_t0² + (n·σ_P)²).

- **Con la regla del entrenamiento** (el sector con más tránsitos), el 8 % de los TOIs tenía σ_t > 1 duración en el sector usado, y el 54 % de ellos salía NO_SIGNAL. El plegado no encontraba el tránsito. Histórico en `results/toi/history/`.
- **Decisión de Jairo (opción 1):** entre los sectores con ≥ 2 tránsitos esperados, el de menor deriva; si varios tienen deriva ≤ 0,1, el de más tránsitos, como en el entrenamiento. Se marca `efemeride_incierta` si la deriva supera 0,25.

| Deriva del sector usado (σ_t / duración) | TOIs | NO_SIGNAL | FP |
|---|---|---|---|
| ≤ 0,1 | 3 084 | 4,1 % | 14,9 % |
| 0,1–0,25 | 126 | 22,2 % | 27,0 % |
| 0,25–0,5 | 43 | 14,0 % | 23,3 % |
| 0,5–1 | 61 | 23,0 % | 24,6 % |
| > 1 | 222 | 39,2 % | 41,9 % |

- 326 TOIs (9,2 %) quedan con efeméride incierta. Ningún sector de 2 min les da menos de 0,25: suelen ser TOIs con periodo poco preciso o con los sectores de 2 min lejos de la época. Hay que excluirlos de la validación prospectiva o tratarlos aparte.
- **Aviso:** entre 0,1 y 0,25 la tasa de NO_SIGNAL ya sube al 22 % (n = 126). La marca a 0,25 se fijó antes de ver esta tabla y no se cambia en la v1. Al analizar (#92), conviene informar también con el corte en 0,1.
- La v1 elige el sector con información que el entrenamiento no usaba (la deriva). Es la única diferencia con la cadena de entrenamiento.

## Primera comparación con ExoMiner++ (descriptiva)

Entre los 1 947 TOIs con deriva ≤ 0,25 y puntuación de ExoMiner++:
- correlación de Spearman entre nuestro P(planeta | señal) y ExoMiner++: 0,37;
- coinciden en el lado del umbral 0,5 en el 76 % de los casos.

No es una medida de acierto, porque no hay etiquetas. El análisis caso a caso es la tarea #102.

## Qué falta

- Publicar el manifiesto con un sello de tiempo externo (release de GitHub o Zenodo). Pendiente de permiso de Jairo.
- #92: repetir el cruce con ExoFOP cada pocos meses. Las disposiciones del TFOP tardan en llegar, así que conviene medir primero cuántos PC cambian al mes.
