# Fase 3: transferencia entre telescopios (LOMO)

Protocolo en `03-protocolo-evaluacion.md`.
- Por cada misión excluida, la configuración se elige en los folds internos (train de A → val de B), con PR-AUC macro.
- Se reentrena con train + val de las otras dos, sin las estrellas que la excluida también observa, con 3 semillas en ensemble.
- Se evalúa una vez en la misión excluida.
- Decisión de Jairo (2026-11-04): la misión excluida se evalúa **sin su test** (train + val); el test queda para la evaluación final.

Script: `scripts/models/lomo.py` (GPU, bf16). Resultados: `results/models/lomo_dummy.csv`, `lomo_cnn.csv` y `lomo_cnn_inner.csv`.

## CNN solo curva (2026-11-04)

| Misión excluida | Entrena con | n ajuste / n evaluación | PR-AUC macro (dummy 0,333) | Macro-F1 (dummy 0,22) | F1 PLANET / FP / NO_SIGNAL |
|---|---|---|---|---|---|
| Kepler | K2 + TESS | 3 280 / 9 935 | **0,894** | **0,828** | 0,76 / 0,79 / 0,93 |
| K2 | Kepler + TESS | 12 566 / 650 | **0,927** | **0,894** | 0,87 / 0,88 / 0,94 |
| TESS | Kepler + K2 | 10 579 / 2 634 | **0,764** | **0,691** | 0,65 / 0,54 / 0,88 |
| **Media** | | | **0,862** | **0,804** | |

**Comparación con dentro de misión** (CNN conjunta con ensemble, val, `cnn_gpu_val.csv`). Los conjuntos de evaluación no son idénticos (LOMO usa train + val de la misión), así que es una comparación aproximada:

| | Kepler | K2 | TESS | Media |
|---|---|---|---|---|
| PR-AUC dentro de misión | 0,937 | 0,956 | 0,840 | 0,911 |
| PR-AUC LOMO | 0,894 | 0,927 | 0,764 | 0,862 |
| **Pérdida al cambiar de telescopio** | −0,04 | −0,03 | −0,08 | **−0,05** |
| Macro-F1 dentro / LOMO | 0,866 / 0,828 | 0,904 / 0,894 | 0,779 / 0,691 | 0,850 / 0,804 |

**Lectura.**
- **El modelo solo curva es portable.** Sin haber visto el telescopio, pierde de media 0,05 de PR-AUC y 0,05 de macro-F1, y en las tres misiones queda muy por encima del dummy. Kepler y K2 casi no pierden.
- **TESS es el telescopio difícil** (−0,08), sobre todo en FP frente a PLANET (F1 de FP 0,54). Tiene píxeles 5 veces mayores (más contaminación de estrellas vecinas), un solo sector por estrella (pocos tránsitos) y periodos cortos. Lo que se aprende en Kepler y K2 no cubre del todo cómo se ven sus falsos positivos.
- **Más datos de ajuste ayudan:** con K2 excluido (12 566 estrellas para ajustar) casi no hay pérdida; con Kepler excluido (solo 3 280) la pérdida es mayor, aunque pequeña.
- **La selección interna es inestable** cuando los folds internos son pequeños (val de K2 = 69 estrellas): con Kepler excluido, las PR-AUC internas van de 0,64 a 0,89 según la configuración. Con TESS excluido son estables (0,82–0,93).
- NO_SIGNAL se reconoce bien en todos los telescopios (F1 0,88–0,94): distinguir "hay señal" de "no hay" se transfiere casi perfecto. Lo que se degrada es separar planeta de falso positivo.

## Pruebas descartadas (2026-11-04)

- CNN + log(duración/periodo) + log(periodo) (`--ephem`): dentro de misión empata con la base (media de semillas 0,833 en las dos; ensemble 0,840 frente a 0,850). Mejora Kepler (0,866 → 0,877) y empeora K2. La duración/periodo ya está implícita en la vista global (anchura del tránsito = duty·201 bins). El LOMO con efeméride se canceló por decisión de Jairo.
- Gráfico de variables candidatas: `scripts/checks/feature_candidates.py`, `results/eda/feature_candidates.png`.

## Pendiente

- Ablación de variables que delatan la misión (#43): profundidad y SNR (`03-protocolo-evaluacion.md`).
- LOMO con centroide (el centroide ayudaba en Kepler y empeoraba TESS dentro de misión).
- Calibración y umbral en una misión sin etiquetas (cambio de prior).
- Evaluación final en test del modelo elegido (fase 4).
