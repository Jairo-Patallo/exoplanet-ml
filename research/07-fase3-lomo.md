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

## Diagnóstico de TESS: ¿plegado o modelo? (2026-11-04)

**Complejidad del modelo** (`lomo.py --model logreg`; PR-AUC, dentro de misión → LOMO):

| | Kepler | K2 | TESS |
|---|---|---|---|
| Regresión logística sobre vistas | 0,895 → 0,871 | 0,817 → 0,895 | 0,679 → **0,716** |
| CNN (ensemble) | 0,937 → 0,894 | 0,956 → 0,927 | 0,840 → **0,764** |

La CNN generaliza mejor a TESS que la logística (+0,05), así que la capacidad ayuda. Pero la CNN entrenada con TESS llega a 0,84, y sin TESS se queda en 0,764: TESS tiene patrones propios que no están en Kepler ni en K2. Un modelo más complejo solo no cierra esa brecha.

**Dónde falla la CNN en TESS** (`scripts/checks/tess_errors.py`, `results/models/tess_errors.csv`; fold con TESS excluido, 2 634 estrellas):
- Los **FP de TESS se aciertan solo el 42 %** (344 de 811): 279 se toman por PLANET y 188 por NO_SIGNAL.
- **Pocos tránsitos.** Con ≤ 8 tránsitos el acierto de FP es ~30 %; con más de 8, 52–61 %. La mediana de tránsitos en TESS es 8 (FP) y 5 (PLANET), frente a 37 y 17 en Kepler y 54 y 9 en K2. Con un solo sector, la mayoría de estrellas de TESS tienen pocos tránsitos y la forma del eclipse se ve peor.
- **Poca señal.** Los positivos con SNR ≤ 7 se aciertan mal (PLANET 38 %, FP 17 %). Son el 20 % de los positivos de TESS.
- **Cambio de población: planetas de periodo corto.** En TESS el 14 % de los planetas tienen P < 2 d, frente al 6–7 % en Kepler y K2, donde los periodos cortos son sobre todo FP (33 % y 59 % de los FP). El modelo aprendió "periodo corto → binaria". En TESS falla en la mitad de esos planetas (40 de 82, 21 tomados por FP).

**Conclusión.** La pérdida en TESS tiene dos causas de datos, no de modelo:
1. **Poca información por estrella** (un sector): un plegado con más sectores daría más tránsitos y más señal.
2. **Población distinta** (más planetas cortos): solo se corrige viendo ejemplos de TESS, con pocas etiquetas o reponderando.

Más sectores reabre el atajo de la fase 1 (los positivos tenían más sectores que los negativos). Si se hace, los negativos deben tener el mismo número de sectores que su pareja.

## Curva de aprendizaje: ¿cuántas etiquetas de TESS hacen falta? (2026-11-04)

`scripts/models/learning_curve.py`.
- **Entrenamiento:** Kepler + K2 (train + val) más una fracción del train de TESS, sorteada por clase.
- **Evaluación:** siempre en el val de TESS (297 estrellas).
- **Configuración fija:** la elegida por LOMO para TESS (base 16, dropout 0,5); ensemble de 3 semillas.
- Resultados: `results/models/learning_curve_tess.csv`.

| % del train de TESS | Estrellas TESS | PR-AUC macro | Macro-F1 | F1 FP |
|---|---|---|---|---|
| 0 (LOMO) | 0 | 0,771 | 0,694 | 0,51 |
| 2 % | 46 | 0,780 | 0,717 | 0,55 |
| 5 % | 117 | 0,775 | 0,715 | 0,55 |
| 10 % | 234 | 0,778 | 0,724 | 0,56 |
| 25 % | 583 | 0,792 | 0,712 | 0,56 |
| 50 % | 1 167 | 0,786 | 0,735 | 0,59 |
| 100 % | 2 332 | **0,836** | **0,785** | **0,69** |

Referencia: la CNN conjunta (entrenada con las tres misiones) da 0,840 / 0,779 en el mismo val.

**Lectura.**
- **Unas pocas etiquetas de TESS ayudan poco:** con 46–234 estrellas, el macro-F1 sube +0,02 a +0,03 y la PR-AUC casi nada. Hasta no tener todo el train de TESS no se recupera el nivel dentro de misión.
- La razón probable es el **desequilibrio**: 234 estrellas de TESS frente a ~10 000 de Kepler y K2 pesan muy poco en el entrenamiento, y el modelo apenas las tiene en cuenta.
- Formas estándar de aprovechar mejor pocas etiquetas, sin probar todavía: **dar más peso** a las estrellas de TESS en la pérdida, o **ajuste fino** (preentrenar con Kepler + K2 y reentrenar solo con las pocas de TESS).
- El acierto en planetas de P < 2 d en val (11 estrellas) oscila entre 64 % y 82 % sin tendencia: la muestra es demasiado pequeña para concluir.
