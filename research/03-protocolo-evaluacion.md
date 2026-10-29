# Protocolo de evaluación (Fase 3, decisión de la tarea #24)

**Objetivo:** que el modelo generalice a curvas de instrumentos que no ha visto, no solo a nuevos datos de Kepler, K2 y TESS.
Por eso se evalúa **dejando una misión fuera** (*leave-one-mission-out*, LOMO): cada misión se evalúa con un modelo
entrenado en las otras dos. Al ser la misión constante dentro de cada fold de test, el atajo de la misión desaparece solo.

## Datos (recuentos con `python -m exoplanet.data`)

| Misión | train PLANET / FP | % FP | val PLANET / FP |
|---|---|---|---|
| Kepler | 2 859 / 3 712 | 56 % | 358 / 464 |
| K2 | 996 / 192 | 16 % | 124 / 24 |
| TESS | 3 977 / 845 | 18 % | 497 / 106 |

## Decisiones

| Decisión | Por qué |
|---|---|
| LOMO **solo para vetting** (PLANET vs FP) | NO_SIGNAL solo existe en Kepler: al dejar Kepler fuera no se entrena esa clase, y en K2/TESS no se puede evaluar |
| La tarea de **3 clases se queda dentro de Kepler** (splits originales) | Es la única misión con las tres clases |
| Fold "misión M fuera": **entrenar** en *train* de las otras dos; **ajustar** hiperparámetros en *val* de las otras dos; **puntuar en desarrollo** en *val* de M | El scaler y todo el preprocesado se ajustan solo con el *train* del fold. Así el *test* no se toca |
| **Test de M una sola vez**, al final (Fase 5) | Elegir modelos mirando la misión excluida sobreajusta a estas tres misiones |
| Resultado **por fold**, nunca solo la media | Folds muy desiguales: K2 tiene 24 FP en val (mucha varianza) |
| Métricas: **macro-F1 + PR-AUC y ROC-AUC** | El % de FP cambia de 16 % a 56 % entre misiones (*label shift*): el macro-F1 cae por el umbral aunque el modelo separe bien. Las AUC no dependen del umbral |
| **Pesos de clase** ajustados en el train de cada fold | Mismo motivo: el train sin Kepler es un 83 % PLANET |
| **Referencia in-domain**: entrenar con las 3 misiones y evaluar por misión | La diferencia con LOMO es la **brecha de generalización**, que es la cifra que responde al objetivo |
| El listón de 0.687 **deja de valer**; se recalculan dummy y reglas físicas con LOMO | 0.687 salía de usar la misión, que en LOMO no aporta nada |

## Implementación

- `exoplanet.data.lomo_folds()` genera los 3 folds `(misión excluida, fit, tune, dev)` y `load_vetting()` filtra
  PLANET/FP con vistas plegadas. Tests en `tests/test_data.py` (`python -m pytest tests`): cada misión sale una vez,
  sin estrellas compartidas entre fit/tune/dev y sin leer `test.parquet`.
- `exoplanet.evaluation.vetting_metrics()`: macro-F1, PR-AUC y ROC-AUC (positivo = FP).
- La referencia in-domain entrena con el train de las 3 misiones y se evalúa en el val de cada una. Para los modelos
  con hiperparámetros, el ajuste in-domain se hará con CV dentro de train, para no ajustar y evaluar en el mismo val.

## Escalón 1: baseline dummy (`python scripts/baseline.py` → `results/baseline/dummy.csv`)

Clase mayoritaria del fit; como probabilidad, la proporción de clases (ROC-AUC = 0.5 y PR-AUC = % de FP del dev).

| Protocolo | Misión dev | n dev | % FP | Predice | macro-F1 | PR-AUC |
|---|---|---|---|---|---|---|
| LOMO | Kepler | 822 | 56 % | PLANET | 0.303 | 0.564 |
| LOMO | K2 | 119 | 16 % | PLANET | 0.457 | 0.160 |
| LOMO | TESS | 600 | 18 % | **FP** | **0.149** | 0.175 |
| in-domain | Kepler | 822 | 56 % | PLANET | 0.303 | 0.564 |
| in-domain | K2 | 119 | 16 % | PLANET | 0.457 | 0.160 |
| in-domain | TESS | 600 | 18 % | PLANET | 0.452 | 0.175 |

**Lectura:** el fold sin TESS entrena con Kepler + K2, donde los FP son mayoría (51 %), y predice FP en TESS, donde son
el 18 %. Es el *label shift* en estado puro: la clase mayoritaria cambia según qué misiones se usen para entrenar.
Cualquier modelo que no corrija las proporciones de clase (pesos o ajuste del umbral) arrastrará este problema.
**Nuevo listón por fold** = estas cifras; para comparar modelos, PR-AUC es la más estable frente al cambio de proporciones.

## Limitaciones

- Tres misiones son solo tres dominios, todos de fotometría espacial y con el mismo preprocesado (curvas plegadas en fase
  y con bins). Generalizar entre ellas **no garantiza** funcionar con curvas desde tierra ni con otro preprocesado:
  cualquier curva nueva debe pasar por el mismo pipeline. Esto debe figurar como alcance del modelo (model card).
- Con 3 folds la estimación de la generalización tiene mucha varianza; el fold de K2 es el menos fiable.
