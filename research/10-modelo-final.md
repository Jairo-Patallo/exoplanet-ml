# Modelo final y evaluación única en test (fase 4)

**Esta receta se fija y se commitea antes de mirar el test.** Nada de lo que hay aquí se cambia después de ver el resultado. Si algo se cambiara, la cifra de test dejaría de ser honesta y habría que decirlo.

## Receta (decisiones de Jairo, 2026-11-05)

| Elemento | Elección | Motivo |
|---|---|---|
| Alcance | Clasificador de señales ya detectadas: curva + periodo y época dados | #53: con BLS el macro-F1 cae de ~0,86 a ~0,50 (`research/09`) |
| Entrada | Vistas global (201 bins) y local (81 bins) plegadas con la efeméride del catálogo, normalizadas por estrella; **sin log10 de la profundidad** | #43: aportaba +0,003 y es la variable que más delata la misión |
| Datos | Kepler + K2 + TESS juntos; TESS con 1 sector | FFI con todos los sectores queda como trabajo futuro (+0,034 con 2 sectores, `research/07`) |
| Modelo | CNN de dos ramas tipo AstroNet | Mejor modelo de la fase 2 |
| Configuración | **base 16, dropout 0,5**, elegida en val sin profundidad (`results/models/final_select.csv`: macro-F1 medio 0,839; Kepler 0,860, K2 0,877, TESS 0,779; las otras tres, 0,812–0,834) | Misma rejilla y criterio que la fase 2 (macro-F1 medio de misiones en val, semilla 0) |
| Entrenamiento | train + val, 5 semillas, ensemble por media de probabilidades | Práctica estándar; AstroNet-K2 y ExoMiner++ usan ensembles |
| Decisión | argmax | #37: corregir el prior empeora el macro-F1 |

`scripts/models/final_model.py {select, train, test}`; pesos en `data/models/final_seed{0..4}.pt`.

## Cómo se evalúa (una vez)

1. **Por telescopio:** el modelo final en el test de Kepler, K2 y TESS. Macro-F1 (principal) y PR-AUC macro, con IC 95 % bootstrap; matrices de confusión. `results/models/final_test.csv`.
2. **Portabilidad (LOMO en test):** la misma receta con el protocolo LOMO (`lomo.py --model cnn --amp --no-depth --seeds 5 --eval-splits test`). Por cada telescopio excluido: selección de la configuración en los folds internos de los otros dos, reentrenamiento con su train + val sin las estrellas compartidas y evaluación en el **test** del excluido. `results/models/lomo_cnn_nodepth_test.csv`.

Las métricas de val de las fases 2 y 3 son las de referencia para comparar. Se espera que el test quede algo por debajo, porque en val se eligieron configuraciones y variantes.

## Resultados

Ejecutado el 2026-11-05, una sola vez, sin tocar la receta después. Épocas de parada de las 5 semillas: 32, 30, 11, 10, 24.

### 1. Por telescopio (modelo final, ensemble de 5 semillas)

| Misión | n test | Macro-F1 [IC 95 %] | PR-AUC macro [IC 95 %] | Val (selección, 1 semilla) |
|---|---|---|---|---|
| Kepler | 1102 | **0,886** [0,867–0,906] | 0,947 [0,928–0,964] | 0,860 |
| K2 | 81 | **0,928** [0,858–0,983] | 0,972 [0,928–1,000] | 0,877 |
| TESS | 292 | **0,720** [0,661–0,772] | 0,816 [0,762–0,871] | 0,779 |
| Media | | 0,845 [0,814–0,873] | 0,912 [0,888–0,935] | 0,839 |

Matrices de confusión (filas real, columnas predicha; PLANET / FP / NO_SIGNAL):

| | Kepler | K2 | TESS |
|---|---|---|---|
| PLANET | 179 / 7 / 0 | 17 / 2 / 1 | 47 / 8 / 3 |
| FALSE_POSITIVE | 49 / 287 / 39 | 0 / 18 / 1 | 34 / 41 / 14 |
| NO_SIGNAL | 5 / 7 / 529 | 0 / 1 / 41 | 1 / 6 / 138 |

- Kepler y K2 quedan por encima de val. Es lo esperable: el modelo final entrena con train + val y es un ensemble de 5 semillas, y en val había una sola. K2 tiene solo 81 estrellas en test; su IC es amplio.
- TESS baja 0,06 respecto a val (0,779 → 0,720), dentro del IC. El fallo dominante es el mismo que en val: falsos positivos que el modelo llama planeta (34 de 89, 38 %). F1 de FALSE_POSITIVE en TESS: 0,57.
- NO_SIGNAL se separa bien en las tres misiones (F1 0,92–0,96): el problema difícil es planeta frente a falso positivo.

### 2. Portabilidad (LOMO en test, sin profundidad, 5 semillas)

| Excluida | n | Config. elegida | PR-AUC macro | Macro-F1 | LOMO en val (fase 3, sin profundidad) |
|---|---|---|---|---|---|
| Kepler | 1102 | base 16, dropout 0,3 | **0,906** | 0,855 | 0,891 |
| K2 | 81 | base 8, dropout 0,3 | **0,961** | 0,924 | 0,923 |
| TESS | 292 | base 8, dropout 0,5 | **0,761** | 0,667 | 0,761 |

- Las cifras de test coinciden con las de val: lo que se midió en la fase 3 no estaba inflado por la selección.
- Coste de no haber visto la misión (PR-AUC, por telescopio frente a LOMO): Kepler 0,947 → 0,906, K2 0,972 → 0,961, TESS 0,816 → 0,761. TESS sigue siendo la misión que peor se transfiere y la que peor se aprende aun viéndola.

### Conclusión

El modelo solo curva clasifica Kepler y K2 con macro-F1 ≈ 0,89–0,93 y TESS con 0,72. Llevado a una misión que no ha visto pierde 0,01–0,06 de PR-AUC. El cuello de botella es separar planetas de falsos positivos en TESS (1 sector, 2 min, señales más débiles), coherente con la fase 3: la única mejora que funcionó allí fue más datos de TESS (2 sectores, +0,034).

Salidas: `results/models/final_test.csv`, `final_test_confusion.csv`, `final_test_pred.csv`, `lomo_cnn_nodepth_test.csv`, `lomo_cnn_nodepth_test_inner.csv`, `lomo_cnn_nodepth_test_pred.csv`.
