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

(Se rellenan después de ejecutar la evaluación.)
