# BLS: efeméride a partir de la curva (#53, fase 4)

Decisiones de Jairo (2026-11-05): periodos de 0,5 d a la mitad de la duración de los datos; NO_SIGNAL plegados con su pico de BLS; medir primero con el modelo actual en val y después decidir si reentrenar.

## Implementación (`src/exoplanet/bls.py`)

- **Aplanado a ciegas** (ventana de 1,5 d, sin máscara de tránsito) y recorte de picos hacia arriba.
- **BLS de astropy** por bandas de periodo (0,5–2, 2–10 y > 10 d), cada una con duraciones de caja acordes (de 1 h a 12 h).
- **Rejilla gruesa** con resolución de 60 días de datos y **refinado fino** de los 5 mejores picos con toda la base de tiempo. Sin esto, dos trimestres de Kepler separados por años daban millones de periodos (390 s por estrella).
- **Periodograma aplanado:** se le resta su tendencia, calculada por tramos de log(P). Sin ello, BLS elegía siempre el periodo mínimo (con periodos cortos caben más cajas). Una primera versión con una ventana de puntos fija elegía múltiplos ×7 o ×9 en K2, porque el pico real era más ancho que la ventana.
- **Significancia:** SDE = pico del periodograma aplanado en unidades de su dispersión robusta.
- Tras BLS, `curves.prepare` vuelve a aplanar enmascarando el tránsito encontrado y se pliega con `fold_views`, como con el catálogo.
- **Coste:** ~3,5 s por estrella en TESS, ~7 s en K2 y ~15 s en Kepler. Val (1 487 estrellas) tardó ~1 h con 4 procesos.

`scripts/data/build_bls_views.py` → `data/sample/views_bls/`, `results/sample/qc_bls.csv`. 1 error (sin archivos locales) y 84 vistas no válidas (55 de ellas NO_SIGNAL).

## Recuperación del periodo en val

| | Igual (±1 %) | Alias (½, 2, ⅓) | Otro |
|---|---|---|---|
| Kepler PLANET (176) | 68 % | 1 % | 31 % |
| Kepler FP (384) | 65 % | 10 % | 25 % |
| K2 PLANET (16) | 31 % | 0 % | 69 % |
| K2 FP (19) | 95 % | 5 % | 0 % |
| TESS PLANET (59) | 69 % | 7 % | 24 % |
| TESS FP (89) | 66 % | 13 % | 20 % |

Los FP tienen más alias de ½ (binarias con dos eclipses parecidos). Los planetas de K2 salen mal, probablemente por los artefactos periódicos de K2 (correcciones de orientación cada ~6 h).

**La SDE separa señal de ruido** (ROC-AUC señal frente a NO_SIGNAL: Kepler 0,858, K2 0,865, TESS 0,821; mediana NO_SIGNAL 8–13 frente a 17–890 en las señales). Es lo que usan los pipelines como umbral de detección.

## El mismo modelo con efeméride de catálogo o de BLS

`scripts/models/eval_bls.py` (CNN conjunta de la fase 2, semilla 0; val con vista válida en las dos versiones). `results/models/eval_bls.csv` y `eval_bls_por_categoria.csv`.

| Misión | n | PR-AUC catálogo → BLS | Macro-F1 catálogo → BLS | F1 NO_SIGNAL catálogo → BLS |
|---|---|---|---|---|
| Kepler | 1 052 | 0,926 → 0,669 | 0,862 → **0,498** | 0,939 → 0,390 |
| K2 | 54 | 0,939 → 0,608 | 0,882 → **0,448** | 0,889 → 0,160 |
| TESS | 297 | 0,838 → 0,556 | 0,766 → **0,445** | 0,899 → 0,433 |

**Acierto por categoría (Kepler):**

| | Catálogo | BLS |
|---|---|---|
| PLANET con el periodo correcto (119) | 92 % | **96 %** |
| FP con el periodo correcto (242) | 89 % | 74 % |
| **NO_SIGNAL (517)** | **97 %** | **26 %** |
| PLANET con otro periodo (47) | 77 % | 47 % |

**Lectura.**
- **La caída viene sobre todo de NO_SIGNAL**: del 95–97 % de acierto al 10–29 %. En las fases 1–3 los NO_SIGNAL se plegaban con una efeméride al azar, así que su vista era plana y fácil. Plegados con el mejor pico de ruido que encuentra BLS, parecen un tránsito débil, y el modelo, que nunca vio algo así, los llama PLANET o FP. **El F1 de NO_SIGNAL de las fases 2 y 3 (0,88–0,94) era optimista para una curva real sin señal**: vale para clasificar señales ya detectadas, no para decidir si hay señal.
- **Cuando BLS acierta el periodo, el modelo funciona igual o mejor** con los planetas (92 % → 96 % en Kepler). Con los FP baja (89 % → 74 %); falta investigarlo (duración o época estimadas por BLS).
- Cuando BLS da otro periodo, se pierde la mitad de los planetas: es el coste de la detección.

## Decisión de Jairo (2026-11-05): (A), clasificador de señales ya detectadas

El producto es un clasificador de *vetting*: la entrada es la curva más el periodo y la época de una señal ya detectada (catálogo, pipeline del telescopio o el usuario). Se queda el modelo actual. NO_SIGNAL significa "la efeméride dada no corresponde a un tránsito real", no "la curva no tiene señal". BLS queda como herramienta opcional (por ejemplo, para proponer una efeméride en la web, avisando de que acierta ~2 de cada 3 veces) y como medida documentada de lo que costaría ir de principio a fin.

Opciones que se consideraron:

- **(A) Clasificador de señales ya detectadas (*vetting*)**, como AstroNet y ExoMiner. Entrada: curva + periodo y época. Se queda el modelo actual y se documenta que NO_SIGNAL significa "la efeméride dada no corresponde a un tránsito", no "la curva no tiene señal".
- **(B) De principio a fin.** Se reentrena con vistas de BLS para todas las estrellas de train: BLS en ~11 700 estrellas, ~10 h de cálculo (sobre todo Kepler). Se añade la SDE como umbral o entrada, y se reevalúa. Es lo que haría falta para una web que acepte cualquier curva.
- **(C) Híbrido.** Modelo (A) para clasificar, más un umbral de SDE delante como detector ("¿hay señal?"). Barato, pero el umbral solo separa con ROC-AUC ~0,82–0,86.
