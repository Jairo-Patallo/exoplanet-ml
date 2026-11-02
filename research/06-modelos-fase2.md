# Modelos de la fase 2: 3 clases dentro de cada misión

Métricas y listón: `03-protocolo-evaluacion.md` (tarea #51). Todo en **val** de la muestra A (solo vistas válidas); el test no se ha tocado.

## Regresión logística (2026-11-02)

Código: `scripts/models/sklearn_models.py --models logreg` (`PYTHONPATH="src;legacy"`). Resultados: `results/models/logreg_*.csv`. Tarda ~20 s en CPU.

**Diseño (decisión de Jairo, 2026-11-02).** Un modelo por misión; `StandardScaler` ajustado solo en train; `class_weight="balanced"`; C en una rejilla de 1e-4 a 100, elegido en val por macro-F1. Dos variantes de entrada:

- **A · Variables físicas** (5): profundidad y SNR de la vista local, con signo y en log con signo; `ingress_frac`; `odd_even_z`; `secondary_z`. Cálculo: `legacy/features.py`, con `mask_no_dip=False`. Por defecto esa función pone a NaN todas las variables si no hay caída (δ ≤ 0). Eso le pasa a ~50 % de los NO_SIGNAL y a casi ningún positivo, así que imputarlas habría borrado la señal "no hay tránsito". `ingress_frac` sin caída y `odd_even_z` sin par/impar válido se imputan con la mediana de train. No se añade indicador de "faltaba", porque delataría el número de tránsitos.
- **B · Vistas** local (81) y global (201), normalizadas por estrella al estilo AstroNet. Se resta la mediana y se divide por la profundidad (mediana − mínimo de la local), así que la caída vale ~−1. Se añade `log10(profundidad)`, para no perder la escala. Par/impar fuera.

**Resultados en val** (IC 95 % bootstrap por estrella; listón dummy = 0,22):

| Macro-F1 | Kepler | K2 | TESS | Media |
|---|---|---|---|---|
| A · físicas | 0,70 [0,62–0,77] | 0,79 [0,69–0,88] | 0,49 [0,40–0,59] | 0,66 [0,61–0,71] |
| **B · vistas** | **0,84** [0,78–0,90] | **0,83** [0,73–0,91] | **0,72** [0,62–0,83] | **0,80** [0,75–0,85] |

PR-AUC macro: A 0,73, B 0,82. F1 por clase de B (media): PLANET 0,77, FP 0,74, NO_SIGNAL 0,88.

**Lectura.**
- Las dos variantes superan con mucho al dummy. B supera a A en las tres misiones: la forma completa de la curva contiene más información que 5 resúmenes.
- **Error principal: FP clasificado como PLANET.** Ocurre en Kepler (A: 26 de 54 FP; B: 10 de 54) y en TESS (B: 6 de 23). Es la confusión físicamente difícil: binarias de fondo diluidas que parecen planetas.
- A en TESS es casi inútil para FP (F1 0,12). La C elegida es muy pequeña, lo que significa que el modelo apenas encuentra señal. Las variables par/impar y secundario están peor medidas con un solo sector (24 % de par/impar no válidas en PLANET de TESS).
- Coeficientes de A (`logreg_coef_A.csv`), con sentido físico: un SNR negativo empuja a NO_SIGNAL, y `secondary_z` y `odd_even_z` empujan a FP, igual que el vetting clásico.
- B prefiere C muy pequeñas (1e-3 a 3e-3), es decir, regularización fuerte: con 282 entradas y ~600–1350 estrellas, el modelo sobreajusta enseguida.

**Limitaciones.**
- C se elige y se informa en el mismo val, así que las cifras son algo optimistas. La comparación justa con otros modelos usa el mismo procedimiento; el número honesto saldrá del test al final.
- Val pequeño: los IC de K2 y TESS tienen ~0,2 de ancho. A y B solo se distinguen claramente en la media.
- La variante A depende de `legacy/features.py`; si se usa en la versión final, pasarla a `src/`.

## kNN (2026-11-02)

Mismo script (`--models knn`), mismas entradas, mismo escalado (ajustado en train) y mismo protocolo. Se eligen en val por macro-F1 k (de 1 a 51) y los pesos (uniformes o por distancia), con distancia euclídea. Sin pesos de clase, porque kNN no los admite. Resultados: `results/models/knn_*.csv`.

| Macro-F1 (val) | Kepler | K2 | TESS | Media |
|---|---|---|---|---|
| A · físicas | 0,79 [0,72–0,85] | 0,86 [0,77–0,94] | 0,68 [0,55–0,80] | **0,78** [0,72–0,83] |
| B · vistas | 0,44 | 0,61 | 0,42 | 0,49 [0,43–0,54] |

**A: kNN supera a la regresión logística** (0,78 frente a 0,66), sobre todo en TESS (F1 de FP 0,64 frente a 0,12). Las 5 variables físicas separan las clases de forma no lineal, por ejemplo mediante combinaciones de secundario y profundidad, y un modelo lineal no lo capta. Es una pista a favor de modelos no lineales (boosting, MLP).

**B: kNN falla, y la causa es la maldición de la dimensionalidad, no un error.** En un NO_SIGNAL no hay caída, así que dividir por la "profundidad" deja una vista de ruido puro de amplitud ~1 en los 283 valores. Su norma tras escalar es ~15,7, frente a ~7 de PLANET y FP, que son curvas suaves. En tantas dimensiones dos vectores de ruido están muy lejos entre sí. Por eso el vecino más cercano de un NO_SIGNAL es otro NO_SIGNAL solo el 10 % de las veces (Kepler, 5 vecinos), y la búsqueda acaba eligiendo k = 1–3. La regresión logística no sufre esto, porque pondera cada entrada en vez de medir distancias.

**Arreglo con PCA (decisión de Jairo, 2026-11-02; `--models knn_pca`).** Se aplica una PCA, ajustada en train después del escalado, antes del kNN. El número de componentes (5–80) se elige en val junto con k y los pesos.

| Macro-F1 (val) | Kepler | K2 | TESS | Media |
|---|---|---|---|---|
| B · PCA + kNN | 0,83 [0,75–0,89] | 0,83 [0,73–0,92] | 0,72 [0,60–0,84] | **0,79** [0,74–0,85] |

- La PCA arregla el problema: 0,49 → 0,79. Con 5–20 componentes todas las misiones dan 0,72–0,83; con 80 vuelve a caer (0,54–0,67) porque se recupera el ruido. La forma útil del tránsito cabe en unas 20 direcciones.
- K2 elige k = 1 (0,83 frente a 0,80 de la siguiente opción): diferencia dentro del ruido de val.
- PR-AUC macro 0,77, menor que la de la regresión logística sobre vistas (0,82): con k pequeño, las "probabilidades" de kNN son fracciones de pocos vecinos y ordenan peor.

## Resumen de modelos sklearn (val, macro-F1 media de misiones)

| Modelo | A · físicas | B · vistas |
|---|---|---|
| Dummy | 0,22 | 0,22 |
| Regresión logística | 0,66 | **0,80** |
| kNN | **0,78** | 0,49 (sin PCA) / 0,79 (con PCA) |
| Gradient boosting | 0,78 | 0,80 (Kepler 0,89) |

Los mejores (logreg B, kNN A, PCA + kNN B) quedan a ~0,78–0,80, con IC que se solapan: en val no se distinguen. El error común es FP → PLANET. Es el listón que tendrán que superar el boosting, el MLP y la CNN.

## Gradient boosting (2026-11-02)

`--models hgb`: `HistGradientBoostingClassifier` con `class_weight="balanced"` y parada temprana (15 % de train; val no interviene). Recibe los datos ya imputados, así que no puede usar el patrón de NaN como atajo. Rejilla de 24 combinaciones por misión: tasa de aprendizaje 0,03 o 0,1, 7–31 hojas, 10 o 30 estrellas mínimas por hoja, regularización L2 0 o 1. Tarda ~21 min en CPU, casi todo en la variante B (283 entradas). Resultados: `results/models/hgb_*.csv`.

| Macro-F1 (val) | Kepler | K2 | TESS | Media | PR-AUC macro |
|---|---|---|---|---|---|
| A · físicas | 0,80 [0,73–0,87] | 0,82 [0,72–0,91] | 0,71 [0,58–0,81] | 0,78 [0,72–0,83] | 0,78 |
| B · vistas | **0,89** [0,84–0,94] | 0,84 [0,74–0,92] | 0,66 [0,53–0,77] | 0,80 [0,74–0,85] | **0,86** |

- **Kepler con vistas: 0,89**, el mejor resultado hasta ahora (logreg 0,84). Kepler tiene más datos (~1350 estrellas de train) y la mejor fotometría; ahí los árboles aprovechan relaciones no lineales en la forma.
- **TESS empeora con vistas** (0,66, frente a 0,72 de logreg y PCA + kNN). Con ~590 estrellas de train, 283 entradas y curvas más ruidosas, los árboles sobreajustan. Con las variables físicas, TESS da 0,71.
- En la media, boosting empata con los demás (0,78 con físicas, 0,80 con vistas). Su ventaja es la PR-AUC: 0,86, la mejor ordenación de todos los modelos.
- La mayoría de misiones elige 10 estrellas mínimas por hoja y L2 = 0, el extremo menos regularizado de la rejilla. Ampliarla mejoraría como mucho unas centésimas, dentro del ruido de val.

**Lectura conjunta.** El mejor modelo depende de la misión: con muchos datos (Kepler) gana el boosting sobre vistas; con pocos (TESS), los modelos simples. Esto apunta a que la limitación es la cantidad de datos por misión, no la capacidad del modelo, y es un argumento para entrenar con varias misiones a la vez (fases 3–4).

## Dataset de PyTorch (tarea #15, 2026-11-02)

`exoplanet.dataset.ViewDataset`: una muestra es una estrella, con vista local `(1, 81)`, vista global `(1, 201)`, `log_depth` `(1,)`, clase (índice en `CLASSES`) y misión. Usa la misma normalización que la variante B de sklearn (`exoplanet.views.normalize`); un test comprueba que la entrada coincide. `class_weights()` equivale a `class_weight="balanced"`. Todo se carga en memoria como tensores, porque son ~3 000 estrellas. Par/impar fuera, como en sklearn.

**Entorno.** En Windows, Control inteligente de aplicaciones bloquea `torch.dll` (WinError 4551). PyTorch se ejecuta en **WSL2 (Ubuntu 24.04)** con un entorno propio, `~/venvs/exoplanet` (PyTorch 2.15 CPU, mismas versiones de sklearn y NumPy), sobre los mismos archivos del repo (`/mnt/c/...`). Leer de `/mnt/c` es lento: la batería de tests tarda ~3,5 min en WSL frente a ~10 s en Windows. Si molesta al entrenar, copiar `data/sample/views` al disco de Linux.
