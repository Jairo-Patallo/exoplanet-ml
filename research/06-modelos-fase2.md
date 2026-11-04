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
| MLP | — | 0,79 |
| CNN (media de 3 semillas) | — | ~0,82 por misión, ~0,83 conjunta (PR-AUC 0,91) |

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

## MLP (tarea #29, 2026-11-02)

`scripts/models/mlp.py`, ejecutado en WSL. Mismo protocolo que sklearn: entrada del dataset (283 valores, estandarizados con train), un modelo por misión, pesos de clase "balanced", selección en val por macro-F1. Red: 2 capas ocultas con ReLU y dropout, AdamW (lr 1e-3, lotes de 64). Parada temprana con un 15 % de train apartado (paciencia 20 épocas). Rejilla: 64 o 256 neuronas, dropout 0,2 o 0,5, weight decay 1e-4 o 1e-2. La mejor configuración se repite con 3 semillas. Tarda 27 s en CPU. Resultados: `results/models/mlp_*.csv`; pesos y configuración en `data/models/mlp_<misión>.pt`.

| Macro-F1 (val) | Kepler | K2 | TESS | Media | PR-AUC macro |
|---|---|---|---|---|---|
| MLP | 0,84 [0,78–0,90] | 0,82 [0,72–0,91] | 0,73 [0,62–0,83] | **0,79** [0,74–0,84] | 0,86 |
| Semillas (media ± desv.) | 0,844 ± 0,007 | 0,811 ± 0,023 | 0,726 ± 0,014 | | |

- **No supera el listón de 0,80:** empata con la regresión logística (0,80) y el boosting (0,80) sobre las mismas vistas. PR-AUC 0,86, igual que el boosting.
- **La parada temprana llega muy pronto** (épocas 2–14): con ~600–1350 estrellas y 283 entradas, la red empieza a memorizar enseguida. Por eso el weight decay no cambia nada (mismas cifras con 1e-4 y 1e-2): la regularización efectiva es la parada temprana. En la práctica el MLP se queda cerca de un modelo lineal regularizado, y por eso da lo mismo que la regresión logística.
- La variabilidad entre semillas (desv. 0,01–0,02) es mucho menor que el ancho de los IC (~0,2): el resultado no depende de la suerte de la inicialización.

**Lectura.** Cuatro familias de modelos (lineal, vecinos, árboles, red) acaban en 0,78–0,80 con estos datos. Es una señal fuerte de que el techo lo pone la cantidad de datos por misión y la dificultad de FP frente a PLANET, no el modelo. Una CNN (que aprovecha que los puntos vecinos de la curva están relacionados) podría ganar algo, pero con los mismos datos es de esperar un efecto pequeño. Lo que más promete es entrenar con más datos: varias misiones juntas o etiquetas débiles (`04-armonizacion-etiquetas.md`).

## CNN de dos ramas (tarea #16, 2026-11-02)

`scripts/models/cnn.py`, en WSL. Arquitectura al estilo AstroNet (Shallue & Vanderburg 2018):
- rama global de 5 bloques convolucionales, rama local de 2 bloques, con BatchNorm y max-pool;
- se unen con log10 de la profundidad;
- 2 capas densas de 256 con dropout.

Tiene 0,5 millones (base 8) o 1,3 millones (base 16) de parámetros. Aumento por **reflexión en el tiempo** de las vistas. Mismo entrenamiento que el MLP (`exoplanet.training`: pesos de clase, parada temprana con el 15 % de train, AdamW) y selección en val por macro-F1 entre base 8/16 y dropout 0,3/0,5. Dos formas de entrenar (decisión de Jairo):
- **por misión:** comparable con los demás modelos;
- **conjunta:** una CNN con el train de las tres misiones (~3 veces más datos), configuración elegida por el macro-F1 medio de las misiones. Sigue siendo dentro de dominio, no LOMO.

Tarda ~4 min en CPU. Resultados: `results/models/cnn_*.csv`; log en `data/models/cnn_log.txt`.

| Macro-F1 (val) | Kepler | K2 | TESS | Media | PR-AUC macro |
|---|---|---|---|---|---|
| CNN por misión (semilla 0) | 0,87 [0,80–0,92] | 0,89 [0,81–0,96] | 0,80 [0,70–0,88] | **0,85** [0,80–0,90] | **0,91** |
| CNN conjunta (semilla 0) | 0,86 [0,79–0,91] | 0,85 [0,75–0,93] | 0,80 [0,70–0,90] | 0,84 [0,78–0,88] | 0,91 |
| Media de 3 semillas, por misión | 0,85 ± 0,02 | 0,85 ± 0,05 | 0,76 ± 0,04 | ~0,82 | |
| Media de 3 semillas, conjunta | 0,84 ± 0,01 | 0,88 ± 0,04 | 0,78 ± 0,02 | ~0,83 | |

**Lectura.**
- **La CNN es el mejor modelo de la fase 2**, sobre todo en **PR-AUC: 0,91**, frente a 0,86 del MLP y del boosting. Ordena claramente mejor las estrellas. En macro-F1 la mejora es más modesta.
- **La cifra de 0,85 es optimista.** La semilla 0 sirvió para elegir la configuración y es además la mejor de las tres. La media de las 3 semillas (~0,82–0,83) es la estimación honesta. Frente al MLP (0,79, que apenas variaba con la semilla), la ganancia real es de ~+0,03–0,04, dentro de unos IC que se solapan.
- **La CNN es más inestable entre semillas** (desv. hasta 0,05 en K2 y TESS) que el MLP (≤ 0,02): con 0,5 millones de parámetros y ~600 estrellas, cada entrenamiento acaba en un sitio distinto. Si se usa, conviene **promediar varias semillas** (ensemble), que además suele mejorar el resultado.
- **Entrenar con las tres misiones juntas no gana con claridad:** media ~0,83 frente a ~0,82. Ayuda algo en K2 y TESS, las misiones con menos datos (0,88 frente a 0,85 y 0,78 frente a 0,76 en media de semillas), y empata en Kepler. Los datos de una misión sirven poco para otra tal cual; es justo lo que estudia la fase 3.
- La parada temprana sigue llegando pronto (épocas 2–23): la CNN también memoriza enseguida. El aumento por reflexión no lo evita del todo.
- La convolución ayuda por su sesgo inductivo, no por tener más parámetros: aprovecha que los puntos vecinos de la curva están relacionados y reutiliza los mismos filtros en toda la vista.

## Muestra completa (2026-11-03)

Se descargaron todas las estrellas con etiqueta fuerte: Kepler 6 560, TESS 1 624 y K2 398 positivos, más un NO_SIGNAL por positivo. En total 17 164 estrellas, ~14 700 con vista válida, frente a ~3 100 en la muestra A. K2 no cambia: ya estaba completa. Mismo código, mismo protocolo y misma selección en val. Val crece de 311 a 1 482 estrellas (Kepler 166 → 1 116, TESS 76 → 297). Los resultados de la muestra A quedan en `results/models/sampleA/`.

| Macro-F1 en val, media de misiones [IC 95 %] | Muestra A | Muestra completa | PR-AUC (completa) |
|---|---|---|---|
| Dummy | 0,22 | 0,22 | — |
| Regresión logística, físicas / vistas | 0,66 / 0,80 | 0,67 / 0,78 [0,74–0,81] | 0,70 / 0,80 |
| kNN, físicas / PCA + vistas | 0,78 / 0,79 | 0,73 / 0,80 [0,76–0,83] | 0,73 / 0,78 |
| Gradient boosting, físicas / vistas | 0,78 / 0,80 | 0,75 / **0,82** [0,78–0,85] | 0,79 / 0,85 |
| MLP | 0,79 | 0,78 [0,74–0,81] | 0,84 |
| **CNN** por misión / conjunta | 0,85 / 0,84 | **0,84** [0,81–0,87] / **0,84** [0,81–0,88] | **0,91 / 0,90** |
| CNN, media de 3 semillas, por misión / conjunta | ~0,82 / ~0,83 | 0,83 / 0,84 | |

Por misión (CNN por misión, completa): Kepler 0,87 [0,85–0,89], K2 0,87 [0,78–0,95], TESS 0,78 [0,73–0,83].

**Lectura.**
- **Las medidas son mucho más fiables:** el IC de Kepler pasa de ±0,06 a ±0,02. Ahora sí se distinguen modelos dentro de Kepler.
- **Más datos no han roto el techo.** Kepler tiene 6,5 veces más estrellas de entrenamiento y la CNN se queda en 0,87, igual que antes. El límite no es la cantidad de datos sino la información de la curva: algunos falsos positivos no se distinguen de un planeta solo con la forma del tránsito.
- **TESS baja en casi todos los modelos** (logreg 0,72 → 0,68; MLP 0,73 → 0,67). El val anterior (76 estrellas) era optimista por azar; con 297 la cifra es más realista. La excepción es el boosting sobre vistas (0,66 → 0,75): con 4 veces más datos de TESS deja de sobreajustar.
- **Ranking estable:** CNN (0,84, PR-AUC 0,91) > boosting sobre vistas (0,82) > resto (~0,78–0,80). Entrenar con las tres misiones juntas empata con entrenar por misión (0,84).
- **Errores de la CNN en Kepler** (`cnn_val_confusion.csv`): de 384 FP, 39 se toman por PLANET y **42 por NO_SIGNAL**. Estos últimos son probablemente FP sin un tránsito visible en la curva de la propia estrella: el eclipse está en una estrella vecina (desplazamiento del centroide) o es una coincidencia de efemérides. El Robovetter los detecta con información que no está en la curva. Es un **límite físico de un modelo "solo curva"** y conviene medirlo (análisis de errores por tipo de FP).

## MOMENT congelado + regresión logística (2026-11-03)

`scripts/models/moment_probe.py`, en un entorno aparte de WSL (`~/venvs/moment`). momentfm fija versiones antiguas, así que se instaló sin dependencias y con transformers 4.57; el modelo carga y funciona. MOMENT (AutonLab, licencia MIT) es un modelo general de series temporales. Su preentrenamiento (Time-series Pile: Informer, Monash, UCR/UEA, TSB-UAD) no incluye Kepler, K2 ni TESS, así que no hay fuga de etiquetas. El modelo no se entrena:
- cada vista normalizada se rellena hasta 512 puntos con máscara y se convierte en un vector;
- se unen el vector de la local, el de la global y log10 de la profundidad;
- encima va una regresión logística por misión, con el mismo protocolo que el resto.

| Macro-F1 en val (muestra completa) | Kepler | K2 | TESS | Media | PR-AUC |
|---|---|---|---|---|---|
| Regresión logística sobre vistas | 0,83 | 0,83 | 0,68 | 0,78 | 0,80 |
| MOMENT-small (35 M) + logística | 0,86 | 0,85 | 0,74 | **0,82** [0,78–0,85] | 0,84 |
| MOMENT-large (341 M) + logística | 0,85 | 0,87 | 0,72 | 0,81 [0,78–0,84] | 0,85 |
| CNN (referencia) | 0,87 | 0,87 | 0,78 | 0,84 | 0,91 |

- **Los vectores de MOMENT contienen información útil sin haber visto un solo tránsito.** Con la misma regresión logística suben de 0,78 a 0,82, sobre todo en TESS.
- **El modelo grande no mejora al pequeño** (0,81 frente a 0,82) y tarda 52 min frente a 13 min en CPU. Congelado, el tamaño no ayuda.
- No supera a la CNN, sobre todo en PR-AUC (0,84–0,85 frente a 0,91). El siguiente paso razonable sería el fine-tuning (paso 2), ya viable con la GPU.

## GPU AMD en WSL (2026-11-03)

RX 9070 XT (gfx1201) funcionando con **ROCm 7.2.1** y **PyTorch 2.9.1+rocm7.2.1** (wheel oficial de repo.radeon.com) en el entorno `~/venvs/exoplanet-rocm`. En ROCm 7.2.1 el instalador ya no tiene el usecase `wsl`: se instala con `amdgpu-install --usecase=rocm --no-dkms`, más el puente **librocdxg** (paquete `rocdxg-roct` 1.2.2) y `HSA_ENABLE_DXG_DETECTION=1`. El controlador de Windows de octubre de 2026 sirve.

- Rendimiento medido: **bf16 ~112 TFLOPS**, pero **fp32 solo ~1,2 TFLOPS** (las rutinas fp32 aún no están optimizadas para gfx1201 en esta versión; cambiar de librería BLAS no lo arregla). Conclusión: entrenar en GPU con **precisión mixta bf16** (`torch.autocast`).
- Verificación: el MLP entrenado en GPU da macro-F1 medio 0,777 frente a 0,781 en CPU, una diferencia dentro de la variación entre semillas. Matmul GPU frente a CPU: error relativo 2,5e-6.

## Fine-tuning de MOMENT-small en GPU (2026-11-03)

`scripts/models/moment_finetune.py` (WSL + ROCm, bf16; ~21 min en total, 2–4 min por entrenamiento).
- **Arquitectura:** MOMENT-small entero entrenable (35 M de parámetros), compartido para la vista local y la global. Sus dos vectores se unen con log10 de la profundidad y pasan a una cabeza densa de 256.
- **Entrenamiento:** tasa de aprendizaje baja para MOMENT y 1e-3 para la cabeza. Una sola red con las tres misiones. Mismo protocolo que la CNN conjunta: pesos de clase, parada temprana con el 15 % de train, reflexión, selección en val y 3 semillas.
- **Cambio en `exoplanet.training.fit`:** admite tasa de aprendizaje por grupos, bf16 y paciencia. Con los valores por defecto, el MLP da resultados idénticos byte a byte.

| Tasa de MOMENT | 1e-5 | **3e-5** | 1e-4 |
|---|---|---|---|
| Macro-F1 medio en val (semilla 0) | 0,804 | **0,811** | 0,806 |

| Macro-F1 en val | Kepler | K2 | TESS | Media | PR-AUC |
|---|---|---|---|---|---|
| MOMENT fine-tuning (semilla 0) | 0,87 [0,85–0,89] | 0,84 | 0,72 [0,67–0,78] | 0,81 [0,78–0,85] | 0,88 |
| MOMENT fine-tuning, media de 3 semillas | 0,872 | 0,843 | 0,729 | 0,815 | |
| MOMENT congelado (small) | 0,86 | 0,85 | 0,74 | 0,82 | 0,84 |
| CNN conjunta (media de 3 semillas) | 0,868 | 0,886 | 0,764 | 0,839 | 0,90 |

**Lectura.** El fine-tuning **no supera a la CNN** ni al MOMENT congelado en macro-F1, aunque mejora la ordenación respecto al congelado (PR-AUC 0,88 frente a 0,84). En Kepler, con muchos datos, empata con la CNN (0,87). En TESS y K2 pierde: con 35 M de parámetros sobreajusta enseguida (para en las épocas 6–16).

**Conclusión de la fase 2 con la entrada "solo curva":** seis familias de modelos (lineal, kNN, boosting, MLP, CNN y un modelo preentrenado de 35–341 M de parámetros) se quedan en 0,81–0,84 con la muestra completa. El límite está en la información de la entrada, no en el modelo. Siguiente paso propuesto: añadir la vista de centroide, normalizada por estrella para que sea comparable entre telescopios, como variante "curva + centroide", y mantener "solo curva" como versión 100 % portable (CoRoT).

## Variante curva + centroide (2026-11-04)

**Vista de centroide** (`curves.read_centroid`, `folding.fold_centroid`, `scripts/data/build_centroid.py`): centroide (MOM_CENTR1/2) de los mismos segmentos que el flujo, sin la deriva lenta del apuntado. Se pliega cada eje como el flujo y se calcula r = √(dx² + dy²) por bin en **unidades de su ruido** fuera del tránsito, para que sea comparable entre píxeles de 4″ (Kepler) y 21″ (TESS). El modelo recibe sign·log(1+|z|). Se construyó con los FITS en caché: segmentos iguales que el flujo en el 100 % y comprobación del flujo reproducida en 286 estrellas al azar. Entra en la CNN como **segundo canal** de cada vista (`ViewDataset(centroid=True)`, `cnn.py --centroid`).

**Arreglos hechos por el camino.**
- `curves.mast_sequence` convertía también los sectores 100–107 de TESS en el 10. 47 negativos de TESS (2,9 %) se habían emparejado con el sector equivocado. Corregido (solo K2 tiene mitades de campaña), con un test nuevo, y esos 47 se regeneraron.
- Al limpiar se borraron por error 6 607 vistas de negativos; se replegaron, y `qc.csv` coincide con git en n_points, valid y frac_empty de los 8 535 negativos comunes.
- MAST se colgó una vez durante el replegado; se añadió un vigilante que relanza.

**Test de atajo del centroide** (vista sin el 20 % central, CV en train; `scripts/checks/shortcut_centroid.py`): PLANET vs NO_SIGNAL Kepler 0,62, K2 0,68, TESS 0,52; FP vs NO_SIGNAL 0,67 / 0,61 / 0,52. Parecido al del flujo (0,60 / 0,59 / 0,55), salvo un atajo moderado en K2 (apuntado inestable).

**CNN en GPU (bf16), con y sin centroide; mismas semillas y protocolo.** Macro-F1 en val, media de 3 semillas:

| | Kepler | K2 | TESS | Media |
|---|---|---|---|---|
| Solo curva, por misión | 0,871 | 0,861 | 0,763 | 0,832 |
| **Curva + centroide, por misión** | **0,896** | 0,743 | 0,683 | 0,774 |
| Solo curva, conjunta | 0,865 | 0,892 | 0,772 | 0,843 |
| **Curva + centroide, conjunta** | **0,893** | 0,851 | 0,737 | 0,827 |

En Kepler (conjunta, semilla 0): F1 de FP 0,853 → **0,888** y de PLANET 0,804 → **0,846**, con NO_SIGNAL igual (0,95). Los FP confundidos con PLANET pasan de 39 a 31 y con NO_SIGNAL de 42 a 31; PR-AUC 0,927 → 0,947.

**Lectura.**
- **En Kepler el centroide ayuda de verdad:** +0,03 en macro-F1, consistente en las 3 semillas y en las dos formas de entrenar. La mejora está en separar FP de PLANET, no en NO_SIGNAL, así que es física (binarias de fondo) y no el atajo.
- **En K2 y TESS empeora:** −0,04 a −0,12. En TESS los píxeles son 5 veces mayores, el centroide se diluye entre estrellas vecinas y apenas aporta señal; en K2 el apuntado es inestable y hay un atajo moderado. Con poca información útil, el canal extra es ruido que el modelo sobreajusta.
- **Implicación para la portabilidad:** el valor del centroide depende del telescopio (de su escala de píxel). No sirve como entrada universal tal cual. Opciones: usarlo solo donde la escala de píxel lo permite, o un modelo con centroide opcional ("dropout" del canal). Se evaluará en la fase 3.

## Análisis de errores por tipo de falso positivo (Kepler, 2026-11-04)

`scripts/checks/error_analysis_fp.py`. Usa las CNN conjuntas guardadas (semilla 0); las predicciones reproducen las métricas (0,848 y 0,831). Cada FP de Kepler en val (384) se cruzó con su KOI (tabla cumulative, `fetch_labels.py`) por kepid y periodo más cercano; todos encontrados. Un FP puede tener varios motivos. Resultados: `results/models/error_analysis_fp_kepler.csv` y predicciones estrella a estrella en `error_analysis_val_predictions.csv`.

| Motivo del Robovetter | n | Acierta FP (solo curva → con centroide) | Lo toma por PLANET | Lo toma por NO_SIGNAL |
|---|---|---|---|---|
| nt: no parece tránsito | 79 | 73 % → 72 % | 5 % → 8 % | **22 % → 20 %** |
| ss: binaria eclipsante | 209 | 88 % → **92 %** | 9 % → 6 % | 3 % → 2 % |
| co: centroide desplazado | 172 | 76 % → **87 %** | 16 % → 8 % | 8 % → 5 % |
| ec: coincidencia de efemérides | 107 | 82 % → **88 %** | 8 % → 5 % | 10 % → 8 % |
| **Solo co/ec** (invisibles en la curva) | 91 | **66 % → 80 %** | **24 % → 12 %** | 10 % → 8 % |
| Algún nt/ss (visibles) | 288 | 84 % → 87 % | 8 % → 6 % | 8 % → 7 % |
| Todos | 384 | 79 % → 84 % | 12 % → 8 % | 9 % → 8 % |

**Lectura.**
1. **El centroide hace justo lo que debía.** En los FP "invisibles" (solo centroide o efemérides, 24 % de los FP) el acierto sube de 66 % a 80 % y los tomados por planeta bajan a la mitad (24 % → 12 %). Es la prueba directa de que su mejora en Kepler es física y no un atajo.
2. **Casi la mitad de los FP que la CNN solo curva toma por planeta son de tipo invisible** (~22 de ~46). Ese es el **techo físico del enfoque solo curva**: esos errores no se pueden arreglar sin información de fuera de la curva.
3. **Aun así, la curva capta parte de los invisibles** (66 %): muchas coincidencias de efemérides copian la señal de una binaria vecina, cuya forma (V, secundario) sigue delatándola.
4. **El otro error grande: FP "no parece tránsito" tomados por NO_SIGNAL (22 %).** No es un fallo del modelo sino de la definición de las clases: un FP de tipo nt es ruido o un artefacto instrumental, y para el modelo se parece a "no hay señal". Unir esa frontera (o tratar nt como NO_SIGNAL) es una decisión de etiquetado, no de modelo.
5. Los errores mejorables están en las binarias (ss, 88 % de acierto) y en esa frontera nt/NO_SIGNAL. Más datos de la misma estrella (más trimestres, resolución mayor) ayudarían en ss; la frontera nt es conceptual.

## Intentos de subir la CNN solo curva: par/impar, híbrida y ensemble (2026-11-04)

Antes se corrigieron los datos:
- **42 negativos de TESS** eran estrellas del dataset o candidatos de otra misión, con otro número de catálogo. `build_negatives.py` excluye ahora todo lo que esté a menos de 2″ de cualquier estrella del dataset, TOI o candidato K2.
- Como el sorteo de candidatos por sector cambia, la lista reproducible nueva **cambia 1 031 negativos de TESS**. Se replegaron (19 sin curva en MAST) y se recalcularon sus centroides. K2 no cambia.

**Diagnósticos con ventana ajustada** (`exoplanet.features`, que sustituye a `legacy/features.py`). En la global, la ventana del tránsito y del secundario se ajusta a duty·201 bins, en vez de 11 fijos. AUC FP vs PLANET en val: odd_even_z 0,52 → 0,56 y secondary_z 0,56 → 0,64. Sin `duty` reproduce exactamente la versión antigua.

**Variantes de la CNN** (GPU, bf16, entrenamiento conjunto, mismo protocolo; `cnn.py --odd-even --features`). Macro-F1 en val:

| Variante | Kepler | K2 | TESS | **Media (ensemble de 3)** [IC 95 %] | PR-AUC | Media de 3 semillas |
|---|---|---|---|---|---|---|
| Base (solo curva) | 0,866 | 0,904 | 0,779 | **0,850** [0,82–0,88] | 0,911 | 0,833 |
| + par/impar (canales globales) | 0,860 | 0,873 | 0,760 | 0,831 [0,80–0,86] | 0,900 | 0,825 |
| + diagnósticos (híbrida) | 0,871 | 0,887 | 0,762 | 0,840 [0,80–0,87] | 0,903 | 0,831 |
| + ambas | 0,876 | 0,907 | 0,773 | 0,852 [0,82–0,88] | 0,913 | 0,827 |

**Lectura.**
- **La única mejora real es el ensemble:** promediar las probabilidades de 3 semillas sube la base de 0,833 (media de semillas) a **0,850**, sobre todo en TESS (0,752 → 0,779).
- **Par/impar en la vista global no aporta:** un tránsito ocupa 2–3 de los 201 bins, así que los canales extra son casi solo ruido, y K2/TESS lo sobreajustan. Haría falta una vista par/impar local, que exige replegar todo.
- **Los diagnósticos tampoco aportan** (0,831 frente a 0,833): la CNN ya extrae esa información de las mismas vistas. Solo Kepler mejora un poco en FP (0,837 → 0,845).
- "+ ambas" empata con la base (0,852 frente a 0,850 en ensemble; 0,827 frente a 0,833 en semillas): diferencias dentro del ruido.
- Confirma la conclusión anterior: con estas vistas, la CNN solo curva está en su techo (~0,85 con ensemble). Las ganancias que quedan requieren **otra información**, como el centroide en Kepler, o **vistas nuevas**: par/impar local y vista del secundario, a costa de replegar todo (~5 h).

**Nota de versión de datos.** Los resultados de logreg, kNN, boosting, MLP, MOMENT, la CNN en CPU (`cnn_val.csv`) y la CNN con centroide (`cnn_centroid_gpu_*`) se calcularon antes de regenerar 1 031 negativos de TESS (2026-11-04). Las variantes de esta sección (`cnn_gpu_*`, `cnn_oddeven_*`, `cnn_feats_*`, `cnn_oddeven_feats_*`) usan ya los datos corregidos. La base salió casi igual (0,843 frente a 0,848), así que las conclusiones no cambian, pero las comparaciones finas deben hacerse dentro de la misma versión.
