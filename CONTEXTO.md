# Contexto del proyecto exoplanet-ml (para continuar en otro chat)

Actualizado el 2026-11-05. Lee también `CLAUDE.md` (reglas de trabajo), `README.md` (estructura y cómo reproducir) y `research/` (decisiones razonadas, sobre todo `05-negativos-no-signal.md` y `06-modelos-fase2.md`).

## 1. Objetivo

Un modelo que decide si una señal de tránsito es **PLANET**, **FALSE_POSITIVE** o **NO_SIGNAL** usando **solo la curva de luz** (y variables derivadas de ella), **portable a telescopios nuevos**. La portabilidad se mide entrenando con dos misiones y probando en la tercera (LOMO: Kepler, K2, TESS). CoRoT sería una cuarta misión si llegan sus etiquetas. Fases finales: modelo portable con efeméride dada (fase 4), búsqueda sin efeméride para telescopios nuevos, como PLATO o el seguimiento terrestre en España (fase 5, desde 2026-11-06), y web (fase 6).

## 2. Cómo se trabaja

- Responder en español, con explicaciones sencillas. El usuario (Jairo) dirige; Claude implementa, evalúa cada instrucción y explica código y resultados.
- Las decisiones importantes de ML son del usuario: proponer con pros y contras y esperar. No empezar una fase sin su visto bueno.
- Tareas en Notion (sección 7). Una tarea es Hecho solo con un artefacto verificable. Revisar al empezar la columna "Nota de Jairo" (Tareas y Fases).
- Pedir permiso para commit/push. Descargas: presupuesto de 50 GB acumulados (llevan ~23 GB en `data/raw_lc`).
- **Antes de borrar datos:** imprimir cuántas filas o archivos se borran, comparar por `star_id` (único) y parar si no es lo esperado. En 2026-11-03 se borraron por error 6 607 vistas al comparar por una columna con vacíos.
- Procesos largos: en segundo plano, con informes. MAST se cuelga a veces sin límite de tiempo; usar un vigilante que relance si el log no avanza en 10 min. `qc.csv` se escribe de forma atómica (un reinicio lo corrompió una vez).

## 3. Datos y entorno

- Dataset de partida: Hugging Face `bingbangboom/exoplanet-transit-detection` en `data/{train,val,test}.parquet`, con splits por estrella. Sus vistas plegadas no se usan: la global y la par/impar no están centradas y las de NO_SIGNAL son ceros.
- **Muestra completa** (2026-11-03): todas las estrellas con etiqueta fuerte, es decir, positivos Kepler 6 560, TESS 1 624 y K2 398, más 8 582 NO_SIGNAL 1:1. Son 17 164 estrellas, ~14 700 con vista válida.
  - Curvas en `data/raw_lc` (~23 GB) y vistas en `data/sample/views`.
  - Centroide en `data/sample/centroid`.
  - Calidad por estrella en `results/sample/qc.csv`; la muestra A antigua está en `qc_sampleA.csv`.
- **Negativos** (`build_negatives.py`):
  - Kepler: NO_SIGNAL del dataset en su split.
  - K2 y TESS: estrellas sin candidato del mismo sector o campaña y brillo parecido al de su pareja, con efeméride de planetas de train del mismo segmento.
  - Desde 2026-11-04 se excluye todo lo que esté a menos de 2″ de una estrella del dataset o de un candidato de cualquier misión.
  - Reproducible byte a byte.
- **Entornos:**
  - Windows `.venv` (Python 3.12, `requirements.txt`): datos y sklearn.
  - WSL2 Ubuntu 24.04 (`requirements-wsl.txt`): `~/venvs/exoplanet` (PyTorch 2.15 CPU) y `~/venvs/exoplanet-rocm` (GPU RX 9070 XT, ROCm 7.2.1, PyTorch 2.9.1, MOMENT).
  - En la GPU: `export HSA_ENABLE_DXG_DETECTION=1` y entrenar con bf16 (`--amp`), porque fp32 va a ~1 TFLOPS.
  - PyTorch no carga en Windows (Control inteligente de aplicaciones).
  - Desde Git Bash, `wsl.exe ... bash -lc '...'` pierde las variables: lanzar scripts `.sh` con `MSYS_NO_PATHCONV=1`.

## 4. Código (`src/exoplanet/`)

| Módulo | Qué hace |
|---|---|
| `data.py` | Carga por misión y split, limpieza (RP_MAX = 22 R⊕ para PLANET) |
| `labels.py` | Fuerza de etiqueta (fuerte / débil / excluida), `strong_stars`, `nearest_period` |
| `curves.py` | MAST: selección de segmentos (1 en K2/TESS, ~100 d en Kepler), descarga, aplanado; `read_centroid`; `run_star` |
| `folding.py` | `fold_views` (local 81, global/par/impar 201) y `fold_centroid` (desplazamiento en unidades de ruido) |
| `negatives.py` | Emparejamiento de negativos y efemérides al azar |
| `features.py` | Diagnósticos físicos (profundidad, SNR, forma, par/impar, secundario), con ventana ajustada a la duración (`duty`) |
| `views.py` | Carga de la muestra, normalización por estrella (AstroNet) y del centroide |
| `dataset.py` | `ViewDataset` de PyTorch (canales opcionales: centroide, par/impar; diagnósticos) |
| `cnn.py` | CNN de dos ramas tipo AstroNet, con diagnósticos opcionales en la cabeza |
| `training.py` | Entrenamiento común (parada temprana con el 15 % de train, pesos de clase, reflexión, lr por grupos, bf16) |
| `evaluation.py` | Métricas de #51 (macro-F1, PR-AUC, IC bootstrap por misión) y folds del protocolo LOMO |

Scripts en `scripts/data`, `scripts/checks` y `scripts/models` (no se importan entre sí). `legacy/` es el primer enfoque; solo dos comprobaciones antiguas lo usan. 41 tests (`python -m pytest -q`; los de PyTorch, en WSL).

## 5. Decisiones tomadas

1. **Métrica (#51):** macro-F1 de 3 clases por misión y media de misiones; PR-AUC one-vs-rest como secundaria; IC 95 % bootstrap; selección en val; el test no se ha tocado.
2. **Etiquetas:** evaluación solo con etiquetas fuertes.
3. **Entrada:** solo la curva y sus derivadas. Profundidad normalizada por estrella y log10 de la profundidad aparte. Nunca metadatos del proceso (n_points, data_days, cadence, n_transits, frac_empty, n_files).
4. **Protocolo LOMO** (`research/03`): la misión excluida no participa en ninguna decisión y se evalúa una vez, entera; el fold interno es train de A → val de B; criterio PR-AUC; las estrellas compartidas entre misiones se quitan del ajuste.
5. **Centroide:** variante aparte ("curva + centroide"); la versión solo curva es la portable (CoRoT no tiene centroide).
6. **Sin profundidad (#43, decisión de Jairo 2026-11-04):** el modelo final no usa log10 de la profundidad; solo la forma normalizada. En el LOMO aportaba +0,003 (dentro del ruido) y es la variable que más delata la misión. Se aplica al construir el modelo de la fase 4; los resultados de las fases 2 y 3 se quedan como están.
7. **Alcance: clasificador de señales ya detectadas (#53, decisión de Jairo 2026-11-05).** Entrada: curva + periodo y época. Con la efeméride de BLS el macro-F1 cae de ~0,86 a ~0,50, sobre todo porque los NO_SIGNAL plegados con un pico de ruido parecen tránsitos (`research/09-bls.md`). NO_SIGNAL = "la efeméride dada no es un tránsito real".
8. **Prioridad: triaje en modo búsqueda (decisión de Jairo 2026-11-06).** Lo que tiene que ser fiable es separar CANDIDATO (PLANET o FP) de SIN_SEÑAL (`NO_SIGNAL` en el código; los nombres nuevos son solo de presentación) sin efeméride. Objetivo (fijado por Jairo 2026-11-06): **detectar el 95 % de los candidatos y que pase el mínimo posible de SIN_SEÑAL**. La métrica principal es la fracción de SIN_SEÑAL que pasa con el umbral que detecta el 95 % de los candidatos en val (menor es mejor); también se reporta el recall con 1 % de falsas alarmas y la detección por SNR. Cadena:
   1. CNN sobre la curva sin plegar como filtro rápido (`research/12`);
   2. BLS solo en las estrellas que pasan, para dar el periodo;
   3. puntuación combinada (SDE + CNN plegada + CNN sin plegar);
   4. planeta frente a FP: el modelo con negativos difíciles en búsqueda, y el modelo final (decisión 7) cuando el usuario da la efeméride.
   Si mejorar la CNN sin plegar (semillas, anchura, inyecciones débiles) no basta, se pasa a un Transformer sobre la misma entrada.
9. **Destino de la fase 5: seguimiento terrestre, herramienta para aficionados (decisión de Jairo 2026-11-06).** Un aficionado observa una noche el tránsito de un TOI o planeta conocido, con la efeméride dada, y la herramienta le dice si el tránsito aparece en su curva. Es otro dominio: una sola noche, tránsitos parciales y ruido atmosférico; se reutiliza sobre todo la vista local. Primer paso: estudio de viabilidad de curvas terrestres públicas y etiquetadas. La búsqueda sin efeméride (`research/12`) se cierra y se documenta.
10. **La fase 5 es un recomendador de observación para aficionados (decisión de Jairo 2026-11-06; sustituye el primer paso de la decisión 9).** La herramienta responde a «¿qué candidato observo esta noche?» para los ~4 800 TOIs de TESS en estado PC y los candidatos de K2 sin resolver. Combina:
    - la probabilidad de planeta del modelo final, con la efeméride de ExoFOP y calibrada;
    - la observabilidad desde la ubicación del aficionado;
    - la detectabilidad con un telescopio pequeño;
    - la incertidumbre de la efeméride.
    Incluye una validación prospectiva: las predicciones se congelan y se comparan con las disposiciones futuras de TFOPWG. Kepler y K2 siguen siendo la base del entrenamiento. Analizar las curvas que suban los aficionados queda como trabajo futuro. Primera tarea: revisar las herramientas existentes para confirmar qué aporta la nuestra.
11. **La fase 5 es una herramienta de 3 modos que aprovecha todo el trabajo (decisión de Jairo 2026-11-06; amplía la 10):**
    - **Modo 1, «¿Qué observo esta noche?»:** el recomendador de la decisión 10.
    - **Modo 2, «Informe de un candidato»:** las 3 probabilidades calibradas, las vistas, par/impar, el secundario, qué zona de la curva mira el modelo, el tipo probable de FP y los límites del techo físico; también en lote para astrónomos.
    - **Modo 3, «Busca en una estrella»** (experimental): CNN sin plegar + BLS + modelo con negativos difíciles, con la fiabilidad medida por SNR (`research/12`).
    - **Futuro, modo 4, «Analiza tu observación»:** curvas terrestres de aficionados.
    Antes de nada, mejorar TESS: todos los sectores, todas las etiquetas y binarias sintéticas. Posible colaboración: el grupo de exoplanetas e IA de la Universidad de Oviedo, donde estudia Jairo.

## 6. Resultados y hallazgos clave

- **Fase 2** (val, muestra completa, macro-F1 media de misiones): dummy 0,22; regresión logística 0,78; kNN + PCA 0,80; boosting 0,82; MLP 0,78; MOMENT congelado 0,82 y con fine-tuning 0,815.
  - **CNN 0,833 de media de semillas y 0,850 con ensemble de 3** (Kepler 0,87, K2 0,90, TESS 0,78).
  - Con centroide, Kepler sube a 0,89, pero K2 y TESS empeoran.
  - Par/impar global y diagnósticos en la CNN no aportan.
- **Análisis de errores (Kepler):** casi la mitad de los FP que la CNN toma por planeta son de tipo centroide o coincidencia de efemérides, invisibles en la curva: es el techo físico. El centroide sube su acierto del 66 % al 80 %. El 22 % de los FP "no parece tránsito" se predicen como NO_SIGNAL, porque la frontera entre esas clases es difusa (decisión de etiquetado pendiente).
- **Atajos:** sin mirar el tránsito, PLANET vs NO_SIGNAL da 0,55–0,60 (sesgo de detección aceptado). El centroide tiene un atajo moderado en K2 (0,68).
- **La misión se adivina** a partir de las vistas (AUC 0,73–0,92): a estudiar en la fase 3 (ablación #43).
- **Estrellas en dos misiones:** 29 físicas (26 Kepler–TESS y 3 K2–TESS), 16 con splits distintos. Se excluyen del ajuste en LOMO (`evaluation.shared_stars`).
- **Errores corregidos que no hay que repetir:**
  - conversión MJD→BKJD (400 días de error);
  - `mast_sequence` convertía los sectores ≥ 100 de TESS en el 10 (solo K2 tiene mitades de campaña);
  - el `.gitignore` ignoraba `scripts/data/`;
  - `load_views` abría todos los archivos a la vez;
  - la ventana fija de 11 bins diluía par/impar y secundario;
  - los negativos de TESS incluían estrellas con candidato en Kepler o K2 (ahora se excluyen por coordenadas).
- **CoRoT bloqueado:** el catálogo de etiquetas (Deleuil et al. 2018, VizieR `J/A+A/619/A97`) solo está en el CDS, que no responde.

## 7. Notion

- Página principal: https://app.notion.com/p/40715f33376c81ea9534c0e8bd2b267d
- Tareas: `collection://cf064316-c55a-46e0-a764-9985c623078f` · Fases: `collection://a3702b25-45cb-47e0-860f-5837c94448bb` (la relación se llama `Bloque`).
- Fases:
  1. Datos: Hecho.
  2. Modelos por misión: casi cerrada.
  3. LOMO: Hecho (2026-11-05).
  4. Modelo portable (BLS, test único, CoRoT).
  5. Web y documentación.

## 8. Fase 3 (LOMO) y estado actual

- **Decidido:** la misión excluida se evalúa con train + val, sin su test, que queda para la evaluación final. Script `scripts/models/lomo.py` (dummy, logreg y CNN; `--only`; guarda predicciones por estrella).
- **CNN solo curva en LOMO** (PR-AUC macro; dummy 0,333): Kepler 0,894, K2 0,927, TESS 0,764, media 0,862, frente a 0,911 dentro de misión. Macro-F1 media 0,804. El modelo solo curva es portable; **TESS es el difícil**.
- **Diagnóstico de TESS** (`scripts/checks/tess_errors.py`):
  - los FP de TESS solo se aciertan el 42 %;
  - causa 1, pocos tránsitos con un solo sector: mediana 5–8 frente a 17–54 en Kepler y K2; con ≤ 8 tránsitos el acierto de FP es ~30 %, con más de 8 es 55–60 %;
  - causa 2, población distinta: TESS tiene el doble de planetas con P < 2 d, y en Kepler y K2 eso es sobre todo binarias;
  - la logística en LOMO TESS da 0,716 y la CNN 0,764: la complejidad ayuda, pero no cierra la brecha.
- **Curva de aprendizaje** (`scripts/models/learning_curve.py`): con 46–234 estrellas de TESS etiquetadas, +0,02–0,03 de macro-F1; solo con todo el train de TESS se llega a 0,836 / 0,785. Pendiente: dar más peso a TESS o hacer ajuste fino.
- **Descartado:** CNN con log(duración/periodo) + log(periodo) (`--ephem`): empata dentro de misión, porque la duración/periodo ya está implícita en la vista global. El LOMO con efeméride se canceló.
- **Gráfico de variables candidatas:** `results/eda/feature_candidates.png`.
- **Adaptación a TESS (2026-11-04, `research/07`):**
  - AdaBN (`adabn.py`): 0,764 → 0,740. Descartado: no es un cambio de escala.
  - **2 sectores en las mismas 1 278 estrellas** (`build_tess_2sectors.py`, `tess_sectors.py`): 0,780 → 0,814, IC [+0,018, +0,048]. Única mejora: falta información por estrella. Siguiente palanca: curvas FFI con todos los sectores (sin el sesgo de los TOI en 2 min).
  - Recortar Kepler/K2 a 27,4 d como aumento (`build_cropped.py`, `lomo_crop.py`): 0,753. Descartado.
  - Pseudo-etiquetas en mitades de TESS (`pseudo_labels.py`): 0,763, IC [−0,007, +0,004]. Descartado; la versión grande con candidatos no se hizo.
  - Todas estas variantes se midieron en TESS train + val: la cifra final debe salir del test.
- **Ablación #43** (`lomo.py --fixed --no-depth`): sin log10 de la profundidad, −0,003 de media. El modelo no depende de ella.
- **Calibración** (`calibration.py`): argmax es la mejor regla (la corrección EM de prior empeora el macro-F1 en las tres misiones). TESS es demasiado confiado; con precisión 0,9 solo recupera el 12 % de planetas.

## 9. Qué sigue (decisiones pendientes del usuario)

1. **Fase 3 cerrada (2026-11-05).** #67 y #45 descartadas (razones en `research/07`); #9, #17, #42 y #52 pasan a la fase 4.
2. Cerrar formalmente la fase 2 en Notion (#12, #13, #29 y #16 tienen un criterio antiguo de LOMO). Etiquetado pendiente: FP "no parece tránsito" frente a NO_SIGNAL.
3. **Fase 4 (empieza el 2026-11-06):**
   - BLS (#53) hecho: el modelo final es un clasificador de señales ya detectadas (decisión 7); BLS queda como herramienta opcional.
   - Decisión de Jairo: curvas FFI de TESS con todos los sectores.
   - Modelo final: CNN solo curva con Kepler + K2 + TESS, sin profundidad, ensemble y argmax.
   - Evaluación única en test con la receta fijada antes (#18); análisis de errores y techo físico (#19, #39); estado del arte (#38); CoRoT (#31) si se desbloquea.
