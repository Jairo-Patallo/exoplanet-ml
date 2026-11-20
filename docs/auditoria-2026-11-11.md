# Revisión detallada del proyecto

11 de noviembre de 2026 · Preparada por Claude para Jairo · Segunda versión, ampliada.

Este documento revisa lo hecho. El diseño futuro de los modelos y de la aplicación está en [`diseno-futuro-2026-11-11.md`](diseno-futuro-2026-11-11.md), que es la parte más importante.

Continúa la auditoría del 6 de noviembre ([PDF](auditoria-y-propuestas-2026-11-06.pdf); hallazgos A1–C4 y propuestas P1–P22). Numeración de hoy: **M** hallazgos de modelos, **I** de implementación, **U** de utilidad, **R** pasos de la ruta y **D** decisiones.

## 1. Qué se ha revisado

| Qué | Alcance |
|---|---|
| Código | Los 13 módulos de `src/exoplanet/`, los 51 scripts (`data` 14, `models` 21, `checks` 16), los 8 archivos de tests y `legacy/`: unas 7 000 líneas |
| Notas | `research/01`–`16`, y las versiones técnicas anteriores al 10 de noviembre, que solo quedan en el historial de git |
| Resultados | Los CSV de `results/` y las predicciones por estrella guardadas |
| Notion | Las 106 tareas (65 cerradas, 41 abiertas) y las 10 fases; no hay notas tuyas pendientes |
| Ejecutado hoy | Tests (42 pasan, 4 se saltan); métricas recalculadas desde los CSV; diagnósticos nuevos de planeta frente a FP; inferencia en CPU comparada con la v1 congelada |
| Estado del arte | Búsqueda de novedades de 2025–2026 (sección 7) |

Nada del código ni de Notion se ha modificado. Los diagnósticos nuevos son descriptivos: no se ha elegido nada con ellos y no tocan el test, pero suman una mirada más a train + val de TESS.

## 2. Veredicto

1. **La metodología es sólida.** Test usado una vez con la receta fijada antes, LOMO anidado con tests propios, atajos medidos, resultados nulos escritos y una validación prospectiva con huellas SHA-256.
2. **El modelo es un buen primer filtro y un juez débil.** Rechaza ruido y binarias evidentes y eso se transfiere entre telescopios. Distingue mal los impostores sutiles, que son los que quedan en la lista de TOIs (M2).
3. **TESS no es difícil por falta de datos por estrella, sino porque sus falsos positivos son otra cosa.** Casi no tienen secundario ni profundidad delatora. Eso rebaja la apuesta por más vistas de la curva y sube la de usar varios sectores y la información de contaminación (M2, M3).
4. **La calibración no se transfiere entre telescopios.** La temperatura «de respaldo» para telescopios nuevos empeora la calibración en el caso más parecido a un telescopio nuevo (M5).
5. **Como herramienta, hoy no existe,** pero servirla es barato: 1,29 millones de parámetros, 14 ms por candidato en CPU (U1, U7).
6. **El hueco frente a otras herramientas es más estrecho de lo que dije esta mañana.** ExoMiner++ 2.0 ya puntúa señales de FFI de SPOC, y existen WATSON-Net y DART-Vetter (sección 7).

## 3. Estado por fase

| Fase | Qué se hizo | Resultado verificado | Valoración |
|---|---|---|---|
| 1 Datos | Curvas de MAST, plegado único, negativos emparejados, tests de atajo | 14 694 estrellas con vista válida (Kepler 11 037, TESS 2 926, K2 731) | Muy buena |
| 2 Modelos | Dummy → logística → kNN → boosting → MLP → CNN → MOMENT | CNN 0,850 en val; seis familias entre 0,78 y 0,85 | Muy buena |
| 3 LOMO | Dos misiones para entrenar, la tercera para evaluar | PR-AUC 0,894 / 0,927 / 0,764 | Excelente |
| 4 Modelo final | CNN solo curva, 5 semillas, test único | Macro-F1 0,886 / 0,928 / 0,720 | Ejemplar |
| 5 Búsqueda | CNN sin plegar + BLS | Recall 0,564 con 1 % de falsas alarmas | Bien cerrada |
| 6 TESS | Multisector, etiquetas, temperatura | Sin potencia; ECE de Kepler 0,039 → 0,017 | Débil; ver M3 y M5 |
| 7 Investigación | Tránsitos individuales; v1 congelada | +0,000; 3 536 TOIs con etiqueta `v1` | En curso, sin tareas activas |
| 8 y 9 | — | — | Sin empezar |

Planeta frente a falso positivo, que es la cifra útil para candidatos:

| | Kepler | K2 | TESS |
|---|---|---|---|
| Test, modelo final (ROC / PR-AUC) | 0,966 / 0,909 | 0,995 / 0,995 | 0,803 / 0,766 |
| Val, viendo la misión | 0,953 / 0,889 | 0,972 / 0,966 | 0,856 / 0,801 |
| LOMO, sin ver la misión | 0,897 / 0,771 | 0,955 / 0,956 | 0,780 / 0,699 |
| Señales en test / val / LOMO | 561 / 560 / 4 996 | 39 / 36 / 329 | 147 / 148 / 1 325 |

## 4. Hallazgos nuevos sobre los modelos

### M1 · La red usa el periodo como prior de población (ALTA)

La puntuación de planeta crece con el periodo: correlación de Spearman con log P de 0,57 (Kepler), 0,62 (K2) y 0,46 (TESS). Dentro de tramos de periodo, la red separa peor que en conjunto:

| LOMO, ROC planeta / FP | Global | Dentro de tramos de periodo |
|---|---|---|
| Kepler | 0,897 | 0,839 |
| K2 | 0,955 | 0,923 |
| TESS | 0,780 | 0,753 |

- El periodo es información legítima (él solo da un AUC de 0,66–0,81), pero es una propiedad de la **población** de cada telescopio, no de la señal.
- En TESS sin verlo, el modelo dice demasiados planetas a periodos largos: con P entre 4 y 8 d predice 0,69 de media y la fracción real es 0,52.
- Corrige la lectura de la fase 3. «Con ≤ 8 tránsitos acierta menos FP» es sobre todo este efecto: con pocos tránsitos el periodo es largo y el modelo se inclina a planeta. El ROC por número de tránsitos es plano (0,73–0,82).

### M2 · Los falsos positivos de TESS son de otra naturaleza (ALTA)

| Planeta frente a FP, estrellas de LOMO | Kepler | K2 | TESS |
|---|---|---|---|
| FP con secundario claro (z > 3) | 20 % | 50 % | 4 % |
| FP con profundidad > 2 % | 28 % | 81 % | 6 % (planetas: 5 %) |
| AUC de la forma en V (`ingress_frac`) | 0,76 | 0,77 | 0,58 |
| AUC de la profundidad sola | 0,57 | 0,93 | 0,46 |
| AUC del secundario solo | 0,60 | 0,71 | 0,54 |

Medido en las vistas plegadas con `curve_features` (en Kepler, sobre una muestra de 2 000 estrellas).

- En Kepler y K2, «falso positivo» incluye binarias evidentes. En TESS, la etiqueta viene de TOIs: candidatos que ya pasaron un filtro profesional y se retiraron después del seguimiento. Lo que queda son impostores que se parecen a un planeta.
- En el test de TESS, los FP que pasan por planeta son sobre todo de contaminación por otra estrella (15 de 32, el 47 %) y sin motivo reconocido (8 de 15); las binarias, 10 de 38.
- El ROC en TESS no mejora con la señal: es 0,75–0,84 en todos los tramos de SNR, también por encima de 80. No es un problema de ruido.
- **Consecuencia:** la P2 de la auditoría anterior (vistas del secundario y par/impar locales) tiene menos recorrido en TESS de lo que estimé. Antes de construirla hay que hacer su comprobación barata. La información que falta en TESS está sobre todo fuera de la curva plegada.
- **La parte buena:** el modelo sí hace bien, y transfiere, el primer filtro. Eso es lo que necesita una lista de candidatos que nadie ha revisado todavía.

### M3 · Más sectores sí ayudan a separar planeta de FP (MEDIA)

Con las predicciones de la fase 3 (mismas 639 estrellas de TESS, modelo que no vio TESS): ROC 0,799 → 0,828 con 2 sectores, IC 95 % de la diferencia [+0,013, +0,046]; PR-AUC 0,762 → 0,802 [+0,014, +0,066]. La ganancia de la fase 3 no era solo de «hay señal o no». Resuelve a medias el hallazgo A1: usar todos los sectores **al predecir** es una mejora probable que no exige reentrenar. Falta medirlo con el modelo final y con más de 2 sectores.

### M4 · Las tres clases mezclan dos preguntas (MEDIA)

- «¿Hay una señal real?» y «¿es planeta o impostor?» comparten una sola salida. Por eso los negativos difíciles mejoraron la primera (ruido que pasa por FP: 69 % → 3 %) y empeoraron la segunda (−0,03/−0,04).
- El 22–24 % de los FP «no parece tránsito» se predicen como «sin señal»: es la misma pregunta mal repartida entre clases.
- La cifra que ya se publica para los TOIs, P(planeta | señal), es la segunda pregunta. El diseño debería separarlas del todo.

### M5 · La calibración no se transfiere (ALTA para la herramienta)

Temperatura óptima cuando el modelo **no** ha visto la misión: Kepler 0,89, K2 0,91, TESS 1,68. Dentro de misión: 0,77, 0,68 y 1,07.

- La temperatura global de 0,80, guardada como respaldo para telescopios nuevos, agudiza las probabilidades. En TESS sin verlo, el ECE pasa de 0,110 a 0,141.
- No se puede predecir desde las otras dos misiones: la temperatura aprendida de ellas empeora la calibración en las tres.
- P(planeta | señal) queda desplazada en una misión no vista: TESS 0,52 de media frente a 0,39 real; Kepler 0,42 frente a 0,33; K2 0,40 frente a 0,51.
- **Consecuencia:** para un telescopio nuevo la herramienta debe dar una puntuación para ordenar, no una probabilidad.

### M6 · Lo que sigue pendiente de la auditoría anterior

- **Cerrado:** C1, C2, C3, A6, paso 1 de P1, #85 y #87.
- **A medias:** A1 (M3 aporta evidencia; falta el modelo final) y A7 (se publica P(planeta | señal), sin calibración propia).
- **Sin hacer:** A2 (selección con una semilla; la desviación entre semillas en TESS es 0,013), A3, A4 (Kepler pesa el 75 % de la pérdida), A5, B1–B5, P6, P7, P8, P12–P17 y C4.
- **El test está gastado.** TESS tiene 147 señales de planeta o FP en test. Una v2 necesita una evaluación nueva e intacta.

## 5. Hallazgos de implementación

**I1 · No hay función de inferencia y el código de predicción está repetido (ALTA).** Cargar un modelo y predecir se repite en 7 scripts (`predict_tois.py`, `final_model.py`, `hard_negatives.py`, `eval_bls.py`, `tess_sectors.py`, `pseudo_labels.py`, `error_analysis_fp.py`). La temperatura está implementada dos veces (`predict_tois.temper` y `temperature.scale`). Para predecir hace falta un CSV de calidad, archivos `.npz` y una clase de relleno ([`predict_tois.py:67`](../scripts/models/predict_tois.py)).

**I2 · Los scripts se importan entre sí (MEDIA).** `AGENTS.md` dice que no, pero 7 lo hacen (y otros 2 importan de `legacy/`): `build_series`, `build_bls_peaks` y `build_transit_feats` importan `build_bls_views`; `hard_negatives` y `tess_multisector` importan `final_model`; `temperature` importa `calibration`; `series_ensemble` importa `triage_val`. Obliga a trucos con `PYTHONPATH`. Esa lógica compartida es justo la que necesita la aplicación y debe estar en `src/`.

**I3 · No queda registrado qué archivos usó cada vista (ALTA).** «Qué FITS usa esta estrella» se recalcula en 5 sitios con fuentes distintas: la tabla de MAST (`curves.download_star`), las cabeceras (`build_negatives`, `build_centroid`, `toi_ephemeris_drift`) y patrones de nombre (`build_bls_views.star_files`, `build_cropped`). El fallo de la campaña 10 de K2 (44 estrellas) salió de ahí. Falta guardar, al plegar, la procedencia de cada vista: archivos, sectores, cadencia y versión del código.

**I4 · La normalización es frágil (MEDIA).** [`views.normalize`](../src/exoplanet/views.py) divide por `mediana − mínimo`, y el mínimo es un solo bin. Con una curva plana da infinitos. Un test exige que el mínimo valga exactamente −1.

**I5 · La red conserva una entrada muerta (MEDIA).** El modelo final recibe `log_depth` a cero, y sus pesos guardan desviación 1e-6. Si alguien pasa la profundidad real, se multiplica por un millón y la predicción es basura sin ningún aviso. Además, el 43 % de los parámetros están en una sola capa densa (2 177 → 256).

**I6 · El plegado supone la cadencia de 2 min (MEDIA).** La vista local tiene 81 bins fijos. Con datos de FFI de 30 min y pocos tránsitos caben ~2 puntos por bin, y un tránsito de menos de 1 h se declara no válido. `SEARCH` solo conoce SPOC de 2 min. El modelo no sabe cuántos puntos hay detrás de cada bin.

**I7 · Al predecir se usa 1 sector por una razón que solo vale al entrenar (MEDIA).** `MAX_SEGMENTS = 1` evita el atajo del número de sectores en el entrenamiento. En inferencia no hay atajo que evitar, y M3 indica que se pierde rendimiento.

**I8 · El entrenamiento no deja rastro de sus datos (MEDIA).** Los pesos guardan configuración y semilla, pero no el commit, ni la lista de estrellas, ni el hash de las vistas. El manifiesto de la v1 fija los pesos, no con qué se entrenaron. Los hiperparámetros son constantes repartidas por los scripts.

**I9 · El paquete solo funciona dentro del repositorio (MEDIA).** `DATA` se calcula desde la ruta del archivo y `views.QC` es relativo al directorio actual: instalado con `pip` fallaría. `pyproject.toml` no declara dependencias, y `astroquery`, `requests` y `streamlit` están sin fijar. Importar el paquete cambia la codificación de la salida estándar.

**I10 · Tests: buenos donde existen, ausentes en lo que se va a publicar (MEDIA).** Cubren bien el plegado, los folds, las etiquetas y los negativos. No hay ninguno de entrenamiento, predicción, temperatura, BLS ni de la regla de elección de sector que fijó la v1. Los 3 tests con curvas reales se saltan sin avisar: falta el FITS del sector 1 de TIC 261136679. Los tests en sí tardan 2 s; el resto (1 min con el disco caliente, 4,5 min en frío) es cargar librerías pesadas y recorrer `data/raw_lc` al importar `test_folding_real.py`.

**I11 · La inferencia depende del dispositivo (MEDIA; verificado).** La v1 se predijo en GPU con bf16. En CPU con fp32, 12 de los 3 536 TOIs cambian de clase y la probabilidad difiere hasta en 0,014 (mediana 0,0008). La v1 congelada sigue siendo válida como registro, pero un tercero no puede reproducirla exactamente. La referencia debería ser CPU y fp32.

**I12 · Documentación y orden (BAJA).**
- `research/` son ahora guías de redacción; las tablas técnicas originales solo están en git. `docs/decisiones.md` (#40) no existe.
- [`README.md:50`](../README.md) dice que el test no se ha usado; `CONTEXTO.md` cuenta 41 tests (son 46); [`build_toi_views.py:105`](../scripts/data/build_toi_views.py) habla de 50 GB.
- Sin `LICENSE`, `CITATION.cff`, CI ni Docker. Dos PDF sin versionar.
- `data/models` guarda 78 archivos (496 MB) de experimentos sin índice.

**Lo que está bien y hay que conservar:** `fold_views` es una función pura con un test que le prohíbe recibir catálogo o misión; la escritura atómica y los scripts reanudables; los negativos reproducibles byte a byte; el aviso de deriva de la efeméride; y que cada script explique en su cabecera qué decisión lo motivó.

## 6. Hallazgos de utilidad (actualizados)

- **U1 · Sin camino de inferencia.** Ver I1.
- **U2 · El modelo no se puede obtener ni reutilizar.** Pesos solo en disco (5 archivos de 5,2 MB), sin licencia ni DOI.
- **U3 · Cobertura de datos.** De 4 813 TOIs abiertos, el 27 % queda sin respuesta. Por origen se puntúan el 81 % de los de SPOC, el 73 % de los de QLP y solo el 25 % de los de QLP de los sectores ≥ 70. De los 4 737 procesados, 3 591 vienen de QLP.
- **U4 · Clasifica, no detecta.** Necesita la efeméride.
- **U5 · Rendimiento en TESS.** Ver M2: el límite es de información, no de arquitectura.
- **U6 · No sabe decir «no lo sé».** Ver M5: con un telescopio nuevo, además, la probabilidad no es fiable.
- **U7 · Servirlo es barato (verificado).** 1 290 595 parámetros por modelo. En CPU: 3 536 TOIs en 1,7 s; un candidato, 7 ms de plegado y 7 ms de predicción. No hace falta GPU.
- **U8 · No hay ningún usuario.** La propuesta a DIPS Ojos (#105) sigue en stand-by.

## 7. Estado del arte: qué ha cambiado

| Trabajo | Qué es | Qué implica aquí |
|---|---|---|
| ExoMiner++ 2.0 (enero de 2026) | Extiende ExoMiner++ a las señales de FFI de SPOC; PR-AUC ≈ 0,95; usa imagen de diferencia, centroide, parámetros estelares y estrellas vecinas | Corrige lo que dije esta mañana: las FFI de SPOC ya están cubiertas. Quedan fuera los candidatos sin señal de SPOC (solo QLP), los de la comunidad, K2 y otros telescopios |
| WATSON-Net (2025) | Red entrenada en Kepler y probada en TESS sin ajuste: precisión 0,93, recall 0,76. Código abierto, dentro de la tubería SHERLOCK | Ya existe una evaluación Kepler → TESS. La afirmación de `research/08` hay que matizarla: lo propio es el LOMO con tres telescopios, solo curva y con protocolo anidado |
| DART-Vetter (2025) | CNN solo con la curva plegada, entrenada con TESS y Kepler; recall 91 % | «Solo curva» tampoco es exclusivo |
| LEO-Vetter y tess-vetter | Comprobaciones automáticas de flujo y de píxeles para TESS, instalables con `pip` | La parte de píxeles no hay que construirla: se integra. Ojo a las licencias (LEO-Vetter es GPLv3) |
| PLATO | Según la página de estado de la ESA, el lanzamiento pasa a enero–marzo de 2027 | Los datos públicos tardarán; a corto plazo, solo simulaciones |

De ExoMiner++ 2.0 he leído el resumen del artículo; el enlace al catálogo (Zenodo 10.5281/zenodo.17707413) y al código salen de un resumen automático y hay que verificarlos. De WATSON-Net y DART-Vetter, solo el resumen.

**Dónde queda el proyecto.** No en ganar a ExoMiner++ donde llega, sino en tres cosas: (1) primer filtro para candidatos que no tienen los productos de SPOC; (2) validación prospectiva pública frente a ExoMiner++ y el TFOP; (3) informe didáctico en español que reúna en una página lo que dicen todos. Cuánto mide el hueco (1) hay que medirlo cruzando los TOIs abiertos con el catálogo 2.0 (R17).

## 8. Sobre el objetivo

«Que sirva para toda la detección de exoplanetas» no es alcanzable tal como está formulado: detectar son tres trabajos (buscar, clasificar, confirmar) y el modelo hace el segundo. Con lo visto en M2 se puede precisar más: dentro de clasificar, el modelo es bueno en el **primer filtro** y débil en el **juicio final**. El objetivo realista es *un primer filtro portable y honesto para cualquier curva con su efeméride, que dice cuándo no puede decidir y qué información haría falta*.

## 9. Ruta y decisiones

Están desarrolladas en el [documento de diseño](diseno-futuro-2026-11-11.md), secciones 4 y 5. En resumen: primero el núcleo de inferencia y las mejoras que no exigen reentrenar (v1.1), después la ficha de candidato y el catálogo, y solo entonces el modelo v2.

## 10. Qué haría distinto un equipo real

- Definiría al usuario antes que el modelo; aquí la herramienta llega en la fase 8.
- Separaría desde el principio el código de experimentos del código que se publica, con la inferencia probada en CI desde el primer modelo.
- Guardaría la procedencia de cada dato y de cada entrenamiento de forma automática.
- Reservaría un segundo test sin tocar para las versiones siguientes.
- Calibraría y evaluaría en la población donde se va a usar el modelo, no en la de entrenamiento.
- Revisaría el estado del arte cada pocos meses: en este campo, un año cambia el mapa.

## Fuentes

- ExoMiner++ 2.0: https://arxiv.org/abs/2601.14877
- WATSON-Net: https://arxiv.org/abs/2511.08768
- DART-Vetter: https://arxiv.org/abs/2506.05556
- LEO-Vetter: https://arxiv.org/abs/2509.10619 · https://pypi.org/project/leo-vetter/
- tess-vetter: https://pypi.org/project/tess-vetter/
- PLATO, estado de la misión: https://www.cosmos.esa.int/web/plato/status-and-history
- Cifras del proyecto: `results/models/*.csv`, `results/toi/*.csv` y `results/sample/qc.csv`, recalculadas hoy.
