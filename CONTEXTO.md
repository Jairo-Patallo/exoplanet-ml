# Contexto del proyecto exoplanet-ml (para pasárselo a otro chat)

Actualizado el 2026-10-30. Lee también `CLAUDE.md` (reglas de trabajo) y `research/` (decisiones razonadas). Este documento resume lo que otro chat necesita para continuar sin perder nada.

## 1. Objetivo

Un modelo que decide si un tránsito es **PLANET**, **FALSE_POSITIVE** o **NO_SIGNAL** usando **solo la curva de luz** (y, opcionalmente, variables derivadas de ella), **portable a telescopios nuevos**. Se valida entrenando con dos misiones y probando en la tercera (Kepler, K2, TESS); CoRoT sería una cuarta misión si se consiguen sus etiquetas. La web es un extra lejano (idea: BLS para obtener periodo y época a partir de una curva cruda).

Escalera de modelos, todos con el mismo protocolo: reglas físicas → regresión logística → árboles (sklearn, sobre variables derivadas de la curva) → MLP → CNN 1D. Las variables de catálogo (`teff`, `logg`, `log_rp_implied`) solo sirven como comparación; `log_period` queda fuera del conjunto portable.

## 2. Cómo se trabaja (resumen de `CLAUDE.md` y de lo acordado)

- Responder en español. El usuario dirige y Claude implementa; Claude evalúa cada instrucción (si está mal, lo dice y propone alternativa) y explica el código y los resultados.
- Las decisiones importantes de ML (misión reservada para test, umbrales, armonización de etiquetas…) son del usuario: Claude propone con pros y contras y espera.
- Las tareas viven en Notion. Estados: No empezado → En proceso → **Esperando decisión** → Hecho. Máximo 1-2 En proceso. Una tarea solo es Hecho si existe un artefacto verificable (commit, ruta o test que pasa).
- **Autonomía acordada:** ejecutar tareas del tablero y dejar en Notion el estado, los archivos y las métricas. Descargas: hasta **50 GB acumulados** sin preguntar (llevan unos 70 MB en disco). Seguir preguntando antes de: decisiones de ML con alternativas reales, instalar paquetes, commits/push (salvo petición explícita), borrar tareas y cambiar prioridades de secciones existentes.
- Notas del usuario: van en la columna "Nota de Jairo" de cada tarea y de cada bloque (ya no hay base de notas aparte). Antes de cada tarea y al empezar cada sesión, buscar filas con la nota rellena y "Nota revisada" sin marcar, también en tareas Hechas; responder en "Respuesta de Claude" y marcar la casilla.
- Bloques: el tablero se reorganizó en 7 bloques grandes. **El usuario decide qué bloque se hace primero**; Claude solo recomienda y no empieza otro sin su visto bueno.
- Explicar al usuario lo que se hace en cada paso.

## 3. Datos y entorno

- Dataset de partida: Hugging Face `bingbangboom/exoplanet-transit-detection` en `data/{train,val,test}.parquet` (carpeta `data/` ignorada por git). 23 567 estrellas: Kepler 16 054 (7 840 NO_SIGNAL), K2 1 485, TESS 6 028. Splits por estrella. Trae vistas plegadas (`flux_local` 81, `flux_global`/`flux_odd`/`flux_even` 201) y `flux_raw` (20 000 puntos, **sin tiempo**).
- `data/labels/` (tablas TOI y K2 del NASA Exoplanet Archive, `scripts/fetch_labels.py`), `data/raw_lc/` (curvas de MAST), `data/corot/` (`scripts/` con 16 scripts wget del NASA para 97 971 curvas de solo 10 de 23 runs, y `probe/` con 5 curvas de prueba).
- Windows 11, PowerShell. Python 3.14 con `py`. Para los scripts: `$env:PYTHONPATH='src'; py scripts/xxx.py` desde la raíz; `py -m pytest -q` ya incluye `pythonpath = ["src"]`. Para salida con acentos: `$env:PYTHONIOENCODING='utf-8'`. Dependencias nuevas añadidas: `astropy`, `lightkurve`, `pytest`.

## 4. Código (`src/exoplanet/`)

| Módulo | Qué hace |
|---|---|
| `data.py` | Carga por misión y split (`load_split`), reglas de limpieza de anomalías |
| `features.py` | Variables derivadas. Ojo: `odd_even_z` y `secondary_z` usan la vista global del dataset, que **no está centrada** (ver §6) |
| `evaluation.py` | Protocolo anidado: `nested_folds`, `in_domain_split`, `shared_stars` (cruce por coordenadas < 2") |
| `labels.py` | Fuerza de cada etiqueta (fuerte / débil / excluida) según la disposición original |
| `folding.py` | `fold_views(time, flux, period, t0, duration)`: función pura, sin catálogo ni misión. Vistas con el tránsito en el bin 40 (local) y 100 (global) más variables de calidad |
| `curves.py` | Curvas de MAST: PDCSAP, aplanado con el tránsito enmascarado, normalización, tope de ~100 d por estrella eligiendo los segmentos con más tránsitos |
| `kepler_quarters.csv` | Intervalo real (BKJD) de cada trimestre de Kepler, medido en cabeceras (MAST da el mismo intervalo a todos) |

Scripts: `baseline.py` (dummy), `mission_leak.py`, `fetch_labels.py`, `label_audit.py`, `build_kepler_quarters.py`, `transfer_k2.py` (**solo referencia con catálogo, no portable**), `eda.py`, `preprocess.py`. **26 tests** en `tests/` (todos pasan).

Documentos: `research/03-protocolo-evaluacion.md` (protocolo, ablación de variables y negativos NO_SIGNAL), `research/04-armonizacion-etiquetas.md` (mapeo de etiquetas), `research/02-eda.md`.

## 5. Decisiones tomadas

1. **Protocolo LOMO anidado** (`03-protocolo-evaluacion.md`): la misión excluida no participa en ninguna decisión y se evalúa una vez, entera; el fold interno es una transferencia entre las otras dos misiones, con PR-AUC como criterio. Pendiente: regla del umbral en una misión sin etiquetas.
2. **Tres clases** con negativos reales: las vistas plegadas de NO_SIGNAL del dataset son relleno a cero. Negativos = estrellas sin candidato plegadas con periodo/época/duración al azar de la distribución de los positivos de su misión, **mismo método en las cuatro misiones**; test de atajo con variables de nuisance (puntos por bin, bins vacíos, ruido fuera del tránsito). El tránsito enmascarado queda como complemento opcional.
3. **Preprocesado propio** (opción B): rehacer las vistas de todas las misiones con `folding`/`curves` a partir de curvas con tiempo de MAST. Diseño D1-D9 aprobado, con tope de ~4 meses por estrella.
4. **Etiquetas:** evaluación solo con etiquetas **fuertes** (confirmados / FP claros); las débiles (candidatos) solo como variante de entrenamiento elegida en los folds internos. Mapeo y cifras en `04-armonizacion-etiquetas.md`.
5. **Ablación de variables que delatan la misión** (profundidad, SNR): variantes (a) todas, (b) sin profundidad ni SNR, (c) invariantes a la escala (forma normalizada, par/impar, cociente secundario/primario; sin Rp/R★ ni log SNR), (d) estandarización por misión no supervisada (se informa aparte). Selección entre a, b y c por PR-AUC macro en los folds internos. Escrita en el protocolo antes de ejecutarla.

## 6. Hallazgos que cambian cosas

- Las vistas plegadas de NO_SIGNAL del dataset son relleno a cero (784 de 784 en Kepler val).
- **La vista global y las de par/impar del dataset no están centradas en el tránsito** (el mínimo cae a ±3 bins del 100 en solo el 0 % de Kepler, 31 % de K2 y 10 % de TESS con planetas de > 3 000 ppm). La vista local sí está centrada y mi plegado la reproduce (profundidad central Kepler 0,98619 frente a 0,98633, TESS idéntica, K2 0,97785 frente a 0,97602). Las conclusiones del EDA que usan `odd_even_z`/`secondary_z` hay que revisarlas (tarea #46).
- **La misión se adivina** a partir de las variables de la curva (balanced accuracy ~0,70, azar 0,33; AUC 0,86-0,93), también dentro de cada clase; lo delatan `depth_local_ppm` y `snr_local`. Parte es física y de selección real.
- **Etiquetas muy desiguales:** PLANET fuerte vs débil: Kepler 1 943 / 1 631, TESS 674 / 3 802 (+438 APC excluidos), K2 346 / 899. La prevalencia de FP al quedarse con fuertes: Kepler 0,57→0,71, K2 0,16→0,41, TESS 0,18→0,59. Test fuerte dentro de dominio: Kepler 656, TESS 167, **K2 solo 62** estrellas (para K2 usar el fold externo).
- **Estrellas compartidas entre misiones:** 14 (Kepler-TESS) y 50 (K2-TESS); se excluyen del ajuste.
- Dummy (clase más frecuente), vetting con etiquetas fuertes, fold externo, macro-F1 / PR-AUC de FP: excluida Kepler 0,413 / 0,705; K2 0,291 / 0,410; TESS 0,371 / 0,590. Todo en `results/baseline/dummy.csv`.
- Errores propios ya corregidos (que otro chat no debe repetir): conversión MJD→BKJD con 400 días de error; usar `t_min/t_max` de MAST para los trimestres de Kepler (son los de toda la misión); la tarea #27 figuraba Hecha sin existir en el repo.
- Con el tope de 4 meses, los planetas de periodo largo (~200 d) quedan con 2 tránsitos y la vista global no es válida; es el coste aceptado.
- **CoRoT está bloqueado:** el catálogo de etiquetas (VizieR `J/A+A/619/A97`, Deleuil et al. 2018) solo está en el CDS, que no responde; probados CfA/Harvard, IUCAA, NAO, CADC, China-VO e INASAN. El usuario escribió al CDS/EDP Sciences. Las curvas existen en el NASA Exoplanet Archive (`EN2_STAR_CHR_*` y `EN2_STAR_MON_*`; los `AN2_*` son asterosismología).

## 7. Notion

- Página principal: https://app.notion.com/p/40715f33376c81ea9534c0e8bd2b267d
- Tareas: `collection://cf064316-c55a-46e0-a764-9985c623078f` · Bloques: `collection://a3702b25-45cb-47e0-860f-5837c94448bb`
- **7 bloques** (reorganización del 2026-10-30; el antiguo reparto en 11 fases quedó sustituido). Tareas por bloque (hechas/total): 1 · Datos, etiquetas y protocolo (13/15); 2 · Preprocesado propio y muestra (1/5); 3 · Variables y modelos clásicos (2/9); 4 · Deep learning: MLP y CNN (0/4); 5 · Evaluación final y robustez (0/5); 6 · CoRoT (0/1, esperando el catálogo); 7 · Documentación, web y calidad (1/8). Orden recomendado por Claude: 2 → 1 → 3 → 4 → 5 → 7, y el 6 cuando se desbloquee; el usuario decide.
- Tablero: 47 tareas, 17 Hechas, 1 Esperando decisión (#31 CoRoT), el resto No empezado, ninguna En proceso.
- Limpieza hecha el 2026-10-30 con confirmación del usuario: se borraron (a la papelera de Notion, restaurables) la base «Notas de Jairo», 4 bloques viejos vacíos y 13 vistas enlazadas de la página principal, que ahora muestra la tabla de Bloques y la de Tareas abiertas.

## 8. Qué sigue (orden recomendado)

1. **#47 Piloto de descarga y plegado** (~100 estrellas con etiqueta fuerte por misión, estratificado por clase): medir tamaño real, tiempo y porcentaje de vistas válidas. La estimación del techo (~57 GB para las 15 330 estrellas con efemérides, sobre todo TESS) es una extrapolación de solo 3 productos, no una medida a escala. Decidir con eso el tamaño de la muestra.
2. **#32 Negativos NO_SIGNAL** con el método único de las cuatro misiones (necesita bajar estrellas sin candidato de las mismas campañas y sectores).
3. **#44** repetir `mission_leak.py` sobre las vistas propias y **#46** revisar el EDA y las variables que dependían de la vista global.
4. **#43** ejecutar la ablación; luego modelos: Escalones 2-4 (sklearn, #11-#13), dataset PyTorch (#15), MLP (#29), CNN (#16), con métricas de #37 y evaluación final (#18).
5. Abiertos para el usuario: regla del umbral sin etiquetas; si bajar CoRoT a P3; la Fase 4 está en P1 pero sus tareas (#15, #16, #29) son P0; revisar si la tarea #10 (recortar bins de borde de `flux_local`) sigue haciendo falta con el preprocesado propio.

## 9. Trabajo previo útil

`transfer_k2.py` (experimentos A/B/C hacia K2) dio PR-AUC en K2: dummy 0,184, K2 solo 0,717, Kepler→K2 0,602, TESS→K2 0,324, Kepler+TESS→K2 0,478, pero usa variables de catálogo y vistas del dataset: es **referencia, no resultado portable** (`results/catalog_reference/`).
