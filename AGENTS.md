# AGENTS.md: contexto para agentes de código (Codex y similares)

Léelo entero antes de tocar nada. Para el detalle completo está `CONTEXTO.md`; las reglas de trabajo largas están en `CLAUDE.md`; las decisiones razonadas, en `research/`.

## Qué es el proyecto

Clasificador de tránsitos como **PLANET / FALSE_POSITIVE / NO_SIGNAL** usando **solo la curva de luz**, pensado para ser **portable entre misiones** (Kepler, K2, TESS; CoRoT bloqueada hasta tener su catálogo). Es un proyecto de Data Science de un estudiante: lo importante es que él entienda lo que se hace.

## Cómo trabajar (reglas que importan)

- Responde en **español**, con explicaciones sencillas. El usuario dirige y tú implementas.
- Si una instrucción es mala (fuga de datos, métrica incorrecta, complejidad injustificada), dilo y propón alternativa. **Las decisiones importantes de ML son del usuario**: propón con pros y contras y espera.
- Cambios mínimos y legibles; sin abstracciones ni dependencias nuevas sin justificarlas. No instales paquetes sin preguntar.
- Nada de fugas: el preprocesado se ajusta solo con entrenamiento, y la misión excluida (test) no participa en ninguna decisión.
- Nunca digas que algo funciona sin haberlo ejecutado. Ejecuta los tests antes de dar por bueno un cambio.
- No lances ejecuciones largas o descargas grandes sin comprobar la configuración antes (presupuesto de descargas acumulado: 50 GB).
- **El repositorio es público.** Nunca subas claves, tokens ni credenciales; el `.gitignore` ya cubre `.env`, `secrets.toml` y similares. `data/` está ignorado y no se sube. No hagas commit ni push sin que el usuario lo pida.

## Estado (2026-11-06)

Fases 1–6 cerradas. Modelo final (fase 4): CNN de dos ramas solo curva, sin profundidad, Kepler + K2 + TESS, conjunto de
5 semillas y temperatura por misión (`data/models/final_seed*.pt`, `final_temperature.json`). Test (gastado, una vez):
macro-F1 Kepler 0,886, K2 0,928, TESS 0,720; LOMO en test PR-AUC 0,906 / 0,961 / 0,761. Fase 7 (investigación) en curso:
el paso 1 del Transformer de tránsitos individuales no cumplió su criterio (`research/15`). Fases 8 (herramienta) y 9 (web)
pendientes. Auditoría y propuestas: `docs/auditoria-y-propuestas-2026-11-06.pdf`. Detalle en `CONTEXTO.md` y `research/`.

## Estructura

- `src/exoplanet/`: paquete activo (datos, etiquetas, curvas y plegado, centroide, diagnósticos, dataset, CNN, entrenamiento, evaluación). Tabla de módulos en `CONTEXTO.md`.
- `scripts/data/`: pipeline de datos. `scripts/checks/`: diagnósticos que solo escriben informes. `scripts/models/`: modelos. Los scripts no se importan entre sí: lo compartido va en `src/exoplanet/`.
- `legacy/`: primer camino con las vistas del dataset de Hugging Face. **No es el pipeline actual.**
- `research/`: decisiones y resultados razonados. `results/`: salidas pequeñas (`results/sample/history/`: versiones intermedias). `tests/`: pruebas.

## Cómo ejecutar

Windows (`.venv`, `requirements.txt`): pipeline de datos y sklearn.
```
$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python scripts/models/sklearn_models.py
.venv\Scripts\python -m pytest -q
```
WSL2 (`requirements-wsl.txt`): redes. PyTorch no carga en Windows. GPU AMD: `export HSA_ENABLE_DXG_DETECTION=1` y `--amp` (bf16).
```
PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/cnn.py --amp --modes conjunta
```

## Decisiones ya tomadas (no las reabras sin hablar con el usuario)

- **Métrica:** macro-F1 de 3 clases por misión y media de misiones; PR-AUC secundaria; IC bootstrap; selección en val; test sin tocar.
- **Protocolo LOMO anidado:** la misión excluida se evalúa una vez, entera; el fold interno es train de A → val de B, con PR-AUC; las estrellas compartidas entre misiones se quitan del ajuste.
- **Evaluación solo con etiquetas fuertes;** las débiles solo como variante de entrenamiento.
- **Entrada solo de la curva:** nunca metadatos del proceso (n_points, data_days, cadence, n_transits, frac_empty, n_files) ni catálogo. El centroide es una variante aparte.
- **Negativos NO_SIGNAL fabricados** con estrellas reales sin candidato, emparejadas por sector y brillo, y lejos (> 2″) de cualquier candidato o estrella del dataset.

## Cosas que ya salieron mal (no las repitas)

- La vista global y las de par/impar del dataset de Hugging Face **no están centradas** en el tránsito.
- La época del dataset (`epoch_bjd`) ya está en el tiempo de la misión; un error MJD→BKJD costó 400 días.
- Los trimestres de Kepler salen de `kepler_quarters.csv` (MAST da el mismo intervalo a todos).
- TESS tiene sectores >= 100: solo K2 tiene mitades de campaña (`mast_sequence(segment, mission)`). En disco, la C10 de K2
  está en la carpeta `ktwo...-c10_lc` con el archivo `...-c102_llc.fits` (segmento 102).
- La misma estrella tiene un número distinto en cada misión (KIC, EPIC, TIC): cruza por coordenadas.
- **Antes de borrar datos**, cuenta qué vas a borrar comparando por `star_id` y para si no es lo esperado.
- MAST puede colgarse sin límite de tiempo: vigila que el log avance.
- Con muchos archivos, ábrelos y ciérralos de uno en uno (`load_views`).

## Gestión de tareas

Las tareas viven en Notion (página «Exoplanet ML — Gestión del proyecto», enlazada en `CLAUDE.md`). Si no tienes acceso a Notion, no inventes estados: deja lo hecho descrito en el mensaje al usuario.
