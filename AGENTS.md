# AGENTS.md: contexto para agentes de código (Codex y similares)

Léelo entero antes de tocar nada. Para el detalle completo está `CONTEXTO.md`; las reglas de trabajo largas están en `CLAUDE.md`; las decisiones razonadas, en `research/`.

## Qué es el proyecto

Clasificador de tránsitos como **PLANET / FALSE_POSITIVE / NO_SIGNAL** usando **solo la curva de luz**, pensado para ser **portable entre misiones** (Kepler, K2, TESS; CoRoT queda congelada en P3 hasta tener su catálogo). Es un proyecto de Data Science de un estudiante: lo importante es que él entienda lo que se hace.

## Cómo trabajar (reglas que importan)

- Responde en **español**, con explicaciones sencillas. El usuario dirige y tú implementas.
- Si una instrucción es mala (fuga de datos, métrica incorrecta, complejidad injustificada), dilo y propón alternativa. **Las decisiones importantes de ML son del usuario**: propón con pros y contras y espera.
- Cambios mínimos y legibles; sin abstracciones ni dependencias nuevas sin justificarlas. No instales paquetes sin preguntar.
- Nada de fugas: el preprocesado se ajusta solo con entrenamiento, y la misión excluida (test) no participa en ninguna decisión.
- Nunca digas que algo funciona sin haberlo ejecutado. Ejecuta `py -m pytest -q` antes de dar por bueno un cambio.
- No lances ejecuciones largas o descargas grandes sin comprobar la configuración antes (presupuesto de descargas acumulado: 50 GB).
- **El repositorio es público.** Nunca subas claves, tokens ni credenciales; el `.gitignore` ya cubre `.env`, `secrets.toml` y similares. `data/` está ignorado y no se sube. No hagas commit ni push sin que el usuario lo pida.

## Plan por etapas (acordado el 2026-10-30)

Se va paso a paso, sin abrir frentes en paralelo:

1. Un modelo en una misión: Kepler, PLANET frente a FALSE_POSITIVE, vistas propias, regresión logística.
2. El mismo modelo entrenado en Kepler y probado en K2 o TESS (primera medida de portabilidad).
3. Añadir un árbol y comparar; después, el protocolo con las tres misiones.
4. Más adelante: NO_SIGNAL, deep learning (MLP, CNN 1D), Streamlit y web/móvil.

Desde la etapa 1 se mantienen dos reglas: entradas **solo de la curva** (nada de catálogo ni `log_period`) y el plegado propio común a todas las misiones.

## Estructura

- `src/exoplanet/`: paquete activo. `data.py` (carga y limpieza), `labels.py` (fuerza de etiqueta: fuerte/débil/excluida), `curves.py` (descarga y preparación de curvas de MAST), `folding.py` (`fold_views`: vistas local de 81 bins y global, impar y par de 201), `evaluation.py` (protocolo LOMO anidado).
- `scripts/data/`: pipeline de datos (`fetch_labels`, `build_kepler_quarters`, `pilot_fold`, `build_sample`, `build_negatives`).
- `scripts/checks/`: diagnósticos que solo escriben informes (`label_audit`, `check_view_edges`, `shortcut_negatives`, `mission_leak_views`, `recheck_z_features`).
- `scripts/models/`: modelos (`baseline`). Los scripts no se importan entre sí: lo compartido va en `src/exoplanet/`.
- `legacy/`: primer camino, con las vistas del dataset de Hugging Face. **No es el pipeline actual**; se conserva como referencia (ver `legacy/README.md`).
- `research/`: decisiones razonadas (EDA, protocolo, armonización de etiquetas). `results/`: salidas. `tests/`: pruebas.

## Cómo ejecutar (Windows, PowerShell, Python 3.14 con `py`)

```powershell
$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'; py scripts/data/pilot_fold.py --n 100
py -m pytest -q
```

Datos de partida: `data/{train,val,test}.parquet` (Hugging Face `bingbangboom/exoplanet-transit-detection`, splits por estrella), etiquetas externas en `data/labels/`, curvas de MAST en `data/raw_lc/`.

## Decisiones ya tomadas (no las reabras sin hablar con el usuario)

- **Protocolo LOMO anidado:** la misión excluida se evalúa una vez, entera; el fold interno es una transferencia entre las otras dos y el criterio es PR-AUC.
- **Sin umbral** en la misión excluida: se informan PR-AUC y ROC-AUC; el umbral va aparte con calibración.
- **Evaluación solo con etiquetas fuertes;** las débiles solo como variante de entrenamiento.
- **Preprocesado propio:** curvas PDCSAP de MAST, aplanado con el tránsito enmascarado, tope de ~100 días por estrella.
- **Negativos NO_SIGNAL:** las vistas de NO_SIGNAL del dataset son relleno a cero; los negativos reales se fabrican plegando estrellas sin candidato (tarea #32, etapa posterior).
- **Ablación** de variables que delatan la misión (profundidad, SNR), pospuesta.

## Cosas que ya salieron mal (no las repitas)

- La vista global y las de par/impar del dataset de Hugging Face **no están centradas** en el tránsito.
- Conversión de tiempos: la época del dataset (`epoch_bjd`) ya está en el sistema de tiempo de la misión; un error previo MJD→BKJD costó 400 días.
- Para los trimestres de Kepler se usa `kepler_quarters.csv` (MAST da el mismo intervalo a todos).
- No busques archivos de TESS por la marca de tiempo del sector: la comparten muchas estrellas; incluye el identificador.

## Gestión de tareas

Las tareas viven en Notion (página «Exoplanet ML — Gestión del proyecto», enlazada en `CLAUDE.md`). Si no tienes acceso a Notion, no inventes estados: deja lo hecho descrito en el mensaje al usuario.
