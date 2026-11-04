# exoplanet-ml

Clasificador de señales de tránsito en tres clases (**PLANET**, **FALSE_POSITIVE**, **NO_SIGNAL**) usando **solo la curva de luz**, pensado para ser **portable entre telescopios** (Kepler, K2 y TESS; CoRoT como prueba futura). Proyecto de Ciencia de Datos.

## Estado (2026-11-04)

- **Fase 1, datos (cerrada):** curvas propias descargadas de MAST y plegadas con el mismo código en las tres misiones; negativos NO_SIGNAL fabricados con estrellas reales sin candidato, emparejadas por sector y brillo; tests de atajo y de fuga. Muestra completa: 17 164 estrellas (~14 700 con vista válida).
- **Fase 2, modelos dentro de cada misión (casi cerrada):** dummy, regresión logística, kNN, gradient boosting, MLP, CNN 1D de dos ramas (AstroNet) y MOMENT (modelo preentrenado de Hugging Face). Mejor modelo solo curva: **CNN con ensemble de 3 semillas, macro-F1 0,85 en val** (Kepler 0,87, K2 0,90, TESS 0,78). Con el centroide como canal extra, Kepler sube a 0,89. El análisis de errores muestra que el techo lo ponen falsos positivos invisibles en la curva (luz de estrellas vecinas).
- **Fase 3, transferencia entre telescopios (LOMO), cerrada:** entrenar con dos telescopios y evaluar en el tercero sin haberlo visto. PR-AUC Kepler 0,894, K2 0,927, TESS 0,764 (media 0,862 frente a 0,911 dentro de misión): el modelo solo curva es portable. TESS es el difícil por falta de información por estrella: con 2 sectores sube +0,034 (IC [+0,018, +0,048]); AdaBN, recortar Kepler/K2 y pseudo-etiquetas no ayudan. Regla de decisión en un telescopio nuevo: argmax sin corregir el prior. Detalle en [`research/07-fase3-lomo.md`](research/07-fase3-lomo.md).

Detalle y decisiones razonadas en [`research/`](research/); resumen en [`research/06-modelos-fase2.md`](research/06-modelos-fase2.md).

## Estructura

```
src/exoplanet/     paquete: datos, etiquetas, curvas y plegado, centroide, diagnósticos, dataset, modelos, evaluación
scripts/data/      pipeline de datos (descarga, plegado, negativos, centroide)
scripts/checks/    diagnósticos que solo escriben informes (atajos, fugas, bordes, análisis de errores)
scripts/models/    modelos de la fase 2 (sklearn, MLP, CNN, MOMENT)
tests/             pytest
research/          decisiones y resultados razonados (EDA, protocolo, etiquetas, negativos, modelos)
results/           salidas pequeñas (CSV, figuras); results/sample/history: versiones intermedias
legacy/            primer enfoque con las vistas del dataset de Hugging Face (solo referencia)
data/              datos y modelos grandes (no se suben a git)
```

## Entornos

- **Windows** (`.venv`, Python 3.12, [`requirements.txt`](requirements.txt)): pipeline de datos, comprobaciones y sklearn. PyTorch no carga en Windows con Control inteligente de aplicaciones activado.
- **WSL2 Ubuntu 24.04** ([`requirements-wsl.txt`](requirements-wsl.txt)): redes neuronales. `~/venvs/exoplanet` (CPU) y `~/venvs/exoplanet-rocm` (GPU AMD RX 9070 XT con ROCm 7.2.1; `export HSA_ENABLE_DXG_DETECTION=1`; entrenar en bf16 con `--amp`, porque fp32 va ~100 veces más lento en esta GPU).

## Reproducir (desde la raíz del repo, `PYTHONPATH=src`)

1. Dataset de partida: `data/{train,val,test}.parquet` (Hugging Face `bingbangboom/exoplanet-transit-detection`).
2. `python scripts/data/fetch_labels.py`: disposiciones de TOI, K2 y KOI.
3. `python scripts/data/build_sample.py --kepler 100000 --tess 100000 --workers 4`: descarga y pliega los positivos con etiqueta fuerte (~16 GB, horas; se puede reanudar).
4. `python scripts/data/build_negatives.py` y `python scripts/data/build_sample.py --kepler 100000 --tess 100000 --negatives`: negativos 1:1.
5. `python scripts/data/build_centroid.py`: vistas de centroide (no descarga nada).
6. Comprobaciones: `scripts/checks/shortcut_negatives.py`, `shortcut_centroid.py`, `check_view_edges.py`.
7. Modelos: `scripts/models/dummy_sample.py`, `sklearn_models.py` (Windows); `mlp.py`, `cnn.py [--amp] [--centroid] [--modes conjunta]` (WSL).
8. Tests: `python -m pytest -q` (los del dataset de PyTorch, en WSL).

Todo usa semillas fijas; `build_negatives.py` da el mismo CSV byte a byte en dos ejecuciones. El conjunto de test no se ha usado todavía.
