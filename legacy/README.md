# legacy/

Código del primer camino del proyecto, que usa las vistas plegadas del dataset de Hugging Face (la vista global no está centrada) y variables de catálogo. Se conserva como referencia y para reutilizar ideas; **no forma parte del pipeline actual** (vistas propias de MAST: `src/exoplanet/` + `scripts/pilot_fold.py`).

Se ejecuta desde la raíz con `PYTHONPATH=src`, p. ej. `python legacy/eda.py`. Los resultados siguen en `results/`.

| Archivo | Qué hacía | Cuándo se rehace |
|---|---|---|
| `features.py`, `eda.py` | variables derivadas del dataset y EDA | etapa 1: `features.py` nuevo sobre las vistas propias (#46) |
| `mission_leak.py` | la misión se adivina desde la curva | repetir sobre las vistas propias (#44) |
| `transfer_k2.py` | transferencia hacia K2 con catálogo (solo referencia) | no se rehace |
| `preprocess.py`, `app.py` | tablas procesadas y Streamlit sobre ellas | Streamlit nuevo tras los modelos base (#20) |
