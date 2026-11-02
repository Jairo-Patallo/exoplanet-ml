"""Vistas plegadas de la muestra A: carga y normalización por estrella (fase 2).

Lo usan igual los modelos de sklearn (scripts/models/sklearn_models.py) y el dataset de PyTorch (exoplanet.dataset),
para que todos los modelos vean exactamente la misma entrada.
"""
import numpy as np
import pandas as pd

from exoplanet.data import DATA

SAMPLE_VIEWS = DATA / "sample" / "views"
QC = "results/sample/qc.csv"


def load_sample(splits=("train", "val")) -> pd.DataFrame:
    """Estrellas de la muestra A con vista válida en esos splits (por defecto sin test), una fila por estrella."""
    qc = pd.read_csv(QC)
    return qc[qc["valid"].eq(True) & qc.split.isin(splits)].reset_index(drop=True)   # NaN (error de descarga) = no válida


def load_views(star_ids, keys=("local", "glob")) -> dict[str, np.ndarray]:
    """Vistas apiladas por clave: {"local": (n, 81), "glob": (n, 201), ...} en el orden de star_ids."""
    files = [np.load(SAMPLE_VIEWS / f"{s}.npz") for s in star_ids]
    return {k: np.stack([f[k] for f in files]) for k in keys}


def normalize(local: np.ndarray, glob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalización por estrella al estilo AstroNet (decisión 2026-11-02): se resta la mediana de cada vista y se divide
    por la profundidad de la local (mediana - mínimo), así la caída vale ~-1 y el modelo ve la forma. Devuelve también
    log10 de esa profundidad (n, 1) para no perder la escala. Solo usa la propia estrella: no hay fuga."""
    med = np.median(local, axis=1, keepdims=True)
    depth = med - local.min(axis=1, keepdims=True)
    return (local - med) / depth, (glob - np.median(glob, axis=1, keepdims=True)) / depth, np.log10(depth)
