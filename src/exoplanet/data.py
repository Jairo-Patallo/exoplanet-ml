"""Carga de los splits originales (train/val/test) filtrados por misión.

Los splits de HuggingFace ya están estratificados por misión y clase y no comparten estrellas,
así que no se vuelven a dividir: solo se filtra cada uno por misión al leerlo (sin copias en disco).

Uso:  python -m exoplanet.data   -> imprime recuentos por split, misión y clase y comprueba los splits
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"  # funciona desde cualquier directorio
SPLITS = ("train", "val", "test")
MISSIONS = ("kepler", "k2", "tess")
TARGET = "disposition"
CLASSES = ("PLANET", "FALSE_POSITIVE", "NO_SIGNAL")

# Columnas que NO pueden ser entradas de un modelo
LEAKAGE = ["label_source", "confidence_score", "kepoi_name", "disposition_disputed",
           "fp_flag_not_transit", "fp_flag_stellar", "fp_flag_centroid", "fp_flag_ephemeris"]  # derivan de la etiqueta
IDENTIFIERS = ["star_id", "ra", "dec"]  # identifican la estrella / su posición en el cielo
SHORTCUTS = ["mission", "cadence", "has_transit_params", "n_planets_in_system",  # NO_SIGNAL solo existe en Kepler
             "period_days", "epoch_bjd", "duration_hrs", "depth_ppm", "planet_radius_earth",  # 100 % NaN en NO_SIGNAL
             "flux_global_len", "flux_local_len", "flux_folded_len", "flux_raw_len"]  # constantes
# Ojo: los SHORTCUTS de tránsito solo son atajo en la tarea de 3 clases; en vetting (PLANET vs FP) sí son válidos.


def load_split(split: str, mission: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Lee data/<split>.parquet; si se indica `mission`, solo las filas de esa misión."""
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}, no {split!r}")
    if mission is not None and mission not in MISSIONS:
        raise ValueError(f"mission debe ser una de {MISSIONS}, no {mission!r}")
    filters = [("mission", "==", mission)] if mission else None
    return pd.read_parquet(DATA / f"{split}.parquet", columns=columns, filters=filters)


def load_vetting(split: str, missions: tuple[str, ...] = MISSIONS, columns: list[str] | None = None) -> pd.DataFrame:
    """Filas PLANET/FP de `missions` con vistas plegadas (las que no las tienen se excluyen del vetting)."""
    if split not in SPLITS or not set(missions) <= set(MISSIONS):
        raise ValueError(f"split o misiones no válidos: {split!r}, {missions!r}")
    cols = None if columns is None else list(dict.fromkeys(columns + ["mission", TARGET, "flux_global"]))
    filters = [("mission", "in", list(missions)), (TARGET, "!=", "NO_SIGNAL")]
    df = pd.read_parquet(DATA / f"{split}.parquet", columns=cols, filters=filters)
    df = df[df["flux_global"].map(lambda b: np.frombuffer(b, np.float32).any())].reset_index(drop=True)
    return df if columns is None or "flux_global" in columns else df.drop(columns="flux_global")


def lomo_folds(columns: list[str] | None = None):
    """Validación dejando una misión fuera (vetting). Por fold: (misión excluida, fit, tune, dev).

    fit = train de las otras dos, tune = val de las otras dos, dev = val de la excluida.
    El split test no se lee nunca aquí (ver research/03-protocolo-evaluacion.md).
    """
    for held_out in MISSIONS:
        rest = tuple(m for m in MISSIONS if m != held_out)
        yield (held_out, load_vetting("train", rest, columns), load_vetting("val", rest, columns),
               load_vetting("val", (held_out,), columns))


def main() -> None:
    cols = ["star_id", "mission", "disposition"]
    stars = {}
    for split in SPLITS:
        for mission in MISSIONS:
            df = load_split(split, mission, cols)
            assert (df["mission"] == mission).all()
            counts = df["disposition"].value_counts().to_dict()
            print(f"{split:5} {mission:6} n={len(df):6}  {counts}")
            stars.setdefault(split, set()).update(df["star_id"])

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = len(stars[a] & stars[b])
        assert shared == 0, f"{shared} estrellas compartidas entre {a} y {b}"
    print("OK: cada split contiene solo su misión y no hay estrellas compartidas entre splits")


if __name__ == "__main__":
    main()
