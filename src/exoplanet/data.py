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


RP_MAX = 22.0  # R_tierra (~2 R_Júpiter): un PLANET mayor es ruido de etiqueta o un radio estelar erróneo
RP_MIN = 0.3   # R_tierra: por debajo, el radio de catálogo no es creíble
CLEAN_COLS = ["disposition", "period_days", "duration_hrs", "depth_ppm", "planet_radius_earth"]


def anomaly_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Máscaras booleanas (True = fila anómala), una por regla. Los NaN (NO_SIGNAL) nunca se marcan.

    La regla del radio solo se aplica a PLANET: un FP gigante (binaria eclipsante) es justo lo que el modelo debe rechazar.
    Si `df` incluye `flux_global`, se marcan también las filas PLANET/FP con la vista plegada toda a cero.
    """
    masks = {
        "PLANET con Rp > RP_MAX": (df[TARGET] == "PLANET") & (df.planet_radius_earth > RP_MAX),
        "Rp <= RP_MIN": df.planet_radius_earth <= RP_MIN,
        "duración <= 0": df.duration_hrs <= 0,
        "profundidad >= 100 %": df.depth_ppm >= 1e6,
        "duración > periodo": df.duration_hrs / 24 > df.period_days,
    }
    if "flux_global" in df.columns:
        empty = df.flux_global.map(lambda b: not np.frombuffer(b, np.float32).any())
        masks["vista plegada toda a cero"] = empty & (df[TARGET] != "NO_SIGNAL")
    return masks


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas anómalas (ver `anomaly_masks`). `df` debe incluir CLEAN_COLS."""
    drop = pd.concat(anomaly_masks(df).values(), axis=1).any(axis=1)
    return df[~drop]


def cleaning_report() -> None:
    """Filas eliminadas por regla, split y misión (sin tocar nada en disco)."""
    for split in SPLITS:
        df = load_split(split, columns=["mission"] + CLEAN_COLS + ["flux_global"])
        print(f"\n{split}: {len(df)} filas, {len(df) - len(clean(df))} eliminadas")
        for rule, m in anomaly_masks(df).items():
            print(f"  {rule:28} {m.sum():4}  {df.mission[m].value_counts().to_dict()}")


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
    cleaning_report()


if __name__ == "__main__":
    main()
