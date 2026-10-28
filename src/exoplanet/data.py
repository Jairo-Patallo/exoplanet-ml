"""Carga de los splits originales (train/val/test) filtrados por misión.

Los splits de HuggingFace ya están estratificados por misión y clase y no comparten estrellas,
así que no se vuelven a dividir: solo se filtra cada uno por misión al leerlo (sin copias en disco).

Uso:  python -m exoplanet.data   -> imprime recuentos por split, misión y clase y comprueba los splits
"""
from pathlib import Path

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
