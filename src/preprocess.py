"""Preprocesado del split test: decodifica las curvas de luz y guarda arrays listos para usar.

Uso:  python src/preprocess.py
Salida (data/processed/):
  meta.parquet      -> metadatos sin las columnas binarias + variables derivadas
  curves_<vista>.npy -> matrices float32 (n_estrellas, longitud) por vista
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/test.parquet")
OUT = Path("data/processed")
VIEWS = {"flux_raw": 20000, "flux_global": 201, "flux_local": 81, "flux_odd": 201, "flux_even": 201}
# Columnas derivadas de la etiqueta: se conservan en meta para mostrarlas,
# pero NO deben usarse como entradas de un modelo (fuga de información).
LEAKAGE = ["confidence_score", "label_source", "kepoi_name",
           "fp_flag_not_transit", "fp_flag_stellar", "fp_flag_centroid",
           "fp_flag_ephemeris", "disposition_disputed"]


def decode(col: pd.Series, length: int) -> np.ndarray:
    """bytes -> matriz float32 (n, length); comprueba que cada fila tiene la longitud esperada."""
    arr = np.stack([np.frombuffer(b, dtype=np.float32) for b in col])
    assert arr.shape[1] == length, f"{col.name}: longitud {arr.shape[1]} != {length}"
    return arr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW)
    print(f"Filas: {len(df)} | clases:\n{df['disposition'].value_counts().to_string()}")

    curves = {name: decode(df[name], n) for name, n in VIEWS.items()}

    # Limpieza: NaN/inf -> 1.0 (nivel normal de flujo tras dividir por la mediana)
    for name, a in curves.items():
        bad = ~np.isfinite(a)
        if bad.any():
            print(f"{name}: {bad.sum()} valores no finitos reemplazados por 1.0")
            a[bad] = 1.0

    # NO_SIGNAL tiene vistas plegadas a cero; se marcan en lugar de borrarlas
    folded_empty = np.all(curves["flux_global"] == 0, axis=1)
    print(f"Filas con vistas plegadas vacías: {folded_empty.sum()} "
          f"(NO_SIGNAL: {(df['disposition'] == 'NO_SIGNAL').sum()})")

    meta = df.drop(columns=list(VIEWS) + ["source_urls"]).copy()
    meta["folded_empty"] = folded_empty

    # Variables derivadas simples (pensadas para la app y como baseline)
    meta["raw_std_ppm"] = curves["flux_raw"].std(axis=1) * 1e6          # ruido global
    meta["local_min_depth_ppm"] = (1 - curves["flux_local"].min(axis=1)) * 1e6
    odd_min = curves["flux_odd"].min(axis=1)
    even_min = curves["flux_even"].min(axis=1)
    meta["odd_even_diff_ppm"] = np.abs(odd_min - even_min) * 1e6          # >> 0 sugiere binaria
    meta.loc[folded_empty, ["local_min_depth_ppm", "odd_even_diff_ppm"]] = np.nan

    meta.to_parquet(OUT / "meta.parquet", index=False)
    for name, a in curves.items():
        np.save(OUT / f"curves_{name}.npy", a)
    print(f"Guardado en {OUT}/  | columnas meta: {meta.shape[1]}")


if __name__ == "__main__":
    main()
