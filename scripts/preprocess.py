"""Preprocesado de una muestra de TRAIN para la app: decodifica las curvas y guarda arrays listos para usar.

Se usa una muestra estratificada por misión y clase (no test: test no se mira hasta la evaluación final).
Uso:  python scripts/preprocess.py   (desde la raíz del repo)
Salida (data/processed/):
  meta.parquet      -> metadatos sin las columnas binarias + variables derivadas
  curves_<vista>.npy -> matrices float32 (n_estrellas, longitud) por vista
"""
import numpy as np
import pandas as pd

from exoplanet.data import DATA, TARGET, load_split

OUT = DATA / "processed"
N_SAMPLE = 3000
SEED = 42
VIEWS = {"flux_raw": 20000, "flux_global": 201, "flux_local": 81, "flux_odd": 201, "flux_even": 201}
# meta conserva las columnas de exoplanet.data.LEAKAGE para mostrarlas en la app,
# pero NO deben usarse como entradas de un modelo (fuga de información).


def decode(col: pd.Series, length: int) -> np.ndarray:
    """bytes -> matriz float32 (n, length); comprueba que cada fila tiene la longitud esperada."""
    arr = np.stack([np.frombuffer(b, dtype=np.float32) for b in col])
    assert arr.shape[1] == length, f"{col.name}: longitud {arr.shape[1]} != {length}"
    return arr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = load_split("train", columns=["star_id", "mission", TARGET])
    frac = N_SAMPLE / len(ids)
    sample = ids.groupby(["mission", TARGET]).sample(frac=frac, random_state=SEED)["star_id"]
    # El filtro se aplica al leer: no se carga train entero (~1.5 GB) en memoria
    df = pd.read_parquet(DATA / "train.parquet", filters=[("star_id", "in", sample.tolist())])
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
