"""Curvas sin plegar para el detector sin BLS (fase 4, research/12, 2026-11-06).

Para cada estrella válida de los splits pedidos: los mismos archivos que su vista de catálogo y que BLS
(build_bls_views.star_files) -> aplanado a ciegas, sin efeméride (bls.flatten_blind) -> promedio en bloques de 30 min
sobre una rejilla regular desde el primer dato de cada archivo (NaN donde no hay datos) -> los archivos se encadenan
con un hueco fijo de GAP bloques (los trimestres de Kepler elegidos pueden estar separados cientos de días: el hueco
real se acorta; la CNN mira patrones locales y no la fase exacta entre archivos) -> como mucho SERIES_LEN bloques.
No se usa la efeméride del catálogo en ningún paso (salvo la elección de trimestres de Kepler, la misma que en las
vistas, cuando hay más de 2 archivos).

Uso:  PYTHONPATH="src;scripts/data" python scripts/data/build_series.py [--splits train val] [--workers 12]   (reanudable)
Salida: data/sample/series/<star_id>.npy (float32) y results/sample/qc_series.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from build_bls_views import star_files, stars
from exoplanet.bls import flatten_blind
from exoplanet.data import DATA

SERIES = DATA / "sample" / "series"
QC = Path("results/sample/qc_series.csv")
BIN_DAYS = 1 / 48                  # 30 min, la cadencia larga de Kepler y K2
SERIES_LEN = 9216                  # 192 d: dos trimestres de Kepler
GAP = 96                           # 2 d de NaN entre archivos


def binned(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    idx = ((t - t.min()) / BIN_DAYS).astype(int)
    keep = idx < SERIES_LEN
    idx, y = idx[keep], y[keep]
    n = idx.max() + 1
    s, c = np.bincount(idx, y, minlength=n), np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore"):
        return (s / c).astype(np.float32)          # NaN en los bloques vacíos


def job(row) -> dict:
    base = {"star_id": row.star_id, "mission": row.mission, "split": row.split, "disposition": row.disposition}
    try:
        files = star_files(row)
        if not files:
            return {**base, "error": "sin archivos locales"}
        parts = [flatten_blind(lk.read(f, quality_bitmask="default")) for f in files]
        plain = lambda a: np.asarray(a.filled(np.nan) if hasattr(a, "filled") else a, dtype=float)   # astropy Masked
        segs = [binned(plain(t), plain(y)) for t, y in sorted(parts, key=lambda p: float(np.min(plain(p[0]))))]
        gap = np.full(GAP, np.nan, np.float32)
        s = np.concatenate([x for seg in segs for x in (seg, gap)][:-1])[:SERIES_LEN]
        np.save(SERIES / f"{row.star_id}.npy", s)
        return {**base, "n_files": len(segs), "n_bins": len(s), "frac_vacios": float(np.isnan(s).mean())}
    except Exception as ex:
        return {**base, "error": f"{type(ex).__name__}: {str(ex)[:120]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    SERIES.mkdir(parents=True, exist_ok=True)
    rows = stars(args.splits)
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    done = done[done["error"].isna()] if len(done) else done
    todo = [SimpleNamespace(**r) for r in rows[~rows.star_id.isin(set(done.star_id))].to_dict("records")][:args.limit]
    print(f"{len(rows)} estrellas; faltan {len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            done = pd.concat([done, pd.DataFrame([fut.result()])], ignore_index=True)
            if i % 100 == 0 or i == len(todo):
                done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
                QC.with_suffix(".tmp").replace(QC)
                print(f"[{i}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
