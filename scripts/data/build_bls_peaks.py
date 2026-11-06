"""Vistas plegadas con los K mejores picos de BLS (fase 4, triaje en modo búsqueda, research/11 §5, 2026-11-06).

Si BLS da un periodo erróneo, la señal se emborrona al plegar y parece ruido: un tercio de los candidatos de val se
pierden así. Aquí cada estrella se pliega con sus K picos distintos de mayor SDE (exoplanet.bls.find_ephemerides);
el triaje puntúa cada vista y se queda con la mejor. Mismos archivos y preparación que build_bls_views.py.

Uso:  PYTHONPATH=src python scripts/data/build_bls_peaks.py [--splits val] [--k 3] [--workers 12]   (reanudable)
Salida: data/sample/views_bls_peaks/<star_id>_p<i>.npz y results/sample/qc_bls_peaks.csv (una fila por estrella y pico)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from build_bls_views import star_files, stars
from exoplanet.bls import find_ephemerides, flatten_blind, match
from exoplanet.curves import prepare
from exoplanet.data import DATA
from exoplanet.folding import fold_views

VIEWS = DATA / "sample" / "views_bls_peaks"
QC = Path("results/sample/qc_bls_peaks.csv")


def run(row, k: int) -> list[dict]:
    files = star_files(row)
    if not files:
        return [{"error": "sin archivos locales"}]
    lcs = [lk.read(f, quality_bitmask="default") for f in files]
    parts = [flatten_blind(lc) for lc in lcs]
    out = []
    for i, e in enumerate(find_ephemerides(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), k)):
        T = e.duration_hrs / 24
        prep = [prepare(lc, e.period, e.t0, T) for lc in lcs]
        views = fold_views(np.concatenate([p[0] for p in prep]), np.concatenate([p[1] for p in prep]), e.period, e.t0, T)
        np.savez_compressed(VIEWS / f"{row.star_id}_p{i}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
        rel = match(e.period, row.period_days) if row.disposition != "NO_SIGNAL" else "sin señal"
        out.append({"pico": i, "bls_period": e.period, "bls_t0": e.t0, "bls_duration_hrs": e.duration_hrs,
                    "bls_sde": e.sde, "match": rel, **views.qc})
    return out


def job(row, k: int) -> list[dict]:
    try:
        res = run(row, k)
    except Exception as ex:
        res = [{"error": f"{type(ex).__name__}: {str(ex)[:120]}"}]
    base = {"star_id": row.star_id, "mission": row.mission, "split": row.split, "disposition": row.disposition,
            "period_days": row.period_days}
    return [{**base, **r} for r in res]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    VIEWS.mkdir(parents=True, exist_ok=True)
    rows = stars(args.splits)
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    done = done[done["error"].isna()] if len(done) else done
    todo = [SimpleNamespace(**r) for r in rows[~rows.star_id.isin(set(done.star_id))].to_dict("records")][:args.limit]
    print(f"{len(rows)} estrellas; faltan {len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, r, args.k) for r in todo]), 1):
            done = pd.concat([done, pd.DataFrame(fut.result())], ignore_index=True)
            if i % 25 == 0 or i == len(todo):
                done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
                QC.with_suffix(".tmp").replace(QC)
                print(f"[{i}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
