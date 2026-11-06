"""Vistas plegadas con la efeméride de BLS en lugar de la del catálogo (fase 4, #53, 2026-11-05).

Para cada estrella con vista válida de los splits pedidos (por defecto val): los mismos archivos que su vista de
catálogo -> aplanado a ciegas -> BLS (exoplanet.bls) -> curves.prepare con la efeméride de BLS (enmascara el tránsito
encontrado) -> fold_views. Los NO_SIGNAL se pliegan con su pico de BLS (decisión de Jairo 2026-11-05).
Archivos por estrella: TESS y K2, el sector/campaña de la pareja (#32; data/raw_lc tiene además el 2.º sector de
build_tess_2sectors.py, que aquí no se usa); Kepler, sus trimestres (si hay más de 2 archivos, los que habría elegido
curves.select_segments, que usa la efeméride del catálogo: es la misma selección de datos que la vista de catálogo).

Uso:  PYTHONPATH=src python scripts/data/build_bls_views.py [--splits val] [--workers 4] [--limit N]   (reanudable)
      negativos difíciles: --splits train --nosignal --kepler-sample 1500
Salida: data/sample/views_bls/<star_id>.npz y results/sample/qc_bls.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet.bls import find_ephemeris, flatten_blind, match
from exoplanet.curves import RAW, prepare, select_segments
from exoplanet.data import DATA
from exoplanet.folding import fold_views
from exoplanet.labels import strong_stars

VIEWS = DATA / "sample" / "views_bls"
QC = Path("results/sample/qc_bls.csv")


def star_files(row) -> list[Path]:
    num = int(row.star_id.split("_")[1])
    if row.mission == "tess":
        return sorted((RAW / "mastDownload" / "TESS").glob(f"tess*-s{int(row.segment):04d}-{num:016d}-*/*_lc.fits"))
    if row.mission == "k2":     # C10 se publica en mitades: carpeta ktwo...-c10_lc, archivo ...-c102_llc.fits (segmento 102)
        seg = int(row.segment)
        camp = seg // 10 if seg >= 100 else seg
        return sorted((RAW / "mastDownload" / "K2").glob(f"ktwo{num}-c{camp:02d}*_lc/*-c{seg:02d}*_llc.fits"))
    files = sorted((RAW / "mastDownload" / "Kepler").glob(f"kplr{num:09d}_lc_*/*_llc.fits"))
    if len(files) > 2:
        spans = [(float(np.nanmin(t)), float(np.nanmax(t))) for t in (lk.read(f).time.value for f in files)]
        files = [files[i] for i in select_segments(spans, row.period_days, row.epoch_bjd)]
    return files


def run(row) -> dict:
    files = star_files(row)
    if not files:
        return {"error": "sin archivos locales"}
    lcs = [lk.read(f, quality_bitmask="default") for f in files]
    parts = [flatten_blind(lc) for lc in lcs]
    e = find_ephemeris(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    T = e.duration_hrs / 24
    prep = [prepare(lc, e.period, e.t0, T) for lc in lcs]
    views = fold_views(np.concatenate([p[0] for p in prep]), np.concatenate([p[1] for p in prep]), e.period, e.t0, T)
    VIEWS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VIEWS / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    rel = match(e.period, row.period_days) if row.disposition != "NO_SIGNAL" else "sin señal"
    return {"n_files": len(files), "bls_period": e.period, "bls_t0": e.t0, "bls_duration_hrs": e.duration_hrs,
            "bls_depth": e.depth, "bls_sde": e.sde, "match": rel, **views.qc}


def job(row) -> dict:
    try:
        res = run(row)
    except Exception as ex:
        res = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    return {"star_id": row.star_id, "mission": row.mission, "split": row.split, "disposition": row.disposition,
            "period_days": row.period_days, **res}


def stars(splits) -> pd.DataFrame:
    """Estrellas válidas con efeméride de catálogo y el sector/campaña de su vista (de la pareja en K2 y TESS)."""
    qc = pd.read_csv("results/sample/qc.csv")
    ok = qc[qc["valid"].eq(True) & qc.split.isin(splits)][["star_id"]]
    cols = ["star_id", "mission", "split", "disposition", "period_days", "epoch_bjd", "duration_hrs"]
    neg = pd.read_csv("results/sample/negatives.csv")
    pos = strong_stars()[cols].merge(neg[["pair_star_id", "segment"]].rename(columns={"pair_star_id": "star_id"}),
                                     on="star_id", how="left")
    rows = pd.concat([pos, neg[cols + ["segment"]]]).drop_duplicates("star_id").merge(ok, on="star_id")
    lost = rows.mission.isin(["k2", "tess"]) & rows.segment.isna()
    print(f"{lost.sum()} estrellas de K2/TESS sin sector conocido (sin pareja): se omiten", flush=True)
    return rows[~lost]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nosignal", action="store_true", help="solo NO_SIGNAL (negativos difíciles para entrenar)")
    ap.add_argument("--kepler-sample", type=int, default=None, help="como mucho N estrellas de Kepler (al azar, semilla 0)")
    args = ap.parse_args()
    rows = stars(args.splits)
    if args.nosignal:
        rows = rows[rows.disposition == "NO_SIGNAL"]
    if args.kepler_sample:
        kep = rows[rows.mission == "kepler"]
        rows = pd.concat([rows[rows.mission != "kepler"], kep.sample(min(len(kep), args.kepler_sample), random_state=0)])
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    done = done[done["error"].isna()] if len(done) else done
    todo = [SimpleNamespace(**r) for r in rows[~rows.star_id.isin(set(done.star_id))].to_dict("records")][:args.limit]
    print(f"{len(rows)} estrellas ({rows.mission.value_counts().to_dict()}); faltan {len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            done = pd.concat([done, pd.DataFrame([fut.result()])], ignore_index=True)
            if i % 25 == 0 or i == len(todo):
                done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
                QC.with_suffix(".tmp").replace(QC)
                print(f"[{i}/{len(todo)}]", flush=True)
    print(pd.crosstab([done.mission, done.disposition], done["match"]).to_string())


if __name__ == "__main__":
    main()
