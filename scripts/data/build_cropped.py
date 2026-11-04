"""Aumento "tipo TESS": curvas de Kepler y K2 recortadas a un sector de TESS (27,4 d) (fase 3, 2026-11-04).

Para cada estrella de Kepler y K2 de train + val con vista válida: se elige al azar (semilla fija por estrella) uno de sus
archivos ya descargados (un trimestre de Kepler o una campaña de K2) y, dentro, una ventana de 27,4 días con al menos un
tránsito esperado. Se prepara y se pliega con el mismo código que el resto. La etiqueta es la de la estrella.
Solo usa especificaciones públicas de TESS (duración del sector); nada de las curvas ni etiquetas de TESS.
Las vistas originales (data/sample/views) no se tocan: estas son copias extra para entrenar.

Uso:  PYTHONPATH=src python scripts/data/build_cropped.py [--workers 4] [--limit N]   (se puede relanzar)
Salida: data/sample/views_crop/<star_id>.npz y results/sample/qc_crop.csv
"""
import argparse
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet.curves import RAW, expected_transits, prepare
from exoplanet.data import DATA
from exoplanet.folding import fold_views
from exoplanet.labels import strong_stars

VIEWS = DATA / "sample" / "views_crop"
QC = Path("results/sample/qc_crop.csv")
SECTOR_DAYS = 27.4


def raw_files(star_id: str, mission: str) -> list[Path]:
    num = int(star_id.split("_")[1])
    if mission == "kepler":
        return sorted((RAW / "mastDownload" / "Kepler").glob(f"kplr{num:09d}_lc_*/*_llc.fits"))
    return sorted((RAW / "mastDownload" / "K2").glob(f"ktwo{num}-c*_lc/*_llc.fits"))


def crop(row: SimpleNamespace) -> dict:
    rng = np.random.default_rng(zlib.crc32(row.star_id.encode()))
    files = raw_files(row.star_id, row.mission)
    if not files:
        return {"error": "sin archivos locales"}
    path = files[rng.integers(len(files))]
    lc = lk.read(path, quality_bitmask="default").remove_nans()
    t = lc.time.value
    starts = np.arange(t.min(), max(t.min(), t.max() - SECTOR_DAYS) + 1e-9, 0.5)
    ok = [s for s in starts if expected_transits(s, s + SECTOR_DAYS, row.period_days, row.epoch_bjd) >= 1]
    if not ok:
        return {"error": "ninguna ventana con tránsito"}
    s = ok[rng.integers(len(ok))]
    T = row.duration_hrs / 24
    views = fold_views(*prepare(lc[(t >= s) & (t < s + SECTOR_DAYS)], row.period_days, row.epoch_bjd, T),
                       row.period_days, row.epoch_bjd, T)
    VIEWS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VIEWS / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    return {"file": path.name, "start": round(float(s), 2), **views.qc}


def job(row: SimpleNamespace) -> dict:
    try:
        res = crop(row)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"star_id": row.star_id, "mission": row.mission, "split": row.split, "disposition": row.disposition, **res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    qc = pd.read_csv("results/sample/qc.csv")
    ok = set(qc.star_id[qc["valid"].eq(True) & qc.split.isin(["train", "val"]) & qc.mission.isin(["kepler", "k2"])])
    cols = ["star_id", "mission", "split", "disposition", "period_days", "epoch_bjd", "duration_hrs"]
    rows = pd.concat([strong_stars()[cols], pd.read_csv("results/sample/negatives.csv")[cols]]).drop_duplicates("star_id")
    rows = rows[rows.star_id.isin(ok)]
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    done = done[done["error"].isna()] if len(done) else done
    todo = [SimpleNamespace(**r) for r in rows[~rows.star_id.isin(set(done.star_id))].to_dict("records")][:args.limit]
    print(f"{len(rows)} estrellas de Kepler y K2; faltan {len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            done = pd.concat([done, pd.DataFrame([fut.result()])], ignore_index=True)
            if i % 200 == 0 or i == len(todo):
                done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
                QC.with_suffix(".tmp").replace(QC)        # atómico
                print(f"[{i}/{len(todo)}]", flush=True)
    print(done.groupby(["mission", "disposition"]).agg(n=("star_id", "size"), errores=("error", "count"),
                                                       validas=("valid", "sum"), transitos_mediana=("n_transits", "median")).to_string())


if __name__ == "__main__":
    main()
