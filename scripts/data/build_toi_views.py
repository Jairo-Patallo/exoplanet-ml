"""Vistas de los TOIs abiertos (PC) de ExoFOP para las predicciones congeladas (fase 7, #85; 2026-11-06).

Misma cadena que el entrenamiento (curves.run_star): 1 sector de 2 min de SPOC, PDCSAP, aplanado con el tránsito
enmascarado y fold_views con la efeméride de ExoFOP (época BJD − 2457000 = BTJD, el tiempo de la misión que usa el dataset;
comprobado en 399 TOIs).
Sector (decisión de Jairo 2026-11-06, opción 1): el error de la efeméride crece con las órbitas desde la época,
σ_t = sqrt(σ_t0² + (n·σ_P)²); con la regla del entrenamiento (el sector con más tránsitos), el 8 % de los TOIs tenía
σ_t > 1 duración y el 54 % de ellos salía NO_SIGNAL (results/toi/history/). Ahora: entre los sectores con ≥ 2 tránsitos
esperados, el de menor deriva σ_t / duración; si varios tienen deriva ≤ 0,1 (despreciable), el de más tránsitos, como
en el entrenamiento. Se guarda la deriva y se marca efemeride_incierta si supera 0,25. Los TOIs de una misma estrella se procesan juntos en el mismo
proceso, para que dos procesos no descarguen a la vez el mismo FITS (se truncan).
Reanudable: salta los TOIs que ya están en el qc. Por tandas (--batch i de --n-batches) para informar cada ~30 min.

Uso:  PYTHONPATH=src python scripts/data/build_toi_views.py --batch 0 --n-batches 4 [--workers 4] [--limit N]
Salida: data/toi/views/<TOI>.npz (ignorado por git) y results/toi/qc_toi.csv.
"""
import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet.curves import SEARCH, expected_transits, product_spans, run_star, target_name
from exoplanet.data import DATA

EXOFOP = "data/labels/exofop_toi.csv"
VIEWS = DATA / "toi" / "views"
QC = Path("results/toi/qc_toi.csv")
BTJD = 2457000.0
MAX_GB = 98.0                # freno: el presupuesto acumulado de descargas es de 100 GB (CONTEXTO.md)
DRIFT_FLOOR, DRIFT_FLAG = 0.1, 0.25


def open_tois() -> pd.DataFrame:
    e = pd.read_csv(EXOFOP, comment="#", low_memory=False)
    pc = e[e["TFOPWG Disposition"] == "PC"]
    out = pd.DataFrame({"toi": pc["TOI"].map(lambda x: f"{x:.2f}"), "star_id": "TIC_" + pc["TIC ID"].astype(str),
                        "mission": "tess", "period_days": pc["Period (days)"], "epoch_bjd": pc["Epoch (BJD)"] - BTJD,
                        "duration_hrs": pc["Duration (hours)"], "tmag": pc["TESS Mag"], "depth_ppm": pc["Depth (ppm)"],
                        "source": pc["Source"], "sigma_P": pc["Period (days) err"].fillna(0),
                        "sigma_t0": pc["Epoch (BJD) err"].fillna(0)})
    ok = (out.period_days > 0) & out.epoch_bjd.notna() & (out.duration_hrs > 0) & (out.duration_hrs / 24 < out.period_days)
    print(f"{len(out)} TOIs PC; {(~ok).sum()} sin efeméride completa (p. ej. un solo tránsito, periodo 0): se omiten")
    return out[ok].sort_values(["star_id", "toi"]).reset_index(drop=True)


def choose_sector(r: dict) -> tuple[int | None, float]:
    """(sector, deriva σ_t / duración) según la regla de la cabecera; (None, nan) si MAST no tiene curvas de 2 min."""
    res = lk.search_lightcurve(target_name(r["star_id"]), **SEARCH["tess"])
    if len(res) == 0:
        return None, np.nan
    best = None
    for (a, b), seq in zip(product_spans(res.table, "tess"), np.asarray(res.table["sequence_number"])):
        n_tr = expected_transits(a, b, r["period_days"], r["epoch_bjd"])
        n = abs(round(((a + b) / 2 - r["epoch_bjd"]) / r["period_days"]))
        drift = 24 * np.hypot(r["sigma_t0"], n * r["sigma_P"]) / r["duration_hrs"]
        key = (n_tr < 2, max(drift, DRIFT_FLOOR), -n_tr)
        if best is None or key < best[0]:
            best = (key, int(seq), float(drift))
    return best[1], best[2]


def run_tic(rows: list[dict]) -> list[dict]:
    """Todos los TOIs de una estrella, uno detrás de otro (comparten FITS en la caché de lightkurve)."""
    out = []
    for r in rows:
        tmp = VIEWS / "tmp" / r["toi"]
        drift = np.nan
        try:
            seg, drift = choose_sector(r)
            qc = run_star(SimpleNamespace(**r, segment=seg), tmp) if seg is not None else {"error": "sin curvas en MAST"}
            if (tmp / f"{r['star_id']}.npz").exists():
                (tmp / f"{r['star_id']}.npz").replace(VIEWS / f"{r['toi']}.npz")
        except Exception as e:
            qc = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        shutil.rmtree(tmp, ignore_errors=True)
        out.append({"toi": r["toi"], "drift": drift, "efemeride_incierta": bool(drift > DRIFT_FLAG), **qc})
    return out


def save(tois: pd.DataFrame, done: pd.DataFrame, new: list[dict]) -> pd.DataFrame:
    res = pd.concat([done, tois.merge(pd.DataFrame(new), on="toi")], ignore_index=True)
    QC.parent.mkdir(parents=True, exist_ok=True)
    tmp = QC.with_suffix(".tmp")
    res.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(QC)                                     # escritura atómica
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    gb = sum(p.stat().st_size for p in (DATA / "raw_lc").rglob("*.fits")) / 1e9
    print(f"data/raw_lc: {gb:.1f} GB", flush=True)
    if gb > MAX_GB:
        raise SystemExit(f"Se superan {MAX_GB} GB del presupuesto de descargas (50 GB): para y pregunta a Jairo")
    tois = open_tois()
    tics = np.array_split(tois.star_id.unique(), args.n_batches)[args.batch]
    todo = tois[tois.star_id.isin(tics)].head(args.limit)
    done = pd.read_csv(QC, dtype={"toi": str}) if QC.exists() else pd.DataFrame(columns=["toi"])
    todo = todo[~todo.toi.isin(done.toi)]
    print(f"tanda {args.batch + 1}/{args.n_batches}: {len(todo)} TOIs por hacer en {todo.star_id.nunique()} estrellas",
          flush=True)
    (VIEWS / "tmp").mkdir(parents=True, exist_ok=True)
    t0, new = time.perf_counter(), []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run_tic, g.to_dict("records")) for _, g in todo.groupby("star_id")]
        for i, f in enumerate(as_completed(futs), 1):
            new += f.result()
            if i % 100 == 0 or i == len(futs):
                res = save(tois, done, new)
                print(f"{i}/{len(futs)} estrellas, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    res = save(tois, done, new)
    err = res["error"].notna() if "error" in res else pd.Series(False, index=res.index)
    print(f"total en qc: {len(res)}; válidas: {res['valid'].eq(True).sum()}; sin curvas de 2 min o error: {err.sum()}; "
          f"MB descargados en esta tanda: {pd.DataFrame(new).get('mb_downloaded', pd.Series(dtype=float)).sum():.0f}")


if __name__ == "__main__":
    main()
