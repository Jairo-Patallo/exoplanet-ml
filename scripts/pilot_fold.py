"""Piloto de descarga y plegado (tarea #47): mide tamaño, tiempo y % de vistas válidas antes de fijar la muestra.

Toma una muestra estratificada por misión y clase de estrellas con etiqueta FUERTE (PLANET / FALSE_POSITIVE), descarga
sus curvas de MAST (exoplanet.curves), las pliega con exoplanet.folding y guarda las vistas y la calidad de cada estrella.
Los NO_SIGNAL no entran: sus negativos se generan aparte (tarea #32). Se puede interrumpir y relanzar: salta las ya hechas.

Uso:  python scripts/pilot_fold.py [--n 100] [--seed 0]     (desde la raíz, con PYTHONPATH=src)
Salida: data/pilot/views/<star_id>.npz (ignorado por git), results/pilot/qc.csv (una fila por estrella) y
        results/pilot/summary.csv (por misión: tamaño, tiempo, % válidas y coste estimado de la muestra completa).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet.curves import download_star, read_prepare
from exoplanet.data import DATA, MISSIONS, SPLITS, TARGET, clean, load_split
from exoplanet.folding import fold_views
from exoplanet.labels import label_strength

VIEWS = DATA / "pilot" / "views"
OUT = Path("results/pilot")


def strong_stars() -> pd.DataFrame:
    """Una fila por estrella con etiqueta fuerte y efeméride completa (todos los splits: el piloto no entrena nada)."""
    cols = ["star_id", "mission", TARGET, "label_source", "period_days", "epoch_bjd", "duration_hrs", "depth_ppm", "planet_radius_earth"]
    df = pd.concat([load_split(s, columns=cols) for s in SPLITS], ignore_index=True)
    df = df[df[TARGET] != "NO_SIGNAL"].dropna(subset=["period_days", "epoch_bjd", "duration_hrs"])
    df = clean(df)
    df = df[label_strength(df) == "fuerte"]
    return df.drop_duplicates("star_id").reset_index(drop=True)


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n estrellas por misión, mitad por clase (todas las que haya si una clase tiene menos)."""
    parts = []
    for mission in MISSIONS:
        for cls, g in df[df.mission == mission].groupby(TARGET):
            parts.append(g.sample(min(len(g), n // 2), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_star(row) -> dict:
    t = time.perf_counter()
    paths = download_star(row.star_id, row.mission, row.period_days, row.epoch_bjd)
    t_down = time.perf_counter() - t
    if not paths:
        return {"error": "sin curvas en MAST", "t_download_s": t_down}
    t = time.perf_counter()
    T = row.duration_hrs / 24
    views = fold_views(*read_prepare(paths, row.period_days, row.epoch_bjd, T), row.period_days, row.epoch_bjd, T)
    VIEWS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VIEWS / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    return {"n_files": len(paths), "mb_downloaded": sum(p.stat().st_size for p in paths) / 1e6,
            "t_download_s": t_down, "t_fold_s": time.perf_counter() - t, **views.qc}


def summarize(qc: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    ok = qc[qc["error"].isna()]
    rows = []
    for m in MISSIONS:
        a, o = qc[qc.mission == m], ok[ok.mission == m]
        if a.empty:
            continue
        n_all = int((universe.mission == m).sum())
        rows.append({"mission": m, "n_tried": len(a), "n_downloaded": len(o), "pct_no_curves_or_error": round(100 * (1 - len(o) / len(a)), 1),
                     "pct_valid_of_downloaded": round(100 * o["valid"].mean(), 1) if len(o) else np.nan,
                     "pct_valid_of_tried": round(100 * o["valid"].sum() / len(a), 1),
                     "pct_odd_even_valid": round(100 * o["odd_even_valid"].mean(), 1) if len(o) else np.nan,
                     "mb_per_star_mean": round(o["mb_downloaded"].mean(), 2), "mb_per_star_median": round(o["mb_downloaded"].median(), 2),
                     "s_per_star_median": round((o["t_download_s"] + o["t_fold_s"]).median(), 1),
                     "n_strong_stars_total": n_all,
                     "est_gb_full": round(n_all * o["mb_downloaded"].mean() / 1000, 1),
                     "est_hours_full": round(n_all * (o["t_download_s"] + o["t_fold_s"]).mean() / 3600, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="estrellas por misión (mitad por clase)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    universe = strong_stars()
    chosen = sample(universe, args.n, args.seed)
    print(chosen.groupby(["mission", TARGET]).size().unstack().to_string(), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    qc_path = OUT / "qc.csv"
    done = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()
    seen = set(done.star_id) if len(done) else set()
    for i, row in enumerate(chosen.itertuples(), 1):
        if row.star_id in seen:
            continue
        try:
            res = run_star(row)
        except Exception as e:      # una estrella rota no debe parar el piloto; se registra y se sigue
            res = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        rec = {"star_id": row.star_id, "mission": row.mission, "disposition": getattr(row, TARGET),
               "period_days": row.period_days, "duration_hrs": row.duration_hrs, **res}
        done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
        done.to_csv(qc_path, index=False, encoding="utf-8")
        print(f"[{i}/{len(chosen)}] {row.star_id} {row.mission}: {res.get('error') or ('valid=%s %.1f MB' % (res['valid'], res['mb_downloaded']))}", flush=True)

    if "error" not in done:
        done["error"] = np.nan
    summary = summarize(done, universe)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
