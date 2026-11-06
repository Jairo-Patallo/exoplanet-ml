"""Rasgos de los tránsitos individuales (propuesta P1, paso 1; 2026-11-06): ¿hay información por tránsito que la mediana
plegada borra? Para cada PLANET/FP con vista válida: los mismos archivos y el mismo aplanado que su vista de catálogo
(build_bls_views.star_files + curves.prepare) -> exoplanet.features.transit_features. No descarga nada.

Uso:  PYTHONPATH=src python scripts/data/build_transit_feats.py [--splits train val] [--missions tess k2 kepler]
      [--workers 6] [--limit N]
Salida: results/sample/transit_feats.csv (una fila por estrella). El análisis está en scripts/checks/transit_feats_check.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd

from build_bls_views import star_files, stars
from exoplanet.curves import prepare
from exoplanet.features import transit_features

OUT = Path("results/sample/transit_feats.csv")


def run(row) -> dict:
    rec = {"star_id": row.star_id}
    try:
        files = star_files(row)
        if not files:
            return {**rec, "error": "sin archivos locales"}
        T = row.duration_hrs / 24
        parts = [prepare(lk.read(f, quality_bitmask="default"), row.period_days, row.epoch_bjd, T) for f in files]
        t, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        return {**rec, "n_files": len(files), **transit_features(t, y, row.period_days, row.epoch_bjd, T)}
    except Exception as e:
        return {**rec, "error": f"{type(e).__name__}: {str(e)[:100]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--missions", nargs="+", default=["tess", "k2", "kepler"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = stars(args.splits)
    rows = rows[(rows.disposition != "NO_SIGNAL") & rows.mission.isin(args.missions)].head(args.limit)
    print(f"{len(rows)} estrellas: {rows.groupby(['mission', 'disposition']).size().to_dict()}", flush=True)
    out = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(run, SimpleNamespace(**r)) for r in rows.to_dict("records")]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 500 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}", flush=True)
    res = rows[["star_id", "mission", "split", "disposition"]].merge(pd.DataFrame(out), on="star_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    res.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(OUT)                                    # escritura atómica (CONTEXTO: un reinicio corrompió qc.csv)
    err = res["error"].notna() if "error" in res else pd.Series(False, index=res.index)
    print(f"errores: {err.sum()}; con beta: {res.beta.notna().mean():.1%}")
    print(res[~err].groupby(["mission", "disposition"])[["n_transits", "beta", "z_odd_even", "z_scatter"]]
          .median().round(2))


if __name__ == "__main__":
    main()
