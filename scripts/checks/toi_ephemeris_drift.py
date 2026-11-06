"""¿Cae el tránsito donde dice la efeméride en el sector usado? (fase 7, antes de congelar #87; 2026-11-06)

El error de una efeméride crece con el número de órbitas desde la época: σ_t = sqrt(σ_t0² + (n·σ_P)²). Si en el sector
plegado σ_t es comparable a la duración, el plegado no encuentra el tránsito y el modelo puede decir NO_SIGNAL aunque
la señal sea real. Para cada TOI válido se recupera el sector usado (la misma regla curves.select_segments sobre los FITS
en caché), se calcula σ_t / duración y, si ya existen predicciones, la fracción de NO_SIGNAL por tramos.

Uso:  PYTHONPATH=src python scripts/checks/toi_ephemeris_drift.py [--pred results/toi/predictions_v1.csv]
Salida: results/toi/ephemeris_drift.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from exoplanet.curves import MAX_DAYS, RAW, select_segments

QC, EXOFOP, OUT = Path("results/toi/qc_toi.csv"), "data/labels/exofop_toi.csv", Path("results/toi/ephemeris_drift.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    args = ap.parse_args()
    qc = pd.read_csv(QC, dtype={"toi": str})
    qc = qc[qc["valid"].eq(True)]
    e = pd.read_csv(EXOFOP, comment="#", low_memory=False)
    e["toi"] = e["TOI"].map(lambda x: f"{x:.2f}")
    qc = qc.merge(e[["toi", "Period (days) err", "Epoch (BJD) err"]].rename(
        columns={"Period (days) err": "sigma_P", "Epoch (BJD) err": "sigma_t0"}), on="toi", how="left")
    files: dict[str, list[Path]] = {}
    for p in (RAW / "mastDownload" / "TESS").glob("tess*/*_lc.fits"):
        files.setdefault(f"TIC_{int(p.name.split('-')[2])}", []).append(p)
    rows = []
    for r in qc.itertuples():
        fs = sorted(files.get(r.star_id, []))
        spans = [(h["TSTART"], h["TSTOP"]) for h in (fits.getheader(f, 1) for f in fs)]
        if not spans:
            continue
        a, b = spans[select_segments(spans, r.period_days, r.epoch_bjd, MAX_DAYS, 1)[0]]
        n = abs(round(((a + b) / 2 - r.epoch_bjd) / r.period_days))
        rows.append({"toi": r.toi, "n_orbits": n, "sigma_P": r.sigma_P, "sigma_t0": r.sigma_t0,
                     "duration_hrs": r.duration_hrs})
    d = pd.DataFrame(rows)
    d["sigma_t_hrs"] = 24 * np.hypot(d.sigma_t0.fillna(0), d.n_orbits * d.sigma_P.fillna(0))
    d["sigma_over_dur"] = d.sigma_t_hrs / d.duration_hrs
    d["sin_error_publicado"] = d.sigma_P.isna()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.round(5).to_csv(OUT, index=False, encoding="utf-8")
    bins = pd.cut(d.sigma_over_dur, [-0.01, 0.1, 0.25, 0.5, 1, np.inf], labels=["≤ 0,1", "0,1–0,25", "0,25–0,5", "0,5–1", "> 1"])
    print(f"{len(d)} TOIs válidos; σ_t / duración: mediana {d.sigma_over_dur.median():.3f}; "
          f"sin error de periodo publicado: {d.sin_error_publicado.sum()}")
    if args.pred and Path(args.pred).exists():
        p = pd.read_csv(args.pred, dtype={"toi": str})[["toi", "pred", "exominerpp_score"]]
        d = d.assign(bin=bins).merge(p, on="toi")
        print(d.groupby("bin", observed=True).agg(n=("toi", "size"), frac_no_signal=("pred", lambda s: (s == "NO_SIGNAL").mean()),
                                                  frac_fp=("pred", lambda s: (s == "FALSE_POSITIVE").mean())).round(3).to_string())
    else:
        print(bins.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
