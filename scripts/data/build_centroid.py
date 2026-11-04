"""Vistas de centroide de la muestra (variante "curva + centroide", 2026-11-03).

Para cada estrella con vista válida en qc.csv se eligen, entre los FITS ya descargados, los mismos segmentos que usó el
plegado del flujo (misma regla que curves.download_star: select_segments con los intervalos de las cabeceras, o el
segmento fijado de los negativos emparejados), se quita la deriva del centroide (curves.read_centroid) y se pliega
(folding.fold_centroid): desplazamiento en unidades de su ruido, local (81) y global (201).
Comprobación: en ~2 % de las estrellas se vuelve a plegar el flujo con esos mismos archivos y se compara con la vista
guardada; si coincide, los segmentos son los mismos. No descarga nada.

Uso:  python scripts/data/build_centroid.py [--workers 8] [--limit N] [--check-frac 0.02] [--out ruta]   (PYTHONPATH=src)
Salida: data/sample/centroid/<star_id>.npz (ignorado por git) y results/sample/centroid_qc.csv.
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.io import fits

from exoplanet.curves import MAX_DAYS, MAX_SEGMENTS, RAW, read_centroid, read_prepare, select_segments
from exoplanet.data import DATA
from exoplanet.folding import fold_centroid, fold_views
from exoplanet.labels import strong_stars
from exoplanet.negatives import segment_of

OUT_DIR = DATA / "sample" / "centroid"
VIEWS = DATA / "sample" / "views"
QC_OUT = Path("results/sample/centroid_qc.csv")
CHECK_FRAC, SEED = 0.02, 0
KEPLER_FILE = re.compile(r"kplr(\d+)-\d+_llc\.fits$")


def file_index() -> dict[str, list[tuple[int | None, Path]]]:
    """star_id -> [(campaña/sector o None en Kepler, ruta)] de los FITS en caché."""
    idx: dict[str, list] = {}
    for mission, folder in (("kepler", "Kepler"), ("k2", "K2"), ("tess", "TESS")):
        for p in (RAW / "mastDownload" / folder).rglob("*.fits"):
            if mission == "kepler":
                m = KEPLER_FILE.search(p.name)
                key = (f"KIC_{int(m.group(1))}", None) if m else None
            else:
                key = segment_of(p, mission)
            if key:
                idx.setdefault(key[0], []).append((key[1], p))
    return idx


def choose(row, files: list[tuple[int | None, Path]]) -> list[Path]:
    if not pd.isna(row.segment):                       # negativo emparejado de K2/TESS: su segmento fijado
        return sorted(p for s, p in files if s == int(row.segment))
    files = sorted(set(files))
    spans = [(h["TSTART"], h["TSTOP"]) for h in (fits.getheader(p, 1) for _, p in files)]
    chosen = select_segments(spans, row.period_days, row.epoch_bjd, MAX_DAYS, MAX_SEGMENTS.get(row.mission))
    return [files[i][1] for i in chosen]


def job(row, paths: list[Path], check: bool) -> dict:
    rec = {"star_id": row.star_id, "n_files_used": len(paths), "n_files_match": len(paths) == row.n_files}
    try:
        T = row.duration_hrs / 24
        local, glob, valid = fold_centroid(*read_centroid(paths, row.period_days, row.epoch_bjd, T),
                                           row.period_days, row.epoch_bjd, T)
        np.savez_compressed(OUT_DIR / f"{row.star_id}.npz", local=local, glob=glob)
        rec["centroid_valid"] = valid
        if check:
            v = fold_views(*read_prepare(paths, row.period_days, row.epoch_bjd, T), row.period_days, row.epoch_bjd, T)
            with np.load(VIEWS / f"{row.star_id}.npz") as saved:
                rec["flux_check"] = bool(np.allclose(v.local, saved["local"], equal_nan=True))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-frac", type=float, default=CHECK_FRAC, help="fracción de estrellas con comprobación del flujo")
    ap.add_argument("--out", default=str(QC_OUT))
    args = ap.parse_args()
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].eq(True)]
    eph = pd.concat([strong_stars()[["star_id", "epoch_bjd"]].assign(segment=np.nan),
                     pd.read_csv("results/sample/negatives.csv")[["star_id", "epoch_bjd", "segment"]]])
    stars = qc.merge(eph.drop_duplicates("star_id"), on="star_id", how="left").head(args.limit)
    rng = np.random.default_rng(SEED)
    check = rng.random(len(stars)) < args.check_frac
    idx = file_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        recs = [SimpleNamespace(**r) for r in stars.to_dict("records")]
        futs = [pool.submit(job, r, choose(r, idx.get(r.star_id, [])), c) for r, c in zip(recs, check)]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 500 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} estrellas", flush=True)
    out = stars[["star_id", "mission", "split", "disposition"]].merge(pd.DataFrame(rows), on="star_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nsegmentos iguales que el flujo: {out.n_files_match.mean():.1%}; comprobación del flujo: "
          f"{out.flux_check.dropna().mean():.1%} de {out.flux_check.notna().sum()}; errores: {out.get('error', pd.Series()).notna().sum()}")
    print(out.groupby(["mission", "disposition"]).centroid_valid.mean().unstack().round(3))


if __name__ == "__main__":
    main()
