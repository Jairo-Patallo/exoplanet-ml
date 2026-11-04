"""TESS con 2 sectores para medir si más tránsitos ayudan (fase 3, decisión de Jairo 2026-11-04: "mismas estrellas, 1 vs 2").

Solo las parejas (positivo de TESS, su negativo NO_SIGNAL) de train + val en las que las DOS estrellas tienen >= 2 sectores
de 2 min (data/sample/tess_sector_counts.csv) y vista válida con 1 sector. A cada estrella se le añade un segundo sector
al que ya tenía: el positivo, el de más tránsitos esperados; el negativo, el mismo que su pareja si lo tiene (si no, el de
más tránsitos de su efeméride al azar). Así cada pareja tiene 2 sectores y la comparación 1 vs 2 es sobre las mismas
estrellas. Las vistas de 1 sector (data/sample/views) no se tocan.

Uso:  PYTHONPATH=src python scripts/data/build_tess_2sectors.py [--workers 4] [--limit N]   (se puede relanzar)
Salida: data/sample/views_tess2/<star_id>.npz y results/sample/qc_tess2.csv
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import socket

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.mast import conf

from exoplanet.curves import RAW, SEARCH, expected_transits, product_spans, read_prepare, target_name
from exoplanet.data import DATA
from exoplanet.folding import fold_views
from exoplanet.labels import strong_stars

VIEWS = DATA / "sample" / "views_tess2"
QC = Path("results/sample/qc_tess2.csv")
conf.timeout = 120                  # MAST a veces se cuelga sin límite: mejor un error que se reintenta al relanzar
socket.setdefaulttimeout(120)


def two_sectors(row: SimpleNamespace) -> dict:
    res = lk.search_lightcurve(target_name(row.star_id), **SEARCH["tess"])
    seqs = np.asarray(res.table["sequence_number"]).astype(int)
    if row.sector not in seqs:
        return {"error": f"sector {row.sector} no está en MAST"}
    spans = product_spans(res.table, "tess")
    others = [i for i in range(len(res)) if seqs[i] != row.sector]
    if not others:
        return {"error": "solo 1 sector en MAST"}
    pref = [i for i in others if seqs[i] == row.preferred]
    extra = pref[0] if pref else max(others, key=lambda i: (expected_transits(*spans[i], row.period_days, row.epoch_bjd),
                                                            spans[i][1] - spans[i][0]))
    first = int(np.flatnonzero(seqs == row.sector)[0])
    paths = [Path(res[i].download(download_dir=str(RAW)).filename) for i in (first, extra)]
    T = row.duration_hrs / 24
    views = fold_views(*read_prepare(paths, row.period_days, row.epoch_bjd, T), row.period_days, row.epoch_bjd, T)
    VIEWS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VIEWS / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    return {"sector_1": row.sector, "sector_2": int(seqs[extra]), **views.qc}


def job(row: SimpleNamespace) -> dict:
    try:
        res = two_sectors(row)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"star_id": row.star_id, "pair_id": row.pair_id, "split": row.split, "disposition": row.disposition, **res}


def pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positivos y negativos de las parejas elegibles, con el sector que ya tienen."""
    counts = pd.read_csv("data/sample/tess_sector_counts.csv").set_index("star_id").n_sectors
    qc = pd.read_csv("results/sample/qc.csv")
    ok = set(qc.star_id[qc["valid"].eq(True) & qc.split.isin(["train", "val"]) & (qc.mission == "tess")])
    neg = pd.read_csv("results/sample/negatives.csv").query("mission == 'tess'")
    neg = neg[neg.star_id.isin(ok) & neg.pair_star_id.isin(ok)
              & (neg.star_id.map(counts) >= 2) & (neg.pair_star_id.map(counts) >= 2)]
    pos = strong_stars().set_index("star_id").loc[neg.pair_star_id].reset_index()
    # el sector del negativo es, por construcción (#32), el sector que se usó para su pareja
    pos = pos.assign(pair_id=pos.star_id, sector=neg.segment.astype(int).to_numpy(), preferred=-1)
    neg = neg.assign(pair_id=neg.pair_star_id, sector=neg.segment.astype(int))
    return pos, neg


def run(rows: pd.DataFrame, done: pd.DataFrame, workers: int) -> pd.DataFrame:
    todo = [SimpleNamespace(**r) for r in rows.to_dict("records")]
    with ProcessPoolExecutor(workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, r) for r in todo]), 1):
            rec = fut.result()
            done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
            done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
            QC.with_suffix(".tmp").replace(QC)        # atómico
            err = rec.get("error")
            print(f"[{i}/{len(todo)}] {rec['star_id']} {rec['disposition']}: "
                  f"{err if isinstance(err, str) else 'sectores %s+%s, %s tránsitos, valid=%s' % (rec['sector_1'], rec['sector_2'], rec['n_transits'], rec['valid'])}", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="solo las N primeras parejas (prueba)")
    args = ap.parse_args()
    pos, neg = pairs()
    neg = neg[neg.pair_star_id.isin(pos.star_id)]
    if args.limit:
        pos, neg = pos.head(args.limit), neg[neg.pair_star_id.isin(pos.head(args.limit).star_id)]
    print(f"parejas elegibles: {len(pos)} positivos ({pos.disposition.value_counts().to_dict()}), {len(neg)} negativos", flush=True)
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    done = done[done["error"].isna()] if len(done) else done      # las que fallaron se reintentan
    seen = set(done.star_id)
    done = run(pos[~pos.star_id.isin(seen)], done, args.workers)   # primero los positivos: fijan el sector preferido
    extra = done.dropna(subset=["sector_2"]).set_index("star_id").sector_2
    neg = neg.assign(preferred=neg.pair_star_id.map(extra).fillna(-1).astype(int))
    done = run(neg[~neg.star_id.isin(seen)], done, args.workers)
    print(done.groupby("disposition").agg(n=("star_id", "size"), errores=("error", "count"), validas=("valid", "sum"),
                                          transitos_mediana=("n_transits", "median")).to_string(), flush=True)


if __name__ == "__main__":
    main()
