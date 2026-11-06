"""TESS con varios sectores por estrella, mismo número dentro de cada pareja (fase 6, opción A de Jairo, 2026-11-06).

Cada pareja de TESS (positivo, su NO_SIGNAL) usa k = mín(sectores del positivo, sectores del negativo) sectores de 2 min,
contados en MAST en el momento de construir (el recuento de data/sample/tess_sector_counts.csv solo preselecciona). Así ninguna clase tiene curvas más limpias por tener más sectores: con todos los
sectores, el número de sectores solo ya separa planeta de FP con AUC 0,71 (en planetas, mediana 7; en sin señal, 2).
Cada estrella usa su sector original (el de la vista de 1 sector) más k − 1 sectores elegidos al azar (semilla fija por
estrella). Mismo plegado que el modelo final (curves.read_prepare + folding.fold_views, efeméride del catálogo; el
negativo usa la de su pareja).

data/sample/views_tess_multi tiene TODAS las estrellas de la muestra, para que el modelo lea de una sola carpeta:
  - parejas con k >= 2: vista nueva;
  - el resto (Kepler, K2, TESS sin pareja o con k = 1): copia de la vista de 1 sector;
  - si un miembro de la pareja falla o su vista no es válida, los dos vuelven a la vista de 1 sector (la pareja sigue
    igualada).

Uso:  PYTHONPATH=src python scripts/data/build_tess_multisector.py [--workers 6] [--limit N]   (reanudable)
Salida: data/sample/views_tess_multi/<star_id>.npz y results/sample/qc_tess_multi.csv
"""
import argparse
import shutil
import socket
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.mast import conf

from exoplanet.curves import RAW, SEARCH, read_prepare, target_name
from exoplanet.data import DATA
from exoplanet.folding import fold_views
from exoplanet.labels import strong_stars
from exoplanet.views import SAMPLE_VIEWS

VIEWS = DATA / "sample" / "views_tess_multi"
QC = Path("results/sample/qc_tess_multi.csv")
conf.timeout = 120
socket.setdefaulttimeout(120)


def available(row: SimpleNamespace):
    """Búsqueda en MAST: (resultado, índice del sector original, índices de los demás sectores sin duplicar)."""
    res = lk.search_lightcurve(target_name(row.star_id), **SEARCH["tess"])
    seqs = np.asarray(res.table["sequence_number"]).astype(int)
    if row.sector not in seqs:
        raise ValueError(f"sector {row.sector} no está en MAST")
    others = [i for i in range(len(res)) if seqs[i] != row.sector and seqs[i] not in seqs[:i]]
    return res, int(np.flatnonzero(seqs == row.sector)[0]), others


def build(row: SimpleNamespace, res, first: int, others: list, k: int) -> dict:
    seqs = np.asarray(res.table["sequence_number"]).astype(int)
    rng = np.random.default_rng(zlib.crc32(row.star_id.encode()))
    pick = [first] + [int(i) for i in rng.choice(others, k - 1, replace=False)]
    paths = [Path(res[i].download(download_dir=str(RAW)).filename) for i in pick]
    T = row.duration_hrs / 24
    views = fold_views(*read_prepare(paths, row.period_days, row.epoch_bjd, T), row.period_days, row.epoch_bjd, T)
    np.savez_compressed(VIEWS / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    return {"k": k, "sectores": " ".join(str(int(seqs[i])) for i in pick), **views.qc}


def job(pair: list) -> list:
    """Las dos estrellas de una pareja: k = mínimo de los sectores realmente disponibles en MAST."""
    base = [{"star_id": r.star_id, "pair_id": r.pair_id, "split": r.split, "disposition": r.disposition} for r in pair]
    try:
        found = [available(r) for r in pair]
        k = min(min(len(o) + 1 for _, _, o in found), pair[0].k)
        if k < 2:
            return [{**b, "k": 1, "error": "sin sectores extra en MAST"} for b in base]
        return [{**b, **build(r, *f, k)} for b, r, f in zip(base, pair, found)]
    except Exception as e:
        return [{**b, "error": f"{type(e).__name__}: {str(e)[:120]}"} for b in base]


def pairs() -> pd.DataFrame:
    """Una fila por estrella de las parejas de TESS válidas, con su sector original, su efeméride y k."""
    counts = pd.read_csv("data/sample/tess_sector_counts.csv").set_index("star_id").n_sectors
    qc = pd.read_csv("results/sample/qc.csv")
    ok = set(qc.star_id[qc["valid"].eq(True) & (qc.mission == "tess")])
    neg = pd.read_csv("results/sample/negatives.csv").query("mission == 'tess'")
    neg = neg[neg.star_id.isin(ok) & neg.pair_star_id.isin(ok)].drop_duplicates("pair_star_id")
    k = np.minimum(neg.star_id.map(counts), neg.pair_star_id.map(counts)).fillna(1).astype(int).to_numpy()
    cols = ["star_id", "split", "disposition", "period_days", "epoch_bjd", "duration_hrs"]
    pos = strong_stars().drop_duplicates("star_id").set_index("star_id").loc[neg.pair_star_id].reset_index()[cols]
    pos = pos.assign(pair_id=pos.star_id, sector=neg.segment.astype(int).to_numpy(), k=k)
    neg = neg[cols].assign(pair_id=neg.pair_star_id.to_numpy(), sector=neg.segment.astype(int).to_numpy(), k=k)
    return pd.concat([pos, neg], ignore_index=True)


def copy_rest(multi: set) -> int:
    """Copia la vista de 1 sector de todas las estrellas que no tienen vista multisector."""
    n = 0
    for f in SAMPLE_VIEWS.glob("*.npz"):
        if f.stem not in multi and not (VIEWS / f.name).exists():
            shutil.copy2(f, VIEWS / f.name)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="solo N parejas con k >= 2 (prueba)")
    args = ap.parse_args()
    VIEWS.mkdir(parents=True, exist_ok=True)
    rows = pairs()
    multi = rows[rows.k >= 2]
    if args.limit:
        multi = multi[multi.pair_id.isin(multi.pair_id.drop_duplicates().head(args.limit))]
    done = pd.read_csv(QC) if QC.exists() else pd.DataFrame(columns=["star_id", "error"])
    if len(done):                                   # una pareja con algún error se repite entera
        bad = set(done.pair_id[done["error"].notna()])
        done = done[~done.pair_id.isin(bad)]
    pend = multi[~multi.pair_id.isin(set(done.pair_id.dropna()))] if len(done) else multi
    todo = [[SimpleNamespace(**r) for r in g.to_dict("records")] for _, g in pend.groupby("pair_id")]
    print(f"parejas: {rows.pair_id.nunique()}, con k >= 2 según el catálogo: {multi.pair_id.nunique()} "
          f"({int(multi.k.sum())} sectores); faltan {len(todo)} parejas", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed([pool.submit(job, p) for p in todo]), 1):
            done = pd.concat([done, pd.DataFrame(fut.result())], ignore_index=True)
            if i % 20 == 0 or i == len(todo):
                done.to_csv(QC.with_suffix(".tmp"), index=False, encoding="utf-8")
                QC.with_suffix(".tmp").replace(QC)
                print(f"[{i}/{len(todo)}] errores {done.error.notna().sum()}", flush=True)

    # parejas incompletas (error o vista no válida en algún miembro): las dos vuelven a 1 sector
    good = done[done.error.isna() & done["valid"].eq(True)]
    ok_pairs = good.groupby("pair_id").star_id.count().pipe(lambda s: set(s[s == 2].index))
    keep = set(good.star_id[good.pair_id.isin(ok_pairs)])
    for s in set(done.star_id) - keep:
        (VIEWS / f"{s}.npz").unlink(missing_ok=True)
    done["usada"] = done.star_id.isin(keep)
    done.to_csv(QC, index=False, encoding="utf-8")
    print(f"copiadas {copy_rest(keep)} vistas de 1 sector; parejas multisector usadas: {len(ok_pairs)}", flush=True)
    print(done[done.usada].groupby("disposition").agg(n=("star_id", "size"), k_mediana=("k", "median"),
                                                       transitos_mediana=("n_transits", "median")).to_string(), flush=True)


if __name__ == "__main__":
    main()
