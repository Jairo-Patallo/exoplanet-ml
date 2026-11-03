"""Muestra A (decisión #49, 2026-11-01): descarga y pliega ~1000 Kepler, todo K2 y ~400 TESS con etiqueta fuerte.

Muestreo proporcional por misión, split y clase: conserva los splits de HF y la prevalencia real de FP de cada misión
(el piloto equilibraba las clases solo para medir). Las vistas no válidas se guardan igual y se marcan en qc.csv;
se descartan al cargar (opción a) y el sesgo se mide por misión y clase en summary_by_class.csv.
Descarga en paralelo con procesos (la descarga espera a la red, no a la CPU). Se puede interrumpir y relanzar: salta las ya hechas
y reintenta las que dieron error. Con --workers > 1 MAST corta algunas descargas (FITS truncados): usar 1 si pasa.

Uso:  python scripts/data/build_sample.py [--kepler 1000] [--tess 400] [--workers 4] [--limit N] [--negatives]   (PYTHONPATH=src)
Salida: data/sample/views/<star_id>.npz (ignorado por git), results/sample/qc.csv, summary.csv y summary_by_class.csv.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from exoplanet.curves import run_star, summarize
from exoplanet.data import DATA, TARGET
from exoplanet.labels import strong_stars

VIEWS = DATA / "sample" / "views"
OUT = Path("results/sample")
SEED = 0


def log(msg: str) -> None:
    print(msg, flush=True)


def job(row: SimpleNamespace) -> dict:
    """Una estrella. Va en un proceso aparte: astroquery comparte una sesión HTTP y con hilos las descargas se truncan."""
    try:
        res = run_star(row, VIEWS, retries=2)
    except Exception as e:      # una estrella rota no para la descarga; se registra y se sigue
        res = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"star_id": row.star_id, "mission": row.mission, "split": row.split, "disposition": getattr(row, TARGET),
            "period_days": row.period_days, "duration_hrs": row.duration_hrs, **res}


def proportional_sample(df: pd.DataFrame, n_per_mission: dict[str, int | None]) -> pd.DataFrame:
    """Por misión, la fracción n/N de cada estrato (split, clase); None = la misión entera."""
    parts = []
    for mission, n in n_per_mission.items():
        g = df[df.mission == mission]
        if n is None or n >= len(g):
            parts.append(g)
            continue
        frac = n / len(g)
        parts.append(g.groupby(["split", TARGET], group_keys=False).sample(frac=frac, random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def summarize_by_class(qc: pd.DataFrame) -> pd.DataFrame:
    """% de estrellas que acaban con vista válida por misión y clase: mide el sesgo de descartar las no válidas."""
    qc = qc.assign(valid=qc["valid"].fillna(False).astype(bool))
    return (qc.groupby(["mission", "disposition"])
              .agg(n=("star_id", "size"), n_error=("error", "count"), pct_valid=("valid", lambda v: round(100 * v.mean(), 1)))
              .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kepler", type=int, default=1000)
    ap.add_argument("--tess", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="solo las N primeras (prueba)")
    ap.add_argument("--negatives", action="store_true", help="añade los NO_SIGNAL de results/sample/negatives.csv (#32)")
    args = ap.parse_args()

    universe = strong_stars()
    chosen = proportional_sample(universe, {"kepler": args.kepler, "k2": None, "tess": args.tess})
    if args.negatives:
        chosen = pd.concat([chosen, pd.read_csv(OUT / "negatives.csv")], ignore_index=True)
    log(chosen.groupby(["mission", "split", TARGET]).size().unstack().to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    qc_path = OUT / "qc.csv"
    done = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()
    if len(done):                   # las que fallaron se reintentan al relanzar
        done = done[done["error"].isna()] if "error" in done else done
    seen = set(done.star_id) if len(done) else set()
    todo = [SimpleNamespace(**r) for r in chosen.to_dict("records") if r["star_id"] not in seen][:args.limit]

    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
            done.to_csv(qc_path.with_suffix(".tmp"), index=False, encoding="utf-8")
            qc_path.with_suffix(".tmp").replace(qc_path)   # atómico: un apagón no deja qc.csv a medias (pasó el 2026-11-02)
            msg = rec.get("error") if isinstance(rec.get("error"), str) else f"valid={rec['valid']} {rec['mb_downloaded']:.1f} MB"
            log(f"[{i}/{len(todo)}] {rec['star_id']} {rec['mission']}: {msg}")

    if "error" not in done:
        done["error"] = np.nan
    summarize(done, universe).to_csv(OUT / "summary.csv", index=False, encoding="utf-8")
    by_class = summarize_by_class(done)
    by_class.to_csv(OUT / "summary_by_class.csv", index=False, encoding="utf-8")
    log("\n" + by_class.to_string(index=False))


if __name__ == "__main__":
    main()
