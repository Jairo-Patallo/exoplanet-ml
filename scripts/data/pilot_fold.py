"""Piloto de descarga y plegado (tarea #47): mide tamaño, tiempo y % de vistas válidas antes de fijar la muestra.

Toma una muestra estratificada por misión y clase de estrellas con etiqueta FUERTE (PLANET / FALSE_POSITIVE), descarga
sus curvas de MAST (exoplanet.curves), las pliega con exoplanet.folding y guarda las vistas y la calidad de cada estrella.
Los NO_SIGNAL no entran: sus negativos se generan aparte (tarea #32). Se puede interrumpir y relanzar: salta las ya hechas.

Uso:  python scripts/data/pilot_fold.py [--n 100] [--seed 0]     (desde la raíz, con PYTHONPATH=src)
Salida: data/pilot/views/<star_id>.npz (ignorado por git), results/pilot/qc.csv (una fila por estrella) y
        results/pilot/summary.csv (por misión: tamaño, tiempo, % válidas y coste estimado de la muestra completa).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet.curves import run_star, summarize
from exoplanet.data import DATA, MISSIONS, TARGET
from exoplanet.labels import strong_stars

VIEWS = DATA / "pilot" / "views"
OUT = Path("results/pilot")


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n estrellas por misión, mitad por clase (todas las que haya si una clase tiene menos)."""
    parts = []
    for mission in MISSIONS:
        for cls, g in df[df.mission == mission].groupby(TARGET):
            parts.append(g.sample(min(len(g), n // 2), random_state=seed))
    return pd.concat(parts, ignore_index=True)


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
            res = run_star(row, VIEWS)
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
