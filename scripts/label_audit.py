"""Auditoría de etiquetas (tarea #33): qué significa PLANET / FALSE_POSITIVE en cada misión.

Cruza las filas del dataset de HF con las disposiciones originales de TESS (TOI) y K2 (NASA Exoplanet Archive,
scripts/fetch_labels.py) y cuenta la fuerza de cada etiqueta (exoplanet.labels). Kepler no necesita cruce: el dataset
ya trae label_source (kepler_confirmed / kepler_robovetter).

Uso:  python scripts/label_audit.py   (desde la raíz del repo, tras scripts/fetch_labels.py)
Salida: results/labels/audit.csv (cruce completo) y results/labels/strength_counts.csv (fuerza por misión y clase)
"""
from pathlib import Path

import pandas as pd

from exoplanet.data import SPLITS, TARGET, load_split
from exoplanet.labels import external_disposition, label_strength

OUT = Path("results/labels")


def main() -> None:
    cols = ["star_id", "mission", TARGET, "label_source", "period_days"]
    df = pd.concat([load_split(s, columns=cols) for s in SPLITS], ignore_index=True)
    df = df[df[TARGET] != "NO_SIGNAL"].copy()
    df["original"] = external_disposition(df)
    df["fuerza"] = label_strength(df)

    audit = df.groupby(["mission", "label_source", TARGET, "original", "fuerza"]).size().rename("n").reset_index()
    counts = df.groupby(["mission", TARGET, "fuerza"]).size().unstack("fuerza", fill_value=0)
    strong = counts["fuerte"].unstack(TARGET)
    counts["total"] = counts.sum(axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / "audit.csv", index=False, encoding="utf-8")
    counts.reset_index().to_csv(OUT / "strength_counts.csv", index=False, encoding="utf-8")
    print(audit.to_string(index=False))
    print("\nfuerza por misión y clase:\n" + counts.to_string())
    all_fp = df[df[TARGET] == "FALSE_POSITIVE"].groupby("mission").size() / df.groupby("mission").size()
    strong_fp = strong["FALSE_POSITIVE"] / (strong["FALSE_POSITIVE"] + strong["PLANET"])
    print("\nprevalencia de FP: todas las etiquetas vs solo fuertes")
    print(pd.DataFrame({"todas": all_fp.round(3), "solo_fuertes": strong_fp.round(3)}).to_string())


if __name__ == "__main__":
    main()
