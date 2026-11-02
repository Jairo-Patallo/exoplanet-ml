"""Dummy de la fase 2 sobre la muestra A (tarea #51): el listón dentro de cada misión, con las métricas de #51.

Por misión, predice siempre la clase más frecuente en su train (probabilidad 1 a esa clase). Se evalúa en val, que es
donde se compararán los modelos; el test no se toca. Solo estrellas con vista válida (qc.csv), como los modelos.
Con scores constantes, la PR-AUC de cada clase es su prevalencia y el F1 de las clases no predichas es 0.

Uso:  python scripts/models/dummy_sample.py   (PYTHONPATH=src)
Salida: results/models/dummy_val.csv (métricas por misión y media, con IC 95 %) y dummy_val_confusion.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet.data import CLASSES
from exoplanet.evaluation import evaluate_by_mission

OUT = Path("results/models")


def main() -> None:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool)]
    train, val = qc[qc.split == "train"], qc[qc.split == "val"].reset_index(drop=True)
    majority = train.groupby("mission").disposition.agg(lambda s: s.value_counts().idxmax())
    proba = np.stack([(val.mission.map(majority) == c).astype(float) for c in CLASSES], axis=1)
    table, conf = evaluate_by_mission(val.mission.to_numpy(), val.disposition.to_numpy(), proba)
    table.insert(1, "predice", table.mision.map(majority).fillna("-"))
    table.insert(2, "n_val", table.mision.map(val.mission.value_counts()).fillna(len(val)).astype(int))
    OUT.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(OUT / "dummy_val.csv", index=False, encoding="utf-8")
    pd.concat(conf, names=["mision", "real"]).to_csv(OUT / "dummy_val_confusion.csv", encoding="utf-8")
    cols = ["mision", "predice", "n_val", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"pr_auc_{c}" for c in CLASSES]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
