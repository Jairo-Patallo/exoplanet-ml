"""Curva de aprendizaje en TESS: ¿cuántas etiquetas de un telescopio nuevo hacen falta? (fase 3, 2026-11-04).

Escenario: un telescopio nuevo (TESS) del que se etiquetan unas pocas estrellas. Se entrena la CNN solo curva con todo
Kepler + K2 (train + val, sin las estrellas que TESS también observa) más una fracción del train de TESS, sorteada por
clase, y se evalúa siempre en el val de TESS (nunca entrena; el test no se toca). La configuración es la que eligió LOMO
para TESS en sus folds internos (no se reelige mirando TESS). 3 semillas en ensemble por punto.
Fracción 0 = LOMO; fracción 1 = con todo el train de TESS.

Uso (WSL, GPU):  export HSA_ENABLE_DXG_DETECTION=1
                 PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/learning_curve.py
Salida: results/models/learning_curve_tess.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import point_metrics, sample_meta, shared_stars
from exoplanet.training import fit, inputs, predict

OUT = Path("results/models/learning_curve_tess.csv")
CFG = {"base": 16, "dropout": 0.5}            # elegida por LOMO para TESS (results/models/lomo_cnn.csv)
FRACTIONS = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS, WEIGHT_DECAY = (0, 1, 2), 1e-4


def train(x: dict, y: torch.Tensor, seed: int):
    def make(xf):
        return AstroCNN(CFG["base"], CFG["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6,
                        channels=xf["local"].shape[1], glob_channels=xf["glob"].shape[1])
    return fit(make, x, y, WEIGHT_DECAY, seed, augment=True, amp=True)[0]


def main() -> None:
    meta = sample_meta(("train", "val"))
    shared = shared_stars(meta)["tess"]
    base = meta[meta.mission.isin(["kepler", "k2"]) & ~meta.star_id.isin(shared)]
    tess_train, tess_val = meta[(meta.mission == "tess") & (meta.split == "train")], meta[(meta.mission == "tess") & (meta.split == "val")]
    x_val, y_val = inputs(ViewDataset(tess_val)), tess_val.disposition.to_numpy()
    short = (tess_val.disposition == "PLANET") & (tess_val.period_days < 2)
    rows = []
    for frac in FRACTIONS:
        extra = tess_train.groupby("disposition", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=0))
        ds = ViewDataset(pd.concat([base, extra]))
        p = np.mean([predict(train(inputs(ds), ds.y, s), x_val, amp=True) for s in SEEDS], axis=0)
        m = point_metrics(y_val, p)
        pred = np.asarray(CLASSES)[p.argmax(1)]
        rows.append({"fraccion_tess": frac, "n_tess_etiquetadas": len(extra), "pr_auc_macro": m["pr_auc_macro"],
                     "macro_f1": m["macro_f1"], **{f"f1_{c}": m[f"f1_{c}"] for c in CLASSES},
                     "acierto_planetas_P<2d": (pred[short.to_numpy()] == "PLANET").mean()})
        print(f"fracción {frac:.2f} ({len(extra)} de TESS): PR-AUC {m['pr_auc_macro']:.3f}, macro-F1 {m['macro_f1']:.3f}", flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
