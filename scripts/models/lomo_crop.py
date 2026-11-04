"""LOMO de TESS con aumento "tipo TESS": Kepler + K2 originales más sus copias recortadas a 27,4 d (fase 3, 2026-11-04).

Mismo fold y configuración que scripts/models/adabn.py (base 16, dropout 0,5, 3 semillas), para comparar con 0,764.
Copias: data/sample/views_crop (scripts/data/build_cropped.py), solo las válidas y de estrellas del conjunto de ajuste.
La parada temprana aparta juntas las dos copias de cada estrella. TESS no entra en el entrenamiento.
Evalúa en TESS train + val (1 sector) y, como referencia, en el subconjunto de parejas con 2 sectores.

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/lomo_crop.py
Salida: results/models/lomo_crop_tess.csv; pesos en data/models/lomo_tess_crop_seed*.pt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import nested_folds, point_metrics, sample_meta, shared_stars
from exoplanet.training import fit, inputs, predict

OUT, MODELS = Path("results/models"), Path("data/models")
CFG, SEEDS, WEIGHT_DECAY = {"base": 16, "dropout": 0.5}, (0, 1, 2), 1e-4


def make(xf):
    return AstroCNN(CFG["base"], CFG["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6)


def row(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    m = point_metrics(y, p)
    return {"evaluacion": name, "n": len(y), "pr_auc_macro": m["pr_auc_macro"], **{f"pr_auc_{c}": m[f"pr_auc_{c}"] for c in CLASSES},
            "macro_f1": m["macro_f1"], **{f"f1_{c}": m[f"f1_{c}"] for c in CLASSES}}


def main() -> None:
    meta = sample_meta()
    fold = next(f for f in nested_folds(meta, shared_stars(meta), eval_splits=("train", "val")) if f.held_out == "tess")
    orig = meta.iloc[fold.fit_idx]
    qc = pd.read_csv("results/sample/qc_crop.csv")
    crop = orig[orig.star_id.isin(qc.star_id[qc.error.isna() & qc["valid"].eq(True)])]
    d_orig, d_crop = ViewDataset(orig), ViewDataset(crop, views_dir=DATA / "sample" / "views_crop")
    x = {k: torch.cat([v, inputs(d_crop)[k]]) for k, v in inputs(d_orig).items()}
    y = torch.cat([d_orig.y, d_crop.y])
    groups = np.concatenate([orig.star_id.to_numpy(), crop.star_id.to_numpy()])
    print(f"ajuste: {len(orig)} originales + {len(crop)} recortadas ({crop.disposition.value_counts().to_dict()})", flush=True)

    tess = meta.iloc[fold.eval_idx]
    x_eval, y_eval = inputs(ViewDataset(tess)), tess.disposition.to_numpy()
    q2 = pd.read_csv("results/sample/qc_tess2.csv")
    q2 = q2[q2.error.isna() & q2["valid"].eq(True)]
    sub = q2[q2.groupby("pair_id").star_id.transform("size").eq(2)].assign(mission="tess")
    x_sub = {n: inputs(ViewDataset(sub, views_dir=DATA / "sample" / d)) for n, d in (("1", "views"), ("2", "views_tess2"))}

    MODELS.mkdir(parents=True, exist_ok=True)
    models = []
    for s in SEEDS:
        m = fit(make, x, y, WEIGHT_DECAY, s, augment=True, amp=True, groups=groups)[0]
        torch.save(m.state_dict(), MODELS / f"lomo_tess_crop_seed{s}.pt")
        models.append(m)
        print(f"semilla {s} entrenada", flush=True)
    ens = lambda xx: np.mean([predict(m, xx, amp=True) for m in models], axis=0)
    rows = [row("TESS train+val, 1 sector", y_eval, ens(x_eval)),
            row("parejas 2 sectores: vista de 1 sector", sub.disposition.to_numpy(), ens(x_sub["1"])),
            row("parejas 2 sectores: vista de 2 sectores", sub.disposition.to_numpy(), ens(x_sub["2"]))]
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT / "lomo_crop_tess.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
