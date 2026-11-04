"""Adaptación sin etiquetas a TESS con AdaBN (fase 3, 2026-11-04; Li et al. 2016, "Revisiting Batch Normalization").

Escenario: TESS como telescopio nuevo del que solo hay curvas, sin etiquetas. Mismo fold que LOMO (ajuste con Kepler +
K2 train + val, sin estrellas compartidas; evaluación en TESS train + val) y la configuración que eligió LOMO para TESS.
AdaBN no reentrena: recalcula la media y la varianza de cada BatchNorm con las curvas de TESS y deja los pesos igual.
Variantes (las etiquetas de TESS solo se usan para medir):
  base        el modelo LOMO tal cual (debería reproducir ~0,764)
  adabn       BatchNorm con las estadísticas de las curvas de TESS evaluadas (transductivo)
  adabn_prof  además, log10 de la profundidad estandarizado con la media y desviación de TESS
Kepler y K2 no cambian: el modelo de origen no se toca; la adaptación solo se aplica al predecir TESS.
Los pesos de las 3 semillas se guardan en data/models/lomo_tess_seed*.pt y se reutilizan si existen.

Uso (WSL, GPU):  export HSA_ENABLE_DXG_DETECTION=1
                 PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/adabn.py
Salida: results/models/adabn_tess.csv y adabn_tess_pred.csv
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import nested_folds, point_metrics, sample_meta, shared_stars
from exoplanet.training import DEVICE, fit, inputs, predict

OUT, MODELS = Path("results/models"), Path("data/models")
CFG = {"base": 16, "dropout": 0.5}            # elegida por LOMO para TESS (results/models/lomo_cnn_inner.csv)
SEEDS, WEIGHT_DECAY = (0, 1, 2), 1e-4


def make(xf):
    return AstroCNN(CFG["base"], CFG["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6,
                    channels=xf["local"].shape[1], glob_channels=xf["glob"].shape[1])


def source_model(x_fit: dict, y_fit: torch.Tensor, seed: int) -> nn.Module:
    path = MODELS / f"lomo_tess_seed{seed}.pt"
    if path.exists():
        model = make(x_fit).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return model.eval()
    model = fit(make, x_fit, y_fit, WEIGHT_DECAY, seed, augment=True, amp=True)[0]
    torch.save(model.state_dict(), path)
    return model


@torch.no_grad()
def adabn(model: nn.Module, x: dict, depth: bool = False) -> nn.Module:
    """Copia del modelo con las BatchNorm recalculadas sobre x (media acumulada exacta: momentum=None).
    Solo las BatchNorm pasan a modo train; el dropout sigue apagado. Sin gradientes: los pesos no cambian."""
    m = copy.deepcopy(model).eval()
    if depth:
        m.depth_mean.copy_(x["log_depth"].mean(0))
        m.depth_std.copy_(x["log_depth"].std(0) + 1e-6)
    for bn in (mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)):
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    for c in torch.arange(len(x["local"])).split(512):    # sin bf16: las estadísticas en fp32
        m(**{k: v[c].to(DEVICE) for k, v in x.items()})
    return m.eval()


def main() -> None:
    meta = sample_meta()
    fold = next(f for f in nested_folds(meta, shared_stars(meta), eval_splits=("train", "val")) if f.held_out == "tess")
    d_fit, d_eval = ViewDataset(meta.iloc[fold.fit_idx]), ViewDataset(meta.iloc[fold.eval_idx])
    x_fit, x_eval, y_eval = inputs(d_fit), inputs(d_eval), meta.disposition.to_numpy()[fold.eval_idx]
    print(f"ajuste {len(fold.fit_idx)} (Kepler + K2), evaluación TESS {len(fold.eval_idx)}", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    models = [source_model(x_fit, d_fit.y, s) for s in SEEDS]
    variants = {"base": lambda m: m, "adabn": lambda m: adabn(m, x_eval), "adabn_prof": lambda m: adabn(m, x_eval, depth=True)}
    rows, preds = [], meta.iloc[fold.eval_idx][["star_id", "disposition"]].copy()
    for name, adapt in variants.items():
        proba = np.mean([predict(adapt(m), x_eval, amp=True) for m in models], axis=0)
        r = point_metrics(y_eval, proba)
        rows.append({"variante": name, "n_eval": len(y_eval), "pr_auc_macro": r["pr_auc_macro"],
                     **{f"pr_auc_{c}": r[f"pr_auc_{c}"] for c in CLASSES}, "macro_f1": r["macro_f1"],
                     **{f"f1_{c}": r[f"f1_{c}"] for c in CLASSES}})
        preds = preds.assign(**{f"{name}_p_{c}": proba[:, i] for i, c in enumerate(CLASSES)})
        print(f"{name:11s} PR-AUC macro {r['pr_auc_macro']:.3f}, macro-F1 {r['macro_f1']:.3f}, F1 FP {r['f1_FALSE_POSITIVE']:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(OUT / "adabn_tess.csv", index=False, encoding="utf-8")
    preds.round(4).to_csv(OUT / "adabn_tess_pred.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
