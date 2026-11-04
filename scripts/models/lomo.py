"""Fase 3: transferencia entre misiones con el protocolo LOMO anidado (research/03-protocolo-evaluacion.md).

Por cada misión excluida M (fold externo), las otras dos (A, B) son las únicas que deciden:
  1. Selección (folds internos): cada configuración se entrena con train de A y se mide en val de B, y viceversa;
     se elige la de mayor PR-AUC macro medio (sin umbral, como fija el protocolo).
  2. Reentrenamiento con train + val de A y B (sin las estrellas que M también observa: evaluation.shared_stars),
     3 semillas, ensemble de sus probabilidades.
  3. Evaluación en M, una sola vez: PR-AUC por clase y macro (principal en LOMO) y macro-F1 con argmax (referencia).
Modelos: dummy (clase más frecuente en A + B) y CNN solo curva (la de la fase 2), opcionalmente con centroide.

Uso (WSL, GPU):  export HSA_ENABLE_DXG_DETECTION=1
  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/lomo.py --model cnn --amp [--centroid] [--ephem]
  --eval-splits train val      evalúa M sin su test (para iterar sin gastarlo; decisión pendiente del usuario)
  --smoke                      prueba rápida con pocas estrellas, sin guardar resultados
Salida: results/models/lomo_<modelo>.csv (métricas por misión excluida) y lomo_<modelo>_inner.csv (selección).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN, feat_stats
from exoplanet.data import CLASSES, SPLITS
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import nested_folds, point_metrics, sample_meta, shared_stars
from exoplanet.training import fit, inputs, predict

OUT = Path("results/models")
GRID = [{"base": b, "dropout": d} for b in (8, 16) for d in (0.3, 0.5)]
WEIGHT_DECAY, SEEDS = 1e-4, (0, 1, 2)


def train_cnn(x: dict, y: torch.Tensor, cfg: dict, seed: int, amp: bool):
    def make(xf):
        return AstroCNN(cfg["base"], cfg["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6,
                        channels=xf["local"].shape[1], glob_channels=xf["glob"].shape[1],
                        feat_stats=feat_stats(xf["feats"]) if "feats" in xf else None)
    return fit(make, x, y, WEIGHT_DECAY, seed, augment=True, amp=amp)[0]


def metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    m = point_metrics(y, proba)
    return {k: m[k] for k in ["pr_auc_macro", *[f"pr_auc_{c}" for c in CLASSES], "macro_f1", *[f"f1_{c}" for c in CLASSES]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dummy", "cnn"], default="cnn")
    ap.add_argument("--centroid", action="store_true")
    ap.add_argument("--ephem", action="store_true", help="log(duración/periodo) y log(periodo) en la cabeza")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval-splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    grid, seeds = (GRID[:1], SEEDS[:1]) if args.smoke else (GRID, SEEDS)

    meta = sample_meta()
    if args.smoke:
        meta = meta.groupby(["mission", "split"], group_keys=False).apply(lambda g: g.sample(min(len(g), 150), random_state=0))
        meta = meta.reset_index(drop=True)
    folds = nested_folds(meta, shared_stars(meta), eval_splits=tuple(args.eval_splits))
    y_all = meta.disposition.to_numpy()
    name = f"lomo_{args.model}" + ("_centroid" if args.centroid else "") + ("_ephem" if args.ephem else "")
    rows, inner_rows = [], []
    for f in folds:
        print(f"== excluida {f.held_out}: ajuste {len(f.fit_idx)} ({'+'.join(f.fit_missions)}), evaluación {len(f.eval_idx)}", flush=True)
        y_eval = y_all[f.eval_idx]
        if args.model == "dummy":
            pred = pd.Series(y_all[f.fit_idx]).value_counts().idxmax()
            proba = np.tile([float(c == pred) for c in CLASSES], (len(f.eval_idx), 1))
            rows.append({"excluida": f.held_out, "n_eval": len(f.eval_idx), "config": f"predice {pred}", **metrics(y_eval, proba)})
            continue
        ds = lambda idx: ViewDataset(meta.iloc[idx], centroid=args.centroid, ephem=args.ephem)
        scores = []
        for cfg in grid:
            s = []
            for fit_idx, val_idx in f.inner:
                d_fit, d_val = ds(fit_idx), ds(val_idx)
                model = train_cnn(inputs(d_fit), d_fit.y, cfg, seeds[0], args.amp)
                s.append(point_metrics(y_all[val_idx], predict(model, inputs(d_val), amp=args.amp))["pr_auc_macro"])
            scores.append(np.mean(s))
            inner_rows.append({"excluida": f.held_out, **cfg, "pr_auc_interno_A_B": s[0], "pr_auc_interno_B_A": s[1], "media": scores[-1]})
            print(f"   {cfg}: PR-AUC interno {s[0]:.3f} / {s[1]:.3f}", flush=True)
        cfg = grid[int(np.argmax(scores))]
        d_fit, d_eval = ds(f.fit_idx), ds(f.eval_idx)
        probas = [predict(train_cnn(inputs(d_fit), d_fit.y, cfg, seed, args.amp), inputs(d_eval), amp=args.amp) for seed in seeds]
        res = {"excluida": f.held_out, "n_eval": len(f.eval_idx), "config": str(cfg), **metrics(y_eval, np.mean(probas, axis=0))}
        rows.append(res)
        print(f"   {f.held_out}: PR-AUC macro {res['pr_auc_macro']:.3f}, macro-F1 {res['macro_f1']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    print("\n" + out.round(3).to_string(index=False))
    if args.smoke:
        print("\n(--smoke: no se guarda nada)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    if inner_rows:
        pd.DataFrame(inner_rows).round(4).to_csv(OUT / f"{name}_inner.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
