"""Prueba barata de pseudo-etiquetas en TESS (fase 3, variante d del protocolo: adaptación sin etiquetas; 2026-11-04).

TESS train + val (2 634 estrellas, fold LOMO) se parte en dos mitades estratificadas. Para cada mitad A:
  1. El modelo LOMO (Kepler + K2, 3 semillas; data/models/lomo_tess_seed*.pt) predice A sin mirar sus etiquetas.
  2. Pseudo-etiqueta = clase predicha, solo si el ensemble da prob >= TAU y las 3 semillas coinciden.
  3. Se reentrena con Kepler + K2 + las pseudo-etiquetadas de A (misma configuración, 3 semillas).
  4. Se evalúa en la otra mitad B, que no se ha usado para nada.
Las predicciones de las dos mitades se juntan y se comparan con el modelo LOMO en las mismas 2 634 estrellas (IC 95 %
bootstrap pareado). Las etiquetas reales de A solo se usan para informar de la calidad de las pseudo-etiquetas.
Criterio fijado antes de ejecutar: sale bien si el límite inferior del IC de la mejora en PR-AUC macro es > 0.

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/pseudo_labels.py
Salida: results/models/pseudo_labels_tess.csv y pseudo_labels_tess_pred.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import nested_folds, point_metrics, sample_meta, shared_stars
from exoplanet.training import DEVICE, fit, inputs, predict, take

OUT, MODELS = Path("results/models"), Path("data/models")
CFG, SEEDS, WEIGHT_DECAY, TAU, N_BOOT = {"base": 16, "dropout": 0.5}, (0, 1, 2), 1e-4, 0.9, 1000


def make(xf):
    return AstroCNN(CFG["base"], CFG["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6)


def lomo_model(seed: int) -> AstroCNN:
    m = AstroCNN(CFG["base"], CFG["dropout"], torch.zeros(1), torch.ones(1))
    m.load_state_dict(torch.load(MODELS / f"lomo_tess_seed{seed}.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def main() -> None:
    meta = sample_meta()
    fold = next(f for f in nested_folds(meta, shared_stars(meta), eval_splits=("train", "val")) if f.held_out == "tess")
    d_fit = ViewDataset(meta.iloc[fold.fit_idx])
    x_fit, y_fit = inputs(d_fit), d_fit.y
    tess = meta.iloc[fold.eval_idx].reset_index(drop=True)
    x_t, y_t = inputs(ViewDataset(tess)), tess.disposition.to_numpy()

    per_seed = np.stack([predict(lomo_model(s), x_t, amp=True) for s in SEEDS])   # (3, n, 3)
    base = per_seed.mean(0)
    agree = (per_seed.argmax(2) == per_seed[0].argmax(1)).all(0)
    confident = agree & (base.max(1) >= TAU)

    adapted, info = np.zeros_like(base), []
    halves = list(StratifiedKFold(2, shuffle=True, random_state=0).split(tess, y_t))
    for h, (a, b) in enumerate(halves):
        sel = a[confident[a]]
        pseudo = torch.as_tensor(base[sel].argmax(1))
        x = {k: torch.cat([v, take(x_t, torch.as_tensor(sel))[k]]) for k, v in x_fit.items()}
        y = torch.cat([y_fit, pseudo])
        models = [fit(make, x, y, WEIGHT_DECAY, s, augment=True, amp=True)[0] for s in SEEDS]
        x_b = take(x_t, torch.as_tensor(b))
        adapted[b] = np.mean([predict(m, x_b, amp=True) for m in models], axis=0)
        pred = np.asarray(CLASSES)[pseudo.numpy()]
        info.append({"mitad": h, "n_mitad": len(a), "n_pseudo": len(sel), "acierto_pseudo": (pred == y_t[sel]).mean(),
                     **{f"pseudo_{c}": int((pred == c).sum()) for c in CLASSES},
                     **{f"pseudo_{c}_acierto": (y_t[sel][pred == c] == c).mean() for c in CLASSES}})
        print(info[-1], flush=True)

    rng, diff = np.random.default_rng(0), []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y_t), len(y_t))
        diff.append(point_metrics(y_t[i], adapted[i])["pr_auc_macro"] - point_metrics(y_t[i], base[i])["pr_auc_macro"])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    rows = []
    for name, p in (("LOMO", base), ("LOMO + pseudo-etiquetas", adapted)):
        m = point_metrics(y_t, p)
        rows.append({"modelo": name, "n": len(y_t), "pr_auc_macro": m["pr_auc_macro"],
                     **{f"pr_auc_{c}": m[f"pr_auc_{c}"] for c in CLASSES}, "macro_f1": m["macro_f1"],
                     **{f"f1_{c}": m[f"f1_{c}"] for c in CLASSES}})
    out = pd.DataFrame(rows).assign(dif_pr_auc_ic95=["", f"[{lo:+.3f}, {hi:+.3f}]"])
    print(out.round(3).to_string(index=False))
    print(f"\nSALE BIEN: {lo > 0}  (IC 95 % de la mejora en PR-AUC: [{lo:+.3f}, {hi:+.3f}])", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([out, pd.DataFrame(info)]).round(4).to_csv(OUT / "pseudo_labels_tess.csv", index=False, encoding="utf-8")
    tess[["star_id", "disposition"]].assign(confiable=confident, **{f"lomo_p_{c}": base[:, i] for i, c in enumerate(CLASSES)},
                                            **{f"pseudo_p_{c}": adapted[:, i] for i, c in enumerate(CLASSES)}
                                            ).round(4).to_csv(OUT / "pseudo_labels_tess_pred.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
