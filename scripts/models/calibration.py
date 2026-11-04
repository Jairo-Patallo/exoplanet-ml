"""Calibración y cambio de prior en LOMO: ¿qué umbral usar en un telescopio sin etiquetas? (fase 3, 2026-11-04)

Entrada: predicciones LOMO por estrella (results/models/lomo_cnn_fixed_pred.csv; el modelo nunca vio esa misión).
La CNN se entrena con pesos de clase "balanced": sus probabilidades suponen clases equiprobables (prior 1/3).
En un telescopio nuevo la mezcla de clases es otra (cambio de prior). Se comparan, por misión excluida:
  argmax        clase más probable, tal cual
  EM            Saerens et al. (2002): estima la mezcla de clases del telescopio nuevo con las propias predicciones,
                sin etiquetas, y reescala p(c|x) · prior_nuevo(c) / (1/3)
  oráculo       el mismo reescalado con la mezcla real (usa etiquetas: solo como techo de referencia)
Métricas: macro-F1, F1 por clase, ECE (calibración de la clase predicha, 10 bins), Brier multiclase y recall a
precisión fija (0,9) de PLANET y FALSE_POSITIVE (no depende del prior de entrenamiento, sí de la mezcla real).
Figura: curvas de fiabilidad (argmax) por misión excluida.

Uso:  PYTHONPATH=src python scripts/models/calibration.py
Salida: results/models/calibration_lomo.csv y results/models/calibration_lomo.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve

from exoplanet.data import CLASSES

PRED, OUT = Path("results/models/lomo_cnn_fixed_pred.csv"), Path("results/models/calibration_lomo.csv")
PRECISION = 0.9
TRAIN_PRIOR = np.full(len(CLASSES), 1 / len(CLASSES))


def rescale(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    q = p * prior / TRAIN_PRIOR
    return q / q.sum(1, keepdims=True)


def em_prior(p: np.ndarray, iters: int = 100, tol: float = 1e-6) -> np.ndarray:
    prior = TRAIN_PRIOR.copy()
    for _ in range(iters):
        new = rescale(p, prior).mean(0)
        if np.abs(new - prior).max() < tol:
            break
        prior = new
    return prior


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    conf, correct = p.max(1), (p.argmax(1) == y)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def recall_at_precision(y: np.ndarray, score: np.ndarray, target: float = PRECISION) -> float:
    prec, rec, _ = precision_recall_curve(y, score)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def reliability(ax, y: np.ndarray, p: np.ndarray, label: str, bins: int = 10) -> None:
    conf, correct = p.max(1), (p.argmax(1) == y)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    used = [b for b in range(bins) if (idx == b).sum() >= 10]
    ax.plot([conf[idx == b].mean() for b in used], [correct[idx == b].mean() for b in used], "o-", label=label)


def main() -> None:
    pred = pd.read_csv(PRED)
    rows = []
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibración perfecta")
    for mission, g in pred.groupby("mission", sort=False):
        y = g.disposition.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        p = g[[f"p_{c}" for c in CLASSES]].to_numpy()
        true_prior = np.bincount(y, minlength=len(CLASSES)) / len(y)
        est = em_prior(p)
        reliability(ax, y, p, mission)
        r_at_p = {f"recall@P{PRECISION}_{c}": recall_at_precision(y == i, p[:, i]) for i, c in enumerate(CLASSES[:2])}
        for name, q, prior in (("argmax", p, TRAIN_PRIOR), ("EM", rescale(p, est), est), ("oráculo", rescale(p, true_prior), true_prior)):
            f1 = f1_score(y, q.argmax(1), average=None, labels=range(len(CLASSES)))
            rows.append({"excluida": mission, "metodo": name, "n": len(y), "macro_f1": f1.mean(),
                         **{f"f1_{c}": v for c, v in zip(CLASSES, f1)}, "ece": ece(y, q),
                         "brier": np.mean(np.sum((q - np.eye(len(CLASSES))[y]) ** 2, axis=1)), **r_at_p,
                         **{f"prior_{c}": v for c, v in zip(CLASSES, prior)},
                         **{f"prior_real_{c}": v for c, v in zip(CLASSES, true_prior)}})
    ax.set(xlabel="confianza de la clase predicha", ylabel="acierto real", title="Fiabilidad en LOMO (argmax)",
           xlim=(0.3, 1), ylim=(0.3, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=120)
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.round(4).to_csv(OUT, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
