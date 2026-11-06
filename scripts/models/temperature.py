"""Calibración del modelo final con temperature scaling por misión (fase 6, decisión de Jairo 2026-11-06).

p_T = softmax(log p / T): un solo número por misión que suaviza (T > 1) o agudiza (T < 1) las probabilidades del
ensemble. No cambia la clase de mayor probabilidad, así que el macro-F1 y el test oficial (research/10) no cambian.
Además se ajusta una T global de respaldo para telescopios nuevos.

Datos: results/models/final_val_pred.csv, la misma receta entrenada solo con train (5 semillas) y predicha en val.
El modelo final usa train + val, así que su predicción en val no sirve para calibrar; las T de aquí se le aplican.
Evaluación honesta dentro de val: 5 folds estratificados por clase en cada misión. T se ajusta en 4 folds y se mide
en el quinto (NLL, Brier, ECE de la clase predicha con 10 tramos). Después se ajusta T con todo val y se guarda.

Uso:  PYTHONPATH=src .venv/Scripts/python scripts/models/temperature.py
Salida: data/models/final_temperature.json, results/models/calibration_final.csv y calibration_final.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from calibration import ece, reliability
from exoplanet.data import CLASSES, DATA, MISSIONS

OUT = "results/models"


def scale(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-7, 1)) / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def nll(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1))))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


def fit_T(y: np.ndarray, p: np.ndarray) -> float:
    return float(minimize_scalar(lambda T: nll(y, scale(p, T)), bounds=(0.05, 20), method="bounded").x)


def metrics(y, p) -> dict:
    return {"nll": nll(y, p), "brier": brier(y, p), "ece": ece(y, p), "confianza_media": float(p.max(1).mean()),
            "acierto": float((p.argmax(1) == y).mean())}


def main() -> None:
    d = pd.read_csv(f"{OUT}/final_val_pred.csv")
    p = d[[f"p_{c}" for c in CLASSES]].to_numpy()
    y = d.disposition.map(CLASSES.index).to_numpy()
    groups = {m: (d.mission == m).to_numpy() for m in MISSIONS} | {"global": np.ones(len(d), bool)}
    rows, oof, temps = [], {}, {}
    for g, k in groups.items():
        yg, pg, out = y[k], p[k], np.zeros_like(p[k])
        for fit, ev in StratifiedKFold(5, shuffle=True, random_state=0).split(pg, yg):
            out[ev] = scale(pg[ev], fit_T(yg[fit], pg[fit]))
        oof[g] = out
        temps[g] = fit_T(yg, pg)
        rows += [{"grupo": g, "n": int(k.sum()), "version": "sin calibrar", **metrics(yg, pg)},
                 {"grupo": g, "n": int(k.sum()), "version": "T fuera de pliegue", **metrics(yg, out)}]
    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{OUT}/calibration_final.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print("\ntemperaturas (todo val):", {g: round(t, 3) for g, t in temps.items()})
    (DATA / "models" / "final_temperature.json").write_text(json.dumps(
        {"temperatura_por_mision": {m: temps[m] for m in MISSIONS}, "temperatura_global": temps["global"],
         "ajustada_con": "results/models/final_val_pred.csv (train-only, 5 semillas)", "metodo": "temperature scaling, NLL"},
        indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axes, MISSIONS):
        k = groups[m]
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        reliability(ax, y[k], p[k], "sin calibrar")
        reliability(ax, y[k], oof[m], f"T={temps[m]:.2f}")
        ax.set(title=f"{m} (val, n={k.sum()})", xlabel="confianza", ylabel="acierto")
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/calibration_final.png", dpi=120)


if __name__ == "__main__":
    main()
