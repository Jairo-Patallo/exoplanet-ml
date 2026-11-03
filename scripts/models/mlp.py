"""MLP de 3 clases dentro de cada misión sobre las vistas (fase 2, tarea #29), con las métricas de #51.

Mismo protocolo que scripts/models/sklearn_models.py para que la comparación sea justa: misma entrada (vistas local y
global normalizadas + log10 de la profundidad, exoplanet.dataset), estandarizada con la media y desviación de train;
un modelo por misión; pesos de clase "balanced"; hiperparámetros elegidos en val por macro-F1; el test no se toca.
Red: 2 capas ocultas (ReLU + dropout). Entrenamiento común con la CNN (exoplanet.training): AdamW, entropía cruzada con
pesos de clase y parada temprana con un 15 % de train apartado, así val no decide cuándo parar. La mejor configuración se repite con 3 semillas para medir
la variabilidad entre entrenamientos.

Uso (WSL, PyTorch no carga en Windows):  PYTHONPATH=src ~/venvs/exoplanet/bin/python scripts/models/mlp.py
Salida: results/models/mlp_val.csv, mlp_grid.csv, mlp_seeds.csv, mlp_val_confusion.csv;
        pesos y configuración del mejor modelo por misión en data/models/mlp_<misión>.pt (ignorado por git).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from exoplanet.data import CLASSES, DATA, MISSIONS
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.training import BATCH, DEVICE, LR, fit, inputs, predict
from exoplanet.views import load_sample

OUT = Path("results/models")
WEIGHTS = DATA / "models"
GRID = [{"hidden": h, "dropout": d, "weight_decay": wd} for h in (64, 256) for d in (0.2, 0.5) for wd in (1e-4, 1e-2)]
SEEDS = (0, 1, 2)


def flat(local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
    """(n, 283) = local + global + log_depth."""
    return torch.cat([local.squeeze(1), glob.squeeze(1), log_depth], dim=1)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int, dropout: float, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)    # estandarización con estadísticas de train, guardada con el modelo
        self.register_buffer("std", std)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, len(CLASSES)))

    def forward(self, local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
        return self.net((flat(local, glob, log_depth) - self.mean) / self.std)


def train(x: dict, y: torch.Tensor, cfg: dict, seed: int):
    def make(xf):
        X = flat(**xf)
        return MLP(X.shape[1], cfg["hidden"], cfg["dropout"], X.mean(0), X.std(0) + 1e-6)
    return fit(make, x, y, cfg["weight_decay"], seed)


def main() -> None:
    print(f"dispositivo: {DEVICE}", flush=True)
    stars = load_sample()
    grid, seeds, p_val, chosen = [], [], [], {}
    val_parts = []
    for m in MISSIONS:
        ds_tr = ViewDataset(stars[(stars.mission == m) & (stars.split == "train")])
        Xtr, ytr = inputs(ds_tr), ds_tr.y
        va = stars[(stars.mission == m) & (stars.split == "val")].reset_index(drop=True)
        Xva = inputs(ViewDataset(va))
        yva = va.disposition.to_numpy()
        scores = []
        for cfg in GRID:
            model, epoch = train(Xtr, ytr, cfg, SEEDS[0])
            s = point_metrics(yva, predict(model, Xva))["macro_f1"]
            scores.append(s)
            grid.append({"mision": m, **cfg, "epoca_parada": epoch, "macro_f1_val": s})
        cfg = GRID[int(np.argmax(scores))]
        chosen[m] = ", ".join(f"{k}={v}" for k, v in cfg.items())
        for seed in SEEDS:
            model, epoch = train(Xtr, ytr, cfg, seed)
            p = predict(model, Xva)
            seeds.append({"mision": m, "semilla": seed, "epoca_parada": epoch, **point_metrics(yva, p)})
            if seed == SEEDS[0]:
                p_val.append(p)
                val_parts.append(va)
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": cfg, "n_in": 283, "classes": CLASSES,
                            "seed": seed, "lr": LR, "batch": BATCH}, WEIGHTS / f"mlp_{m}.pt")
        print(f"mlp {m}: {chosen[m]} macro-F1 val={max(scores):.3f}", flush=True)

    val = pd.concat(val_parts, ignore_index=True)
    table, conf = evaluate_by_mission(val.mission.to_numpy(), val.disposition.to_numpy(), np.vstack(p_val))
    table.insert(1, "hiperparametros", table.mision.map(chosen))
    OUT.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(OUT / "mlp_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / "mlp_grid.csv", index=False, encoding="utf-8")
    sd = pd.DataFrame(seeds)
    sd.round(4).to_csv(OUT / "mlp_seeds.csv", index=False, encoding="utf-8")
    pd.concat(conf, names=["mision", "real"]).to_csv(OUT / "mlp_val_confusion.csv", encoding="utf-8")
    cols = ["mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\nVariabilidad entre semillas (macro-F1 val):")
    print(sd.groupby("mision").macro_f1.agg(["mean", "std", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
