"""CNN 1D de dos ramas al estilo AstroNet (Shallue & Vanderburg 2018) para 3 clases (fase 2, tarea #16).

Rama global: 5 bloques convolucionales (2 conv de núcleo 5 + BatchNorm + ReLU, y max-pool 2) con base, 2·base ... 16·base
filtros. Rama local: 2 bloques (base, 2·base). Se aplanan, se unen con log10 de la profundidad (estandarizado con train)
y pasan por 2 capas densas de 256 con dropout. Misma entrada y protocolo que el resto (exoplanet.training): pesos de clase,
parada temprana con un 15 % de train, selección en val por macro-F1, 3 semillas. Aumento por reflexión en el tiempo.

Dos formas de entrenar (decisión de Jairo, 2026-11-02):
  por_mision: una CNN por misión, comparable con los modelos anteriores.
  conjunta:   una sola CNN con el train de las tres misiones (~3 veces más datos), evaluada en el val de cada misión;
              la configuración se elige por el macro-F1 medio de las misiones. Sigue siendo dentro de dominio (no LOMO).

Uso (WSL):  PYTHONPATH=src ~/venvs/exoplanet/bin/python scripts/models/cnn.py
Salida: results/models/cnn_val.csv, cnn_grid.csv, cnn_seeds.csv, cnn_val_confusion.csv; pesos en data/models/cnn_*.pt.
"""
import time
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
GRID = [{"base": b, "dropout": d} for b in (8, 16) for d in (0.3, 0.5)]
WEIGHT_DECAY, FC = 1e-4, 256
SEEDS = (0, 1, 2)


def conv_column(n_blocks: int, base: int) -> nn.Sequential:
    layers, c_in = [], 1
    for i in range(n_blocks):
        c_out = base * 2 ** i
        layers += [nn.Conv1d(c_in, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(),
                   nn.Conv1d(c_out, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)]
        c_in = c_out
    return nn.Sequential(*layers, nn.Flatten())


class AstroCNN(nn.Module):
    def __init__(self, base: int, dropout: float, depth_mean: torch.Tensor, depth_std: torch.Tensor):
        super().__init__()
        self.register_buffer("depth_mean", depth_mean)
        self.register_buffer("depth_std", depth_std)
        self.glob = conv_column(5, base)     # 201 -> 6 posiciones x 16·base canales
        self.local = conv_column(2, base)    # 81 -> 20 posiciones x 2·base canales
        n_flat = 6 * 16 * base + 20 * 2 * base + 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_flat, FC), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(FC, FC), nn.ReLU(), nn.Dropout(dropout), nn.Linear(FC, len(CLASSES)))

    def forward(self, local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
        d = (log_depth - self.depth_mean) / self.depth_std
        return self.head(torch.cat([self.glob(glob), self.local(local), d], dim=1))


def train(x: dict, y: torch.Tensor, cfg: dict, seed: int):
    def make(xf):
        return AstroCNN(cfg["base"], cfg["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6)
    return fit(make, x, y, WEIGHT_DECAY, seed, augment=True)


def per_mission_f1(p: np.ndarray, val: pd.DataFrame) -> dict[str, float]:
    return {m: point_metrics(val.disposition[val.mission == m].to_numpy(), p[(val.mission == m).to_numpy()])["macro_f1"]
            for m in MISSIONS if (val.mission == m).any()}


def run(mode: str, groups: list[tuple[str, pd.DataFrame, pd.DataFrame]], grid: list, seeds: list) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """groups: (nombre, estrellas de train, estrellas de val). Devuelve probabilidades en val (semilla 0) y su tabla."""
    p_all, val_all, chosen = [], [], {}
    for name, tr, va in groups:
        ds_tr = ViewDataset(tr)
        xtr, ytr, xva = inputs(ds_tr), ds_tr.y, inputs(ViewDataset(va))
        scores = []
        for cfg in GRID:
            t = time.perf_counter()
            model, epoch = train(xtr, ytr, cfg, SEEDS[0])
            f1 = per_mission_f1(predict(model, xva), va)
            scores.append(np.mean(list(f1.values()) if mode == "conjunta" else [f1[name]]))
            grid.append({"entrenamiento": mode, "modelo": name, **cfg, "epoca_parada": epoch, "macro_f1_val": scores[-1],
                         "segundos": round(time.perf_counter() - t)})
            print(f"  {mode} {name} {cfg}: época {epoch}, macro-F1 {scores[-1]:.3f} ({grid[-1]['segundos']} s)", flush=True)
        cfg = GRID[int(np.argmax(scores))]
        chosen[name] = ", ".join(f"{k}={v}" for k, v in cfg.items())
        for seed in SEEDS:
            model, epoch = train(xtr, ytr, cfg, seed)
            p = predict(model, xva)
            seeds += [{"entrenamiento": mode, "mision": m, "semilla": seed, "epoca_parada": epoch, "macro_f1": f}
                      for m, f in per_mission_f1(p, va).items() if mode == "conjunta" or m == name]
            if seed == SEEDS[0]:
                p_all.append(p)
                val_all.append(va)
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": cfg, "fc": FC, "weight_decay": WEIGHT_DECAY,
                            "classes": CLASSES, "seed": seed, "lr": LR, "batch": BATCH, "augment": "reflexion"},
                           WEIGHTS / f"cnn_{mode}_{name}.pt")
        print(f"cnn {mode} {name}: {chosen[name]}", flush=True)
    return np.vstack(p_all), pd.concat(val_all, ignore_index=True), chosen


def main() -> None:
    print(f"dispositivo: {DEVICE}, hilos: {torch.get_num_threads()}", flush=True)
    stars = load_sample()
    train_s, val_s = stars[stars.split == "train"], stars[stars.split == "val"]
    grid, seeds, tables, confs = [], [], [], {}
    plans = {"por_mision": [(m, train_s[train_s.mission == m], val_s[val_s.mission == m].reset_index(drop=True)) for m in MISSIONS],
             "conjunta": [("todas", train_s, val_s.reset_index(drop=True))]}
    for mode, groups in plans.items():
        p, val, chosen = run(mode, groups, grid, seeds)
        table, conf = evaluate_by_mission(val.mission.to_numpy(), val.disposition.to_numpy(), p)
        cfg_of = (lambda m: chosen.get(m, "-")) if mode == "por_mision" else (lambda m: chosen["todas"])
        table.insert(0, "entrenamiento", mode)
        table.insert(2, "hiperparametros", table.mision.map(cfg_of))
        tables.append(table)
        confs.update({(mode, m): c for m, c in conf.items()})

    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat(tables, ignore_index=True)
    out.round(3).to_csv(OUT / "cnn_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / "cnn_grid.csv", index=False, encoding="utf-8")
    sd = pd.DataFrame(seeds)
    sd.round(4).to_csv(OUT / "cnn_seeds.csv", index=False, encoding="utf-8")
    pd.concat(confs, names=["entrenamiento", "mision", "real"]).to_csv(OUT / "cnn_val_confusion.csv", encoding="utf-8")
    cols = ["entrenamiento", "mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + out[cols].round(3).to_string(index=False))
    print("\nVariabilidad entre semillas (macro-F1 val):")
    print(sd.groupby(["entrenamiento", "mision"]).macro_f1.agg(["mean", "std", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
