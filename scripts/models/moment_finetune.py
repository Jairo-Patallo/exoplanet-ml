"""Fine-tuning de MOMENT-small para las 3 clases (paso 2 de la tarea de Hugging Face, 2026-11-03).

MOMENT (AutonLab, MIT; preentrenado sin Kepler/K2/TESS) se entrena entero: el mismo codificador convierte la vista local
y la global (rellenadas por la izquierda hasta 512 puntos, con máscara) en dos vectores, que se unen con log10 de la
profundidad (estandarizado con train) y pasan por una cabeza densa de 256. Tasa de aprendizaje baja para MOMENT (no
borrar lo preentrenado) y 1e-3 para la cabeza nueva. Una sola red con las tres misiones (como la CNN conjunta): con
35 M de parámetros, K2 sola (~600 estrellas) sobreajustaría. Mismo protocolo que la CNN (exoplanet.training): pesos de
clase, parada temprana con un 15 % de train, aumento por reflexión, selección en val por el macro-F1 medio de las
misiones, 3 semillas. bf16 en GPU (en esta GPU fp32 va ~100 veces más lento).

Uso (WSL, GPU):  export HSA_ENABLE_DXG_DETECTION=1
                 PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/moment_finetune.py
Salida: results/models/moment_ft_val.csv, _grid.csv, _seeds.csv, _val_confusion.csv; pesos en data/models/ (no versionado).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from momentfm import MOMENTPipeline
from torch import nn

from exoplanet.data import CLASSES, DATA, MISSIONS
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.training import DEVICE, fit, inputs, predict
from exoplanet.views import load_sample

OUT = Path("results/models")
WEIGHTS = DATA / "models"
SIZE, DIM, SEQ = "small", 512, 512
HEAD_LR, WEIGHT_DECAY, PATIENCE = 1e-3, 0.01, 5
GRID = [{"lr_moment": lr, "dropout": 0.3} for lr in (1e-5, 3e-5, 1e-4)]
SEEDS = (0, 1, 2)


class MomentClassifier(nn.Module):
    def __init__(self, dropout: float, depth_mean: torch.Tensor, depth_std: torch.Tensor):
        super().__init__()
        self.moment = MOMENTPipeline.from_pretrained(f"AutonLab/MOMENT-1-{SIZE}", model_kwargs={
            "task_name": "embedding", "freeze_encoder": False, "freeze_embedder": False})
        self.moment.init()
        self.register_buffer("depth_mean", depth_mean)
        self.register_buffer("depth_std", depth_std)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * DIM + 1, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, len(CLASSES)))

    def embed(self, v: torch.Tensor) -> torch.Tensor:
        L = v.shape[-1]
        mask = torch.zeros(len(v), SEQ, device=v.device)
        mask[:, -L:] = 1
        return self.moment(x_enc=F.pad(v, (SEQ - L, 0)), input_mask=mask).embeddings

    def forward(self, local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
        d = (log_depth - self.depth_mean) / self.depth_std
        return self.head(torch.cat([self.embed(local), self.embed(glob), d.to(local.dtype)], dim=1))

    def param_groups(self, lr: float) -> list[dict]:
        return [{"params": self.moment.parameters(), "lr": lr}, {"params": self.head.parameters(), "lr": HEAD_LR}]


def train(x: dict, y: torch.Tensor, cfg: dict, seed: int):
    def make(xf):
        return MomentClassifier(cfg["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6)
    return fit(make, x, y, WEIGHT_DECAY, seed, augment=True, lr=cfg["lr_moment"], amp=True, patience=PATIENCE)


def per_mission_f1(p: np.ndarray, val: pd.DataFrame) -> dict[str, float]:
    return {m: point_metrics(val.disposition[val.mission == m].to_numpy(), p[(val.mission == m).to_numpy()])["macro_f1"]
            for m in MISSIONS}


def main() -> None:
    print(f"dispositivo: {DEVICE}", flush=True)
    stars = load_sample()
    tr, va = stars[stars.split == "train"], stars[stars.split == "val"].reset_index(drop=True)
    ds_tr = ViewDataset(tr)
    xtr, ytr, xva = inputs(ds_tr), ds_tr.y, inputs(ViewDataset(va))
    grid, seeds, scores = [], [], []
    for cfg in GRID:
        t = time.perf_counter()
        model, epoch = train(xtr, ytr, cfg, SEEDS[0])
        f1 = per_mission_f1(predict(model, xva, amp=True), va)
        scores.append(np.mean(list(f1.values())))
        grid.append({**cfg, "epoca_parada": epoch, **{f"f1_{m}": v for m, v in f1.items()}, "macro_f1_val": scores[-1],
                     "segundos": round(time.perf_counter() - t)})
        print(f"  {cfg}: época {epoch}, macro-F1 medio {scores[-1]:.3f} ({grid[-1]['segundos']} s)", flush=True)
        del model
        torch.cuda.empty_cache()
    cfg = GRID[int(np.argmax(scores))]
    for seed in SEEDS:
        model, epoch = train(xtr, ytr, cfg, seed)
        p = predict(model, xva, amp=True)
        seeds += [{"mision": m, "semilla": seed, "epoca_parada": epoch, "macro_f1": f} for m, f in per_mission_f1(p, va).items()]
        if seed == SEEDS[0]:
            p_best = p
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": cfg, "size": SIZE, "head_lr": HEAD_LR,
                        "weight_decay": WEIGHT_DECAY, "classes": CLASSES, "seed": seed}, WEIGHTS / "moment_ft.pt")
        print(f"  semilla {seed}: época {epoch}, " + ", ".join(f"{s['mision']} {s['macro_f1']:.3f}" for s in seeds[-3:]), flush=True)
        del model
        torch.cuda.empty_cache()

    table, conf = evaluate_by_mission(va.mission.to_numpy(), va.disposition.to_numpy(), p_best)
    table.insert(1, "hiperparametros", ", ".join(f"{k}={v}" for k, v in cfg.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(OUT / "moment_ft_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / "moment_ft_grid.csv", index=False, encoding="utf-8")
    sd = pd.DataFrame(seeds)
    sd.round(4).to_csv(OUT / "moment_ft_seeds.csv", index=False, encoding="utf-8")
    pd.concat(conf, names=["mision", "real"]).to_csv(OUT / "moment_ft_val_confusion.csv", encoding="utf-8")
    cols = ["mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + table[cols].round(3).to_string(index=False))
    print("\nMedia de 3 semillas:", sd.groupby("mision").macro_f1.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
