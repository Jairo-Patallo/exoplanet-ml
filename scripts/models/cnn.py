"""CNN 1D de dos ramas al estilo AstroNet (Shallue & Vanderburg 2018) para 3 clases (fase 2, tarea #16).

Rama global: 5 bloques convolucionales (2 conv de núcleo 5 + BatchNorm + ReLU, y max-pool 2) con base, 2·base ... 16·base
filtros. Rama local: 2 bloques (base, 2·base). Se aplanan, se unen con log10 de la profundidad (estandarizado con train)
y pasan por 2 capas densas de 256 con dropout. Misma entrada y protocolo que el resto (exoplanet.training): pesos de clase,
parada temprana con un 15 % de train, selección en val por macro-F1, 3 semillas. Aumento por reflexión en el tiempo.

Ensemble: además de la semilla 0 se informa la media de las probabilidades de las 3 semillas ("<modo>_ensemble").
Dos formas de entrenar (decisión de Jairo, 2026-11-02):
  por_mision: una CNN por misión, comparable con los modelos anteriores.
  conjunta:   una sola CNN con el train de las tres misiones (~3 veces más datos), evaluada en el val de cada misión;
              la configuración se elige por el macro-F1 medio de las misiones. Sigue siendo dentro de dominio (no LOMO).

Uso (WSL):  PYTHONPATH=src ~/venvs/exoplanet/bin/python scripts/models/cnn.py [--centroid] [--odd-even] [--features] [--ephem]
                 [--modes por_mision conjunta] [--amp]
            (--amp en GPU: ~/venvs/exoplanet-rocm y HSA_ENABLE_DXG_DETECTION=1)
Salida: results/models/cnn_val.csv, cnn_grid.csv, cnn_seeds.csv, cnn_val_confusion.csv; pesos en data/models/cnn_*.pt.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN, feat_stats
from exoplanet.data import CLASSES, DATA, MISSIONS
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.training import BATCH, DEVICE, LR, fit, inputs, predict
from exoplanet.views import load_sample

OUT = Path("results/models")
WEIGHTS = DATA / "models"
GRID = [{"base": b, "dropout": d} for b in (8, 16) for d in (0.3, 0.5)]
WEIGHT_DECAY = 1e-4
SEEDS = (0, 1, 2)
CENTROID, ODD_EVEN, FEATURES, EPHEM, AMP, NAME = False, False, False, False, False, "cnn"   # se fijan en main() con las opciones


def train(x: dict, y: torch.Tensor, cfg: dict, seed: int, amp: bool = False):
    def make(xf):
        return AstroCNN(cfg["base"], cfg["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6,
                        channels=xf["local"].shape[1], glob_channels=xf["glob"].shape[1],
                        feat_stats=feat_stats(xf["feats"]) if "feats" in xf else None)
    return fit(make, x, y, WEIGHT_DECAY, seed, augment=True, amp=amp)


def per_mission_f1(p: np.ndarray, val: pd.DataFrame) -> dict[str, float]:
    return {m: point_metrics(val.disposition[val.mission == m].to_numpy(), p[(val.mission == m).to_numpy()])["macro_f1"]
            for m in MISSIONS if (val.mission == m).any()}


def run(mode: str, groups: list[tuple[str, pd.DataFrame, pd.DataFrame]], grid: list, seeds: list) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """groups: (nombre, estrellas de train, estrellas de val). Devuelve probabilidades en val (semilla 0) y su tabla."""
    p_all, val_all, chosen, ensembles = [], [], {}, []
    for name, tr, va in groups:
        opts = dict(centroid=CENTROID, odd_even=ODD_EVEN, features=FEATURES, ephem=EPHEM)
        ds_tr = ViewDataset(tr, **opts)
        xtr, ytr, xva = inputs(ds_tr), ds_tr.y, inputs(ViewDataset(va, **opts))
        p_seeds = []
        scores = []
        for cfg in GRID:
            t = time.perf_counter()
            model, epoch = train(xtr, ytr, cfg, SEEDS[0], AMP)
            f1 = per_mission_f1(predict(model, xva, amp=AMP), va)
            scores.append(np.mean(list(f1.values()) if mode == "conjunta" else [f1[name]]))
            grid.append({"entrenamiento": mode, "modelo": name, **cfg, "epoca_parada": epoch, "macro_f1_val": scores[-1],
                         "segundos": round(time.perf_counter() - t)})
            print(f"  {mode} {name} {cfg}: época {epoch}, macro-F1 {scores[-1]:.3f} ({grid[-1]['segundos']} s)", flush=True)
        cfg = GRID[int(np.argmax(scores))]
        chosen[name] = ", ".join(f"{k}={v}" for k, v in cfg.items())
        for seed in SEEDS:
            model, epoch = train(xtr, ytr, cfg, seed, AMP)
            p = predict(model, xva, amp=AMP)
            p_seeds.append(p)
            seeds += [{"entrenamiento": mode, "mision": m, "semilla": seed, "epoca_parada": epoch, "macro_f1": f}
                      for m, f in per_mission_f1(p, va).items() if mode == "conjunta" or m == name]
            if seed == SEEDS[0]:
                p_all.append(p)
                val_all.append(va)
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": cfg, "weight_decay": WEIGHT_DECAY,
                            "classes": CLASSES, "seed": seed, "lr": LR, "batch": BATCH, "augment": "reflexion"},
                           WEIGHTS / f"{NAME}_{mode}_{name}.pt")
        ensembles.append(np.mean(p_seeds, axis=0))     # media de las probabilidades de las semillas
        print(f"cnn {mode} {name}: {chosen[name]}", flush=True)
    return np.vstack(p_all), pd.concat(val_all, ignore_index=True), chosen, np.vstack(ensembles)


def main() -> None:
    global CENTROID, ODD_EVEN, FEATURES, EPHEM, AMP, NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroid", action="store_true", help="canal con el centroide (variante curva + centroide)")
    ap.add_argument("--odd-even", action="store_true", help="canales par e impar en la rama global")
    ap.add_argument("--features", action="store_true", help="diagnósticos físicos en la cabeza (CNN híbrida)")
    ap.add_argument("--ephem", action="store_true", help="log(duración/periodo) y log(periodo) en la cabeza")
    ap.add_argument("--modes", nargs="+", default=["por_mision", "conjunta"], choices=["por_mision", "conjunta"])
    ap.add_argument("--amp", action="store_true", help="bf16 en GPU")
    args = ap.parse_args()
    CENTROID, ODD_EVEN, FEATURES, EPHEM, AMP = args.centroid, args.odd_even, args.features, args.ephem, args.amp
    NAME = ("cnn" + ("_centroid" if CENTROID else "") + ("_oddeven" if ODD_EVEN else "") + ("_feats" if FEATURES else "") + ("_ephem" if EPHEM else "")
            + ("_gpu" if AMP else ""))
    print(f"dispositivo: {DEVICE}, hilos: {torch.get_num_threads()}", flush=True)
    stars = load_sample()
    train_s, val_s = stars[stars.split == "train"], stars[stars.split == "val"]
    grid, seeds, tables, confs = [], [], [], {}
    plans = {"por_mision": [(m, train_s[train_s.mission == m], val_s[val_s.mission == m].reset_index(drop=True)) for m in MISSIONS],
             "conjunta": [("todas", train_s, val_s.reset_index(drop=True))]}
    for mode in args.modes:
        p, val, chosen, p_ens = run(mode, plans[mode], grid, seeds)
        cfg_of = (lambda m: chosen.get(m, "-")) if mode == "por_mision" else (lambda m: chosen["todas"])
        for label, probs in ((mode, p), (f"{mode}_ensemble", p_ens)):
            table, conf = evaluate_by_mission(val.mission.to_numpy(), val.disposition.to_numpy(), probs)
            table.insert(0, "entrenamiento", label)
            table.insert(2, "hiperparametros", table.mision.map(cfg_of))
            tables.append(table)
            confs.update({(label, m): c for m, c in conf.items()})

    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat(tables, ignore_index=True)
    out.round(3).to_csv(OUT / f"{NAME}_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / f"{NAME}_grid.csv", index=False, encoding="utf-8")
    sd = pd.DataFrame(seeds)
    sd.round(4).to_csv(OUT / f"{NAME}_seeds.csv", index=False, encoding="utf-8")
    pd.concat(confs, names=["entrenamiento", "mision", "real"]).to_csv(OUT / f"{NAME}_val_confusion.csv", encoding="utf-8")
    cols = ["entrenamiento", "mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + out[cols].round(3).to_string(index=False))
    print("\nVariabilidad entre semillas (macro-F1 val):")
    print(sd.groupby(["entrenamiento", "mision"]).macro_f1.agg(["mean", "std", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
