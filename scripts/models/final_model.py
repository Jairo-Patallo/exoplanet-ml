"""Modelo final (fase 4): CNN solo curva, sin profundidad, Kepler + K2 + TESS juntos. Receta: research/10-modelo-final.md.

Etapas (en este orden; el test solo se mira en la última, una vez):
  select  rejilla de 4 configuraciones entrenadas con train (semilla 0), elegida por el macro-F1 medio de misiones en
          val, como la CNN conjunta de la fase 2 -> results/models/final_select.csv
  train   la configuración elegida, entrenada con train + val, 5 semillas -> data/models/final_seed*.pt
  test    ensemble de las 5 semillas en el test de cada misión, IC 95 % bootstrap -> results/models/final_test*.csv
  val     análisis de errores: receta entrenada solo con train, predicha en val (catálogo y BLS) -> final_val*_pred.csv
Sin profundidad: log10 de la profundidad constante (0), como lomo.py --no-depth (decisión #43).

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/final_model.py {select,train,test}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES, DATA, MISSIONS
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.training import DEVICE, fit, inputs, predict
from exoplanet.views import load_sample

OUT, MODELS = Path("results/models"), Path("data/models")
GRID = [{"base": b, "dropout": d} for b in (8, 16) for d in (0.3, 0.5)]
SEEDS, WEIGHT_DECAY = (0, 1, 2, 3, 4), 1e-4


def x_nodepth(stars: pd.DataFrame) -> tuple[dict, torch.Tensor]:
    d = ViewDataset(stars)
    return {**inputs(d), "log_depth": torch.zeros_like(d.log_depth)}, d.y


def make(cfg):
    return lambda xf: AstroCNN(cfg["base"], cfg["dropout"], xf["log_depth"].mean(0), xf["log_depth"].std(0) + 1e-6)


def select() -> None:
    tr, va = load_sample(("train",)), load_sample(("val",))
    x_tr, y_tr = x_nodepth(tr)
    x_va, _ = x_nodepth(va)
    rows = []
    for cfg in GRID:
        model, epoch = fit(make(cfg), x_tr, y_tr, WEIGHT_DECAY, SEEDS[0], augment=True, amp=True)
        p = predict(model, x_va, amp=True)
        f1 = {m: point_metrics(va.disposition[va.mission == m].to_numpy(), p[(va.mission == m).to_numpy()])["macro_f1"]
              for m in MISSIONS}
        rows.append({**cfg, "epoca_parada": epoch, **{f"macro_f1_{m}": v for m, v in f1.items()}, "macro_f1_media": np.mean(list(f1.values()))})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "final_select.csv", index=False, encoding="utf-8")
    print("elegida:", out.loc[out.macro_f1_media.idxmax(), ["base", "dropout"]].to_dict())


def chosen() -> dict:
    s = pd.read_csv(OUT / "final_select.csv")
    r = s.loc[s.macro_f1_media.idxmax()]
    return {"base": int(r.base), "dropout": float(r.dropout)}


def train() -> None:
    cfg = chosen()
    x, y = x_nodepth(load_sample(("train", "val")))
    MODELS.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        model, epoch = fit(make(cfg), x, y, WEIGHT_DECAY, s, augment=True, amp=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": s, "depth": False, "classes": CLASSES},
                   MODELS / f"final_seed{s}.pt")
        print(f"semilla {s}: época {epoch}", flush=True)


def test() -> None:
    te = load_sample(("test",))
    x, _ = x_nodepth(te)
    probas = []
    for s in SEEDS:
        ck = torch.load(MODELS / f"final_seed{s}.pt", map_location="cpu")
        m = AstroCNN(ck["config"]["base"], ck["config"]["dropout"], torch.zeros(1), torch.ones(1))
        m.load_state_dict(ck["state_dict"])
        probas.append(predict(m.to(DEVICE).eval(), x, amp=True))
    p = np.mean(probas, axis=0)
    table, conf = evaluate_by_mission(te.mission.to_numpy(), te.disposition.to_numpy(), p)
    print(table[["mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro", "pr_auc_macro_lo", "pr_auc_macro_hi"]].round(3).to_string(index=False))
    table.round(4).to_csv(OUT / "final_test.csv", index=False, encoding="utf-8")
    pd.concat({m: c for m, c in conf.items()}).to_csv(OUT / "final_test_confusion.csv", encoding="utf-8")
    te[["star_id", "mission", "disposition"]].assign(**{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}).round(4).to_csv(
        OUT / "final_test_pred.csv", index=False, encoding="utf-8")


def val() -> None:
    """Análisis de errores sin tocar test: la receta entrenada solo con train (5 semillas), predicha en val con las
    vistas del catálogo y, para las estrellas con vista BLS válida, con la efeméride de BLS (negativos difíciles)."""
    cfg, va = chosen(), load_sample(("val",))
    x, y = x_nodepth(load_sample(("train",)))
    q = pd.read_csv("results/sample/qc_bls.csv")
    vb = q[q.error.isna() & q["valid"].eq(True) & (q.split == "val")].reset_index(drop=True)
    db = ViewDataset(vb, views_dir=DATA / "sample" / "views_bls")
    xb = {**inputs(db), "log_depth": torch.zeros_like(db.log_depth)}
    xv, _ = x_nodepth(va)
    pv, pb = [], []
    for s in SEEDS:
        model = fit(make(cfg), x, y, WEIGHT_DECAY, s, augment=True, amp=True)[0]
        pv.append(predict(model, xv, amp=True))
        pb.append(predict(model, xb, amp=True))
        print(f"semilla {s}", flush=True)
    for stars, p, name in ((va, pv, "final_val_pred.csv"), (vb, pb, "final_val_bls_pred.csv")):
        p = np.mean(p, axis=0)
        stars[["star_id", "mission", "disposition"]].assign(**{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}).round(4).to_csv(
            OUT / name, index=False, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "train", "test", "val"])
    {"select": select, "train": train, "test": test, "val": val}[ap.parse_args().stage]()
