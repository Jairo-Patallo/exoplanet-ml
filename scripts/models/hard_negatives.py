"""Negativos difíciles (fase 4, research/11-analisis-errores.md, decisión de Jairo 2026-11-05).

La receta final (base 16, dropout 0,5, sin profundidad) se entrena con train más una segunda vista de cada NO_SIGNAL
de train: plegada en el pico de BLS (data/sample/views_bls, build_bls_views.py --splits train --nosignal), es decir,
ruido con forma de tránsito, como las falsas detecciones de una tubería real. Ambas copias de una estrella van al
mismo lado de la parada temprana (groups). Pérdida con pesos por clase (training.fit): NO_SIGNAL no gana peso total.
Se evalúa en val igual que la línea base (final_model.py val): vistas del catálogo y vistas BLS, 5 semillas.

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/hard_negatives.py
Salida: results/models/hard_val_pred.csv y hard_val_bls_pred.csv (mismo formato que final_val*_pred.csv).
"""
import numpy as np
import pandas as pd
import torch

from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.training import fit, inputs, predict
from exoplanet.views import load_sample
from final_model import OUT, SEEDS, WEIGHT_DECAY, chosen, make, x_nodepth

BLS_VIEWS = DATA / "sample" / "views_bls"


def bls_stars(split: str, nosignal_only: bool = False) -> pd.DataFrame:
    q = pd.read_csv("results/sample/qc_bls.csv")
    q = q[q.error.isna() & q["valid"].eq(True) & (q.split == split)]
    return (q[q.disposition == "NO_SIGNAL"] if nosignal_only else q).reset_index(drop=True)


def x_bls(stars: pd.DataFrame) -> tuple[dict, torch.Tensor]:
    d = ViewDataset(stars, views_dir=BLS_VIEWS)
    return {**inputs(d), "log_depth": torch.zeros_like(d.log_depth)}, d.y


def main() -> None:
    cfg = chosen()
    tr, hard = load_sample(("train",)), bls_stars("train", nosignal_only=True)
    x_tr, y_tr = x_nodepth(tr)
    x_h, y_h = x_bls(hard)
    assert (y_h == CLASSES.index("NO_SIGNAL")).all()
    x = {k: torch.cat([x_tr[k], x_h[k]]) for k in x_tr}
    y = torch.cat([y_tr, y_h])
    ids = pd.concat([tr.star_id, hard.star_id], ignore_index=True)
    groups = pd.factorize(ids)[0]
    print(f"train {len(tr)} + negativos difíciles {len(hard)} ({hard.mission.value_counts().to_dict()})", flush=True)

    va, vb = load_sample(("val",)), bls_stars("val")
    xv, _ = x_nodepth(va)
    xb, _ = x_bls(vb)
    pv, pb = [], []
    for s in SEEDS:
        model, epoch = fit(make(cfg), x, y, WEIGHT_DECAY, s, augment=True, amp=True, groups=groups)
        pv.append(predict(model, xv, amp=True))
        pb.append(predict(model, xb, amp=True))
        print(f"semilla {s}: época {epoch}", flush=True)
    for stars, p, name in ((va, pv, "hard_val_pred.csv"), (vb, pb, "hard_val_bls_pred.csv")):
        p = np.mean(p, axis=0)
        stars[["star_id", "mission", "disposition"]].assign(**{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}).round(4).to_csv(
            OUT / name, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
