"""Coste de no usar el catálogo: el mismo modelo con vistas plegadas por el catálogo o por BLS (fase 4, #53, 2026-11-05).

Modelo: CNN conjunta de la fase 2 (data/models/cnn_gpu_conjunta_todas.pt, semilla 0, entrenada con train de las tres
misiones y vistas de catálogo). Estrellas: val con vista válida en las dos versiones (qc.csv y qc_bls.csv).
Métricas por misión: PR-AUC macro y macro-F1, y acierto por categoría de recuperación del periodo de BLS.

Uso (WSL):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/eval_bls.py
Salida: results/models/eval_bls.csv, eval_bls_por_categoria.csv y eval_bls_pred.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import point_metrics
from exoplanet.training import DEVICE, inputs, predict

OUT = Path("results/models")


def load_model() -> AstroCNN:
    ck = torch.load("data/models/cnn_gpu_conjunta_todas.pt", map_location="cpu")
    m = AstroCNN(ck["config"]["base"], ck["config"]["dropout"], torch.zeros(1), torch.ones(1))
    m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE).eval()


def main() -> None:
    q = pd.read_csv("results/sample/qc_bls.csv")
    print(f"BLS: {len(q)} estrellas, {q.error.notna().sum()} errores, {(q['valid'] != True).sum()} vistas no válidas", flush=True)
    s = q[q.error.isna() & q["valid"].eq(True)].reset_index(drop=True)
    model = load_model()
    p_cat = predict(model, inputs(ViewDataset(s)), amp=True)
    p_bls = predict(model, inputs(ViewDataset(s, views_dir=DATA / "sample" / "views_bls")), amp=True)
    y = s.disposition.to_numpy()

    rows = []
    for mission in ["kepler", "k2", "tess"]:
        m = (s.mission == mission).to_numpy()
        for name, p in (("catálogo", p_cat), ("BLS", p_bls)):
            r = point_metrics(y[m], p[m])
            rows.append({"mision": mission, "efemeride": name, "n": int(m.sum()), "pr_auc_macro": r["pr_auc_macro"],
                         "macro_f1": r["macro_f1"], **{f"f1_{c}": r[f"f1_{c}"] for c in CLASSES}})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))

    pred = s[["star_id", "mission", "disposition", "match", "bls_sde"]].assign(
        acierto_catalogo=np.asarray(CLASSES)[p_cat.argmax(1)] == y, acierto_bls=np.asarray(CLASSES)[p_bls.argmax(1)] == y)
    cat = (pred.groupby(["mission", "disposition", "match"])
               .agg(n=("star_id", "size"), acierto_catalogo=("acierto_catalogo", "mean"), acierto_bls=("acierto_bls", "mean"))
               .reset_index())
    print(cat.round(3).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT / "eval_bls.csv", index=False, encoding="utf-8")
    cat.round(4).to_csv(OUT / "eval_bls_por_categoria.csv", index=False, encoding="utf-8")
    pred.assign(**{f"cat_p_{c}": p_cat[:, i] for i, c in enumerate(CLASSES)},
                **{f"bls_p_{c}": p_bls[:, i] for i, c in enumerate(CLASSES)}).round(4).to_csv(
        OUT / "eval_bls_pred.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
