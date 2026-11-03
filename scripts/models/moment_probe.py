"""MOMENT congelado + regresión logística (paso 1 de la tarea de Hugging Face, 2026-11-03).

MOMENT (AutonLab, MIT) es un modelo general de series temporales; su preentrenamiento (Time-series Pile: Informer,
Monash, UCR/UEA, TSB-UAD) no incluye Kepler, K2 ni TESS, así que no ha visto nuestras etiquetas. El modelo NO se
entrena: convierte cada vista en un vector (embedding). Entrada: las mismas vistas normalizadas que el resto de
modelos (exoplanet.views.normalize), rellenadas por la izquierda hasta 512 puntos con máscara (MOMENT ignora el
relleno). Se unen el embedding de la local, el de la global y log10 de la profundidad, y encima va una regresión
logística por misión con el mismo protocolo que sklearn_models.py (StandardScaler en train, balanced, C en val).

Uso (WSL, entorno ~/venvs/moment con momentfm):
    PYTHONPATH=src ~/venvs/moment/bin/python scripts/models/moment_probe.py --size small
Salida: results/models/moment_<size>_val.csv, _grid.csv, _val_confusion.csv; embeddings en data/models/ (no versionado).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from momentfm import MOMENTPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.data import CLASSES, DATA, MISSIONS
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.views import load_sample, load_views, normalize

OUT = Path("results/models")
EMB = DATA / "models"
SEQ, BATCH, SEED = 512, 64, 0
C_GRID = np.logspace(-4, 2, 13)


@torch.no_grad()
def embed(model, views: np.ndarray) -> np.ndarray:
    """(n, L) -> (n, d). Relleno con ceros por la izquierda hasta 512 y máscara a 0 en el relleno."""
    n, L = views.shape
    x = np.zeros((n, SEQ), np.float32)
    x[:, -L:] = views
    out = []
    for i in range(0, n, BATCH):
        xb = torch.from_numpy(x[i:i + BATCH]).unsqueeze(1)
        m = torch.zeros(len(xb), SEQ)
        m[:, -L:] = 1
        out.append(model(x_enc=xb, input_mask=m).embeddings.numpy())
    return np.vstack(out)


def embeddings(stars: pd.DataFrame, size: str) -> np.ndarray:
    path = EMB / f"moment_{size}_emb.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=True)
        if list(cached["star_id"]) == list(stars.star_id):
            return cached["X"]
    model = MOMENTPipeline.from_pretrained(f"AutonLab/MOMENT-1-{size}", model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    v = load_views(stars.star_id)
    local, glob, log_depth = normalize(v["local"], v["glob"])
    t = time.perf_counter()
    X = np.hstack([embed(model, local), embed(model, glob), log_depth])
    print(f"embeddings {X.shape} en {time.perf_counter() - t:.0f} s", flush=True)
    EMB.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, star_id=stars.star_id.to_numpy())
    return X


def fit(X: np.ndarray, y: np.ndarray, C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                                             random_state=SEED)).fit(X, y)


def proba(model, X: np.ndarray) -> np.ndarray:
    order = [list(model.classes_).index(c) for c in CLASSES]
    return model.predict_proba(X)[:, order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small", choices=["small", "base", "large"])
    args = ap.parse_args()
    torch.manual_seed(SEED)
    stars = load_sample()
    X = embeddings(stars, args.size)
    y, val = stars.disposition.to_numpy(), (stars.split == "val").to_numpy()
    p_val, grid, chosen = np.zeros((val.sum(), len(CLASSES))), [], {}
    for m in MISSIONS:
        tr, va = (stars.mission == m).to_numpy() & ~val, (stars.mission == m).to_numpy() & val
        scores = [point_metrics(y[va], proba(fit(X[tr], y[tr], C), X[va]))["macro_f1"] for C in C_GRID]
        grid += [{"mision": m, "C": C, "macro_f1_val": s} for C, s in zip(C_GRID, scores)]
        best = C_GRID[int(np.argmax(scores))]
        chosen[m] = f"C={best:.3g}"
        p_val[(stars.mission[val] == m).to_numpy()] = proba(fit(X[tr], y[tr], best), X[va])
        print(f"moment-{args.size} {m}: {chosen[m]} macro-F1 val={max(scores):.3f}", flush=True)
    table, conf = evaluate_by_mission(stars.mission[val].to_numpy(), y[val], p_val)
    table.insert(1, "hiperparametros", table.mision.map(chosen))
    name = f"moment_{args.size}"
    table.round(3).to_csv(OUT / f"{name}_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / f"{name}_grid.csv", index=False, encoding="utf-8")
    pd.concat(conf, names=["mision", "real"]).to_csv(OUT / f"{name}_val_confusion.csv", encoding="utf-8")
    cols = ["mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
