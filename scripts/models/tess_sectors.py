"""¿Más tránsitos ayudan en TESS? LOMO con 1 sector frente a 2 en las mismas estrellas (fase 3, 2026-11-04).

Modelo: el LOMO de TESS (solo Kepler + K2, 3 semillas; data/models/lomo_tess_seed*.pt de scripts/models/adabn.py).
Estrellas: parejas (positivo, negativo) de scripts/data/build_tess_2sectors.py con vista válida en las dos versiones.
Única diferencia entre las dos evaluaciones: la vista plegada con 1 sector (data/sample/views) o con 2 (views_tess2).
IC 95 % de la diferencia con bootstrap pareado por pareja (se remuestrean parejas, no estrellas sueltas).

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/tess_sectors.py
Salida: results/models/tess_sectors.csv y tess_sectors_pred.csv
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

OUT, MODELS = Path("results/models"), Path("data/models")
CFG, SEEDS, N_BOOT = {"base": 16, "dropout": 0.5}, (0, 1, 2), 1000


def load_model(seed: int) -> AstroCNN:
    m = AstroCNN(CFG["base"], CFG["dropout"], torch.zeros(1), torch.ones(1))   # la profundidad sale del state_dict
    m.load_state_dict(torch.load(MODELS / f"lomo_tess_seed{seed}.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def stars() -> pd.DataFrame:
    """Parejas completas: las dos estrellas válidas con 1 sector (qc.csv) y con 2 (qc_tess2.csv)."""
    q2 = pd.read_csv("results/sample/qc_tess2.csv")
    q2 = q2[q2.error.isna() & q2["valid"].eq(True)]
    ok = q2.groupby("pair_id").star_id.transform("size").eq(2)
    q1 = pd.read_csv("results/sample/qc.csv").set_index("star_id")
    s = q2[ok].assign(mission="tess", n_transits_1=lambda d: d.star_id.map(q1.n_transits))
    return s.rename(columns={"n_transits": "n_transits_2"}).reset_index(drop=True)


def main() -> None:
    s = stars()
    y = s.disposition.to_numpy()
    print(f"{len(s)} estrellas ({s.disposition.value_counts().to_dict()})", flush=True)
    models = [load_model(seed) for seed in SEEDS]
    probas = {}
    for name, views in (("1_sector", DATA / "sample" / "views"), ("2_sectores", DATA / "sample" / "views_tess2")):
        x = inputs(ViewDataset(s, views_dir=views))
        probas[name] = np.mean([predict(m, x, amp=True) for m in models], axis=0)

    pairs = s.pair_id.to_numpy()
    groups = [np.flatnonzero(pairs == p) for p in np.unique(pairs)]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(point_metrics(y[idx], probas["2_sectores"][idx])["pr_auc_macro"]
                    - point_metrics(y[idx], probas["1_sector"][idx])["pr_auc_macro"])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    rows = []
    for name, p in probas.items():
        m = point_metrics(y, p)
        pred = np.asarray(CLASSES)[p.argmax(1)]
        col = "n_transits_1" if name == "1_sector" else "n_transits_2"
        rows.append({"version": name, "n": len(y), "pr_auc_macro": m["pr_auc_macro"],
                     **{f"pr_auc_{c}": m[f"pr_auc_{c}"] for c in CLASSES}, "macro_f1": m["macro_f1"],
                     **{f"f1_{c}": m[f"f1_{c}"] for c in CLASSES},
                     **{f"acierto_{c}": (pred[y == c] == c).mean() for c in CLASSES},
                     **{f"transitos_med_{c}": s.loc[y == c, col].median() for c in CLASSES}})
    out = pd.DataFrame(rows)
    out["dif_pr_auc_ic95"] = ["", f"[{lo:+.3f}, {hi:+.3f}]"]
    print(out.round(3).T.to_string(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT / "tess_sectors.csv", index=False, encoding="utf-8")
    s[["star_id", "pair_id", "disposition", "n_transits_1", "n_transits_2"]].assign(
        **{f"{n}_p_{c}": p[:, i] for n, p in probas.items() for i, c in enumerate(CLASSES)}
    ).round(4).to_csv(OUT / "tess_sectors_pred.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
