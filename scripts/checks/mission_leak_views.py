"""Test de fuga de misión sobre las vistas propias (tarea #44; repite #35, que usaba las vistas del dataset de HF).

Mismo método que legacy/mission_leak.py: un clasificador de misión (gradient boosting sin ajustar) entrenado en train y
evaluado en val, con las variables derivadas de la curva (legacy/features.py) y con las vistas plegadas (local+global).
Azar: balanced accuracy 1/3 y AUC 0.5. Si sigue acertando igual que en #35, la causa no es el preprocesado del
dataset sino las poblaciones de estrellas y planetas de cada misión.

Uso:  python scripts/checks/mission_leak_views.py   (PYTHONPATH="src;legacy" en Windows, "src:legacy" en Linux)
Salida: results/leak/mission_leak_own_views.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from exoplanet.data import DATA
from features import curve_features
from mission_leak import SEED, SUBSETS, ci   # legacy: mismos subconjuntos, semilla e IC que #35

VIEWS = DATA / "sample" / "views"
OUT = Path("results/leak/mission_leak_own_views.csv")
ORDER = ["k2", "kepler", "tess"]   # orden alfabético, el de sklearn


def load(split: str, qc: pd.DataFrame):
    df = qc[qc.split == split].reset_index(drop=True)
    v = {k: np.stack([np.load(VIEWS / f"{s}.npz")[k] for s in df.star_id]) for k in ("local", "odd", "even", "glob")}
    feats = {"variables derivadas de la curva": curve_features(v["local"], v["odd"], v["even"], v["glob"]).to_numpy(float),
             "vistas plegadas (local+global)": np.hstack([v["local"], v["glob"]])}
    return df, feats


def main() -> None:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool)]
    (tr, ftr), (va, fva) = load("train", qc), load("val", qc)
    rng = np.random.default_rng(SEED)
    bal = lambda y, p: balanced_accuracy_score(y, np.asarray(ORDER)[p.argmax(1)])
    auc = lambda y, p: roc_auc_score(y, p, multi_class="ovr", average="macro", labels=ORDER)
    rows = []
    for subset, classes in SUBSETS.items():
        mt, mv = tr.disposition.isin(classes).to_numpy(), va.disposition.isin(classes).to_numpy()
        ytr, yva = tr.mission[mt].to_numpy(), va.mission[mv].to_numpy()
        for fname in ftr:
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(ftr[fname][mt], ytr)
            proba = clf.predict_proba(fva[fname][mv])
            rows.append({"subconjunto": subset, "entradas": fname, "n_train": len(ytr), "n_val": len(yva),
                         "balanced_accuracy": ci(bal, yva, proba, rng), "auc_ovr_macro": ci(auc, yva, proba, rng)})
            print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
