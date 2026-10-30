"""Test de fuga de misión (tarea #35): ¿se puede adivinar la misión a partir de la curva?

Un clasificador de misión (gradient boosting sin ajustar, a propósito) se entrena con `train` y se evalúa en `val`, con
dos conjuntos de entradas: variables derivadas de la curva y las vistas plegadas. Si acierta mucho más que el azar,
un modelo de vetting puede usar "qué misión es" como atajo. Azar: balanced accuracy 1/3 y AUC 0.5.
Ojo: acertar no prueba por sí solo que haya un atajo, porque las poblaciones de planetas difieren de verdad entre
misiones (p. ej. TESS ve más planetas calientes); por eso se repite dentro de cada clase.

Uso:  python scripts/mission_leak.py   (desde la raíz del repo)
Salida: results/leak/mission_leak.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from exoplanet.data import MISSIONS, TARGET, load_split
from exoplanet.features import curve_features

OUT = Path("results/leak")
SEED = 42
N_BOOT = 200
VIEWS = ["flux_local", "flux_odd", "flux_even", "flux_global"]
SUBSETS = {"PLANET + FP": ["PLANET", "FALSE_POSITIVE"], "solo PLANET": ["PLANET"], "solo FP": ["FALSE_POSITIVE"],
           "con NO_SIGNAL (todas)": ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]}


def load(split: str):
    df = load_split(split, columns=["mission", TARGET] + VIEWS)
    views = {v: np.stack([np.frombuffer(b, np.float32) for b in df[v]]) for v in VIEWS}
    cf = curve_features(views["flux_local"], views["flux_odd"], views["flux_even"], views["flux_global"])
    feats = {"variables derivadas de la curva": cf.to_numpy(float),
             "vistas plegadas (local+global)": np.hstack([views["flux_local"], views["flux_global"]])}
    return df[["mission", TARGET]].reset_index(drop=True), feats


def ci(score_fn, y, proba, rng) -> str:
    point = score_fn(y, proba)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == len(MISSIONS):
            boots.append(score_fn(y[i], proba[i]))
    return f"{point:.3f} [{np.percentile(boots, 2.5):.3f},{np.percentile(boots, 97.5):.3f}]"


def main() -> None:
    (tr, ftr), (va, fva) = load("train"), load("val")
    rng = np.random.default_rng(SEED)
    order = sorted(MISSIONS)   # sklearn ordena las clases alfabéticamente (predict_proba y roc_auc_score)
    bal = lambda y, p: balanced_accuracy_score(y, np.asarray(order)[p.argmax(1)])
    auc = lambda y, p: roc_auc_score(y, p, multi_class="ovr", average="macro", labels=order)
    rows = []
    for subset, classes in SUBSETS.items():
        mt, mv = tr[TARGET].isin(classes).to_numpy(), va[TARGET].isin(classes).to_numpy()
        ytr, yva = tr.mission[mt].to_numpy(), va.mission[mv].to_numpy()
        for fname in ftr:
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
            clf.fit(ftr[fname][mt], ytr)
            assert list(clf.classes_) == order
            proba = clf.predict_proba(fva[fname][mv])
            rows.append({"subconjunto": subset, "entradas": fname, "n_train": len(ytr), "n_val": len(yva),
                         "balanced_accuracy": ci(bal, yva, proba, rng), "auc_ovr_macro": ci(auc, yva, proba, rng)})
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "mission_leak.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
