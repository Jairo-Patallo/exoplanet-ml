"""Escalón 1: baseline dummy de vetting con el protocolo LOMO y la referencia in-domain.

Uso:  python scripts/baseline.py   (desde la raíz del repo)
Salida: results/baseline/dummy.csv. No lee el split test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from exoplanet.data import MISSIONS, TARGET, load_vetting, lomo_folds
from exoplanet.evaluation import POSITIVE, vetting_metrics

OUT = Path("results/baseline")
COLS = ["star_id"]


def evaluate(fit: pd.DataFrame, dev: pd.DataFrame) -> dict[str, float]:
    # "prior": predice la clase mayoritaria del fit y da como probabilidad su proporción de clases
    model = DummyClassifier(strategy="prior").fit(np.zeros((len(fit), 1)), fit[TARGET])
    X = np.zeros((len(dev), 1))
    p_fp = model.predict_proba(X)[:, list(model.classes_).index(POSITIVE)]
    return {"pred": model.predict(X[:1])[0], **vetting_metrics(dev[TARGET], model.predict(X), p_fp)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"protocolo": "LOMO", "mision_dev": m, "n_fit": len(fit), **evaluate(fit, dev)}
            for m, fit, _, dev in lomo_folds(COLS)]
    fit = load_vetting("train", columns=COLS)
    rows += [{"protocolo": "in-domain", "mision_dev": m, "n_fit": len(fit), **evaluate(fit, load_vetting("val", (m,), COLS))}
             for m in MISSIONS]
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT / "dummy.csv", index=False)


if __name__ == "__main__":
    main()
