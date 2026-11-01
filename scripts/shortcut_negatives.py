"""Test de atajo de los negativos NO_SIGNAL (tarea #32).

¿Se puede distinguir NO_SIGNAL de los positivos (PLANET + FP) SIN mirar el tránsito, solo con detalles del proceso?
Si sí, un modelo podría aprender ese atajo en vez de la forma del tránsito. Por misión: clasificador entrenado en train
con rasgos de "nuisance" (puntos, cobertura, bins vacíos, ruido fuera del tránsito) y ROC-AUC en val+test.
AUC ~0.5 = sin atajo. También comprueba que ninguna vista válida sea constante o tenga NaN.

Uso:  python scripts/shortcut_negatives.py   (PYTHONPATH=src)
Salida: results/sample/shortcut_negatives.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.data import DATA

VIEWS = DATA / "sample" / "views"
OUT = "results/sample/shortcut_negatives.csv"
QC_FEATURES = ["n_points", "data_days", "cadence_days", "n_transits", "n_points_in_transit",
               "frac_empty_local", "frac_empty_glob", "n_files"]
NOISE = "noise_out"
SEED = 0


def out_of_transit_noise(glob: np.ndarray, cut: float = 0.2) -> float:
    """Ruido robusto (MAD) de la vista global quitando el 20 % central, donde está el tránsito."""
    half = int(len(glob) * cut / 2)
    c = len(glob) // 2
    side = np.concatenate([glob[: c - half], glob[c + half + 1:]])
    return float(np.median(np.abs(side - np.median(side))))


def load() -> pd.DataFrame:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool)].copy()
    noise, bad = [], []
    for s in qc.star_id:
        v = np.load(VIEWS / f"{s}.npz")
        noise.append(out_of_transit_noise(v["glob"]))
        bad.append(any(not np.isfinite(v[k]).all() or np.ptp(v[k]) == 0 for k in ("local", "glob")))
    qc[NOISE] = np.log10(np.maximum(noise, 1e-7))
    qc["bad_view"] = bad
    qc["y"] = (qc.disposition == "NO_SIGNAL").astype(int)
    return qc


def auc(model, train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    model.fit(train[cols], train.y)
    return roc_auc_score(test.y, model.predict_proba(test[cols])[:, 1])


def main() -> None:
    df = load()
    print(f"Vistas válidas: {len(df)}; constantes o con NaN: {int(df.bad_view.sum())}")
    rows = []
    for mission, g in df.groupby("mission"):
        train, test = g[g.split == "train"], g[g.split != "train"]
        for name, cols in [("qc", QC_FEATURES), ("ruido", [NOISE]), ("qc+ruido", QC_FEATURES + [NOISE])]:
            rows.append({"mission": mission, "rasgos": name,
                         "auc_logreg": auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), train, test, cols),
                         "auc_hgb": auc(HistGradientBoostingClassifier(random_state=SEED), train, test, cols)})
        for c in QC_FEATURES + [NOISE]:      # AUC de cada rasgo solo, sin modelo (0.5 = no separa)
            a = roc_auc_score(test.y, test[c]) if test[c].nunique() > 1 else 0.5
            rows.append({"mission": mission, "rasgos": c, "auc_logreg": max(a, 1 - a)})
        rows.append({"mission": mission, "rasgos": "n_train/n_test", "auc_logreg": len(train), "auc_hgb": len(test)})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT, index=False, encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
