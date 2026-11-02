"""Test de atajo de los negativos NO_SIGNAL (tarea #32).

¿Se puede distinguir NO_SIGNAL de los positivos (PLANET + FP) SIN mirar el tránsito, solo con detalles del proceso?
Si sí, un modelo podría aprender ese atajo en vez de la forma del tránsito. Por misión, dos pruebas (AUC ~0.5 = sin atajo):
  1. Metadatos del proceso (puntos, cobertura, bins vacíos) y ruido fuera del tránsito: entrenado en train, AUC en val.
  2. Lo que ve el modelo: la vista global con el tránsito tapado (20 % central), por clase positiva; AUC con validación
     cruzada de 5 particiones dentro de train.
El test no se toca (auditoría 2026-11-01). También comprueba que ninguna vista válida sea constante o tenga NaN.

Uso:  python scripts/shortcut_negatives.py   (PYTHONPATH=src)
Salida: results/sample/shortcut_negatives.csv y shortcut_negatives_views.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.data import DATA

VIEWS = DATA / "sample" / "views"
OUT = "results/sample/shortcut_negatives.csv"
QC_FEATURES = ["n_points", "data_days", "cadence_days", "n_transits", "n_points_in_transit",
               "frac_empty_local", "frac_empty_glob", "n_files"]
NOISE = "noise_out"
SEED = 0


def without_transit(glob: np.ndarray, cut: float = 0.2) -> np.ndarray:
    """Vista global sin el 20 % central, donde está el tránsito."""
    half, c = int(len(glob) * cut / 2), len(glob) // 2
    return np.concatenate([glob[: c - half], glob[c + half + 1:]])


def out_of_transit_noise(glob: np.ndarray) -> float:
    """Ruido robusto (MAD) de la vista global fuera del tránsito."""
    side = without_transit(glob)
    return float(np.median(np.abs(side - np.median(side))))


def load() -> pd.DataFrame:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool)].copy()
    noise, bad, side = [], [], []
    for s in qc.star_id:
        v = np.load(VIEWS / f"{s}.npz")
        noise.append(out_of_transit_noise(v["glob"]))
        bad.append(any(not np.isfinite(v[k]).all() or np.ptp(v[k]) == 0 for k in ("local", "glob")))
        side.append(without_transit(v["glob"]))
    qc[NOISE] = np.log10(np.maximum(noise, 1e-7))
    qc["bad_view"] = bad
    qc["y"] = (qc.disposition == "NO_SIGNAL").astype(int)
    return qc.reset_index(drop=True), np.stack(side) - 1


def auc(model, train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    model.fit(train[cols], train.y)
    return roc_auc_score(test.y, model.predict_proba(test[cols])[:, 1])


def main() -> None:
    df, side = load()
    print(f"Vistas válidas: {len(df)}; constantes o con NaN: {int(df.bad_view.sum())}")
    rows, view_rows = [], []
    for mission, g in df.groupby("mission"):
        train, val = g[g.split == "train"], g[g.split == "val"]
        for name, cols in [("qc", QC_FEATURES), ("ruido", [NOISE]), ("qc+ruido", QC_FEATURES + [NOISE])]:
            rows.append({"mission": mission, "rasgos": name,
                         "auc_logreg": auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), train, val, cols),
                         "auc_hgb": auc(HistGradientBoostingClassifier(random_state=SEED), train, val, cols)})
        for c in QC_FEATURES + [NOISE]:      # AUC de cada rasgo solo, sin modelo (0.5 = no separa)
            a = roc_auc_score(val.y, val[c]) if val[c].nunique() > 1 else 0.5
            rows.append({"mission": mission, "rasgos": c, "auc_logreg": max(a, 1 - a)})
        rows.append({"mission": mission, "rasgos": "n_train/n_val", "auc_logreg": len(train), "auc_hgb": len(val)})
        for cls in ["PLANET", "FALSE_POSITIVE"]:
            t = train[train.disposition.isin([cls, "NO_SIGNAL"])]
            cv = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), side[t.index], t.y, cv=5, scoring="roc_auc")
            view_rows.append({"mission": mission, "clase": cls, "n_train": len(t), "auc_cv_vista_sin_transito": cv.mean()})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT, index=False, encoding="utf-8")
    views = pd.DataFrame(view_rows).round(3)
    views.to_csv(OUT.replace(".csv", "_views.csv"), index=False, encoding="utf-8")
    print(res.to_string(index=False))
    print(views.to_string(index=False))


if __name__ == "__main__":
    main()
