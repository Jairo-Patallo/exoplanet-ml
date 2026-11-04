"""Test de atajo del centroide (variante curva + centroide, 2026-11-03).

¿Se distingue NO_SIGNAL de PLANET / FALSE_POSITIVE mirando la vista de centroide SIN la zona del tránsito? Si sí, el
centroide delataría cómo se fabricaron los negativos (ruido, sector, apuntado) en vez de aportar física. Igual que el
test del flujo (shortcut_negatives.py): se tapa el 20 % central de cada vista y se mide el AUC de un boosting con
validación cruzada de 5 particiones dentro de train. 0,5 = no se distingue.

Uso:  python scripts/checks/shortcut_centroid.py   (PYTHONPATH=src)
Salida: results/sample/shortcut_centroid.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from exoplanet.data import MISSIONS
from exoplanet.views import load_centroid, load_sample

OUT = "results/sample/shortcut_centroid.csv"


def without_transit(v: np.ndarray, cut: float = 0.2) -> np.ndarray:
    n = v.shape[1]
    keep = np.abs(np.arange(n) - n // 2) > cut * n / 2
    return v[:, keep]


def main() -> None:
    stars = load_sample(["train"])
    c = load_centroid(stars.star_id)
    X = np.hstack([without_transit(c["local"]), without_transit(c["glob"])])
    rows = []
    for m in MISSIONS:
        for cls in ("PLANET", "FALSE_POSITIVE"):
            sel = ((stars.mission == m) & stars.disposition.isin([cls, "NO_SIGNAL"])).to_numpy()
            y = (stars.disposition[sel] == cls).to_numpy()
            auc = cross_val_score(HistGradientBoostingClassifier(random_state=0), X[sel], y, cv=5, scoring="roc_auc").mean()
            rows.append({"mission": m, "clase": cls, "n_train": int(sel.sum()), "auc_cv_centroide_sin_transito": round(auc, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
