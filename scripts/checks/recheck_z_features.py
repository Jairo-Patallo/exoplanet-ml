"""Recalcula odd_even_z y secondary_z sobre las vistas propias (tarea #46).

En el dataset de HF la vista global y las par/impar no estaban centradas en el tránsito, así que legacy/features.py
medía estas dos variables en la zona equivocada. Aquí se repiten, sobre la muestra A (solo PLANET y FP con vista
válida), las tres cosas del EDA que las usaban: AUC por misión (FP vs PLANET), calibración en PLANET y reglas z > 3.

Uso:  python scripts/checks/recheck_z_features.py   (PYTHONPATH="src;legacy" en Windows)
Salida: results/eda/z_features_own_views.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from exoplanet.data import DATA
from features import curve_features

VIEWS = DATA / "sample" / "views"
OUT = "results/eda/z_features_own_views.csv"
FEATS = ["odd_even_z", "secondary_z"]


def main() -> None:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool) & (qc.disposition != "NO_SIGNAL") & (qc.split == "train")].reset_index(drop=True)
    v = {k: np.stack([np.load(VIEWS / f"{s}.npz")[k] for s in qc.star_id]) for k in ("local", "odd", "even", "glob")}
    df = pd.concat([qc[["mission", "disposition", "odd_even_valid"]], curve_features(v["local"], v["odd"], v["even"], v["glob"])], axis=1)
    df.loc[~df.odd_even_valid.astype(bool), "odd_even_z"] = np.nan
    rows = []
    for mission, g in [("todas", df)] + list(df.groupby("mission")):
        y = (g.disposition == "FALSE_POSITIVE").to_numpy()
        for f in FEATS:
            ok = g[f].notna().to_numpy()
            zp = g.loc[(~y) & ok, f].abs()
            pred = (g[f] > 3).to_numpy()
            tp = (pred & y).sum()
            rows.append({"mission": mission, "variable": f, "n": int(ok.sum()),
                         "auc_fp": roc_auc_score(y[ok], g[f][ok]),
                         "planet_median_abs_z": zp.median(), "planet_p_abs_z_gt3": (zp > 3).mean(),
                         "regla_z3_pct_marcado": pred.mean(), "regla_z3_precision_fp": tp / max(pred.sum(), 1),
                         "regla_z3_recall_fp": tp / y.sum(), "prevalencia_fp": y.mean()})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
