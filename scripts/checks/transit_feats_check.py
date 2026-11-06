"""¿Aportan los tránsitos individuales algo que la CNN no ve? (propuesta P1, paso 1; criterio fijado en Notion antes de mirar)

Entrada: results/sample/transit_feats.csv (scripts/data/build_transit_feats.py) y predicciones de la CNN fuera de muestra:
  lomo: lomo_cnn_nodepth_fixed_pred.csv, train + val de cada misión con el modelo que no la vio (n grande);
  val:  final_val_pred.csv, la receta final entrenada solo con train (dentro de misión, solo val).
Tarea: PLANET frente a FALSE_POSITIVE. Puntuación de la CNN: logit de p_P / (p_P + p_FP).
1. Calibración de los rasgos (|z| par/impar, z_scatter, z_max) en planetas y FP.
2. AUC de cada rasgo (mayor -> FP) con IC bootstrap.
3. Ganancia: regresión logística fuera de pliegue (5 folds) [CNN] frente a [CNN + rasgos]; IC bootstrap pareado.
4. Casos difíciles: entre las estrellas que la CNN llama PLANET (argmax), ¿separan los rasgos planeta de FP?
Los NaN de los rasgos se imputan con 0 (su valor sin señal), sin indicador: el indicador delataría el número de tránsitos.

Uso:  PYTHONPATH=src python scripts/checks/transit_feats_check.py
Salida: results/models/transit_feats_check.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import exoplanet  # noqa: F401  (al importarse fuerza la salida en UTF-8 en Windows)

FEATS = ["abs_z_odd_even", "z_scatter", "z_max"]
PREDS = {"lomo": "results/models/lomo_cnn_nodepth_fixed_pred.csv", "val": "results/models/final_val_pred.csv"}
OUT = Path("results/models/transit_feats_check.csv")
N_BOOT, SEED = 1000, 0


def squash(z: pd.Series) -> pd.Series:
    """sign·log1p(|z|): las binarias dan z de decenas o cientos; sin esto dominarían la logística."""
    return np.sign(z) * np.log1p(np.abs(z))


def oof_proba(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.zeros(len(y))
    for fit, ev in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[fit], y[fit])
        p[ev] = m.predict_proba(X[ev])[:, 1]
    return p


def boot_auc(y, scores: list[np.ndarray], rng) -> np.ndarray:
    """(N_BOOT, len(scores)) ROC-AUC remuestreando estrellas (mismos índices para todas las puntuaciones)."""
    out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append([roc_auc_score(y[i], s[i]) for s in scores])
    return np.array(out)


def ci(a: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def analyse(df: pd.DataFrame, mission: str, source: str, subset: str, rng) -> list[dict]:
    y = (df.disposition == "FALSE_POSITIVE").to_numpy().astype(int)       # 1 = FP: rasgos altos -> FP
    rows, base = [], dict(mission=mission, source=source, subset=subset, n=len(df), n_fp=int(y.sum()))
    if y.min() == y.max() or min(y.sum(), len(y) - y.sum()) < 10:
        return rows
    X_f = df[FEATS].fillna(0).apply(squash).to_numpy()
    cnn = -df["cnn_logit"].to_numpy()                                       # mayor -> FP
    for name, s in [("cnn", cnn)] + [(f, X_f[:, j]) for j, f in enumerate(FEATS)]:
        b = boot_auc(y, [s], rng)[:, 0]
        rows.append({**base, "metric": f"auc_{name}", "value": roc_auc_score(y, s), "lo": ci(b)[0], "hi": ci(b)[1]})
    p_feats = oof_proba(X_f, y)
    b = boot_auc(y, [p_feats], rng)[:, 0]
    rows.append({**base, "metric": "auc_feats_logreg", "value": roc_auc_score(y, p_feats), "lo": ci(b)[0], "hi": ci(b)[1]})
    if subset == "todas":
        p0 = oof_proba(cnn[:, None], y)
        p1 = oof_proba(np.column_stack([cnn, X_f]), y)
        b = boot_auc(y, [p0, p1], rng)
        d = b[:, 1] - b[:, 0]
        rows += [{**base, "metric": "auc_cnn_logreg", "value": roc_auc_score(y, p0), "lo": ci(b[:, 0])[0], "hi": ci(b[:, 0])[1]},
                 {**base, "metric": "auc_cnn_plus_feats", "value": roc_auc_score(y, p1), "lo": ci(b[:, 1])[0], "hi": ci(b[:, 1])[1]},
                 {**base, "metric": "gain_auc", "value": roc_auc_score(y, p1) - roc_auc_score(y, p0), "lo": ci(d)[0], "hi": ci(d)[1]}]
    return rows


def calibration(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(["mission", "disposition"])
    return pd.DataFrame({"n": g.size(), "n_transits_med": g.n_transits.median(), "beta_med": g.beta.median(),
                         "med_abs_z_oe": g.abs_z_odd_even.median(), "p_abs_z_oe_gt3": g.abs_z_odd_even.apply(lambda s: (s > 3).mean()),
                         "med_z_scatter": g.z_scatter.median(), "p_z_scatter_gt3": g.z_scatter.apply(lambda s: (s > 3).mean())}).round(3)


def main() -> None:
    feats = pd.read_csv("results/sample/transit_feats.csv")
    feats = feats[feats.get("error", pd.Series(index=feats.index, dtype=object)).isna()].copy()
    z = ["z_odd_even", "z_scatter", "z_max"]          # z = ±inf: valor p de 0 o 1 (CSV anterior al recorte de features.py)
    feats[z] = feats[z].clip(-8, 38)
    feats["abs_z_odd_even"] = feats.z_odd_even.abs()
    print("Calibración (planeta con ruido bien medido: mediana |z| par/impar ≈ 0,67; P(|z| > 3) ≈ 0,003; z_scatter ≈ 0):")
    print(calibration(feats).to_string(), "\n")
    rng = np.random.default_rng(SEED)
    rows = []
    for source, path in PREDS.items():
        pred = pd.read_csv(path)
        pred = pred[pred.disposition.isin(["PLANET", "FALSE_POSITIVE"])]
        s = (pred.p_PLANET / (pred.p_PLANET + pred.p_FALSE_POSITIVE)).clip(1e-6, 1 - 1e-6)
        pred = pred.assign(cnn_logit=np.log(s / (1 - s)),
                           cnn_says_planet=pred[["p_PLANET", "p_FALSE_POSITIVE", "p_NO_SIGNAL"]].to_numpy().argmax(1) == 0)
        df = pred.merge(feats.drop(columns=["mission", "disposition"]), on="star_id")
        for mission, g in df.groupby("mission"):
            rows += analyse(g, mission, source, "todas", rng)
            rows += analyse(g[g.cnn_says_planet], mission, source, "la CNN dice PLANET", rng)
            if mission == "tess":
                rows += analyse(g[g.n_transits <= 8], mission, source, "≤ 8 tránsitos", rng)
                rows += analyse(g[g.n_transits > 8], mission, source, "> 8 tránsitos", rng)
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT, index=False, encoding="utf-8")
    show = res[res.metric.isin(["auc_cnn", "auc_feats_logreg", "gain_auc", "auc_abs_z_odd_even", "auc_z_scatter"])]
    print(show.round(3).to_string(index=False))
    t = res[(res.mission == "tess") & (res.source == "lomo")]
    gain = t[(t.subset == "todas") & (t.metric == "gain_auc")].iloc[0]
    hard = t[(t.subset == "la CNN dice PLANET") & (t.metric == "auc_feats_logreg")].iloc[0]
    go = gain.lo > 0 or (hard.value >= 0.60 and hard.lo > 0.5)
    print(f"\nCriterio (TESS, LOMO): ganancia de ROC-AUC {gain.value:+.3f} [{gain.lo:+.3f}, {gain.hi:+.3f}]; "
          f"rasgos entre los que la CNN llama planeta: AUC {hard.value:.3f} [{hard.lo:.3f}, {hard.hi:.3f}] "
          f"-> {'SE CUMPLE: pasar al paso 2' if go else 'NO se cumple: no pasar al paso 2'}")


if __name__ == "__main__":
    main()
