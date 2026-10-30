"""REFERENCIA CON CATÁLOGO, NO PORTABLE (usa teff, logg, log_rp_implied y log_period). Experimentos A, B y C: transferencia directa (sin adaptación) hacia K2 en vetting (PLANET vs FALSE_POSITIVE).

  A: entrena en Kepler -> K2     B: entrena en TESS -> K2     C: entrena en Kepler + TESS -> K2
Referencias: dummy (clase más frecuente) y regresión logística entrenada solo con K2 (CV agrupada por estrella).

Uso:  python scripts/transfer_k2.py   (desde la raíz del repo)
Reglas: los modelos fuente nunca ven K2. K2 se evalúa con train+val; el TEST de K2 no se toca.
Clase positiva = FALSE_POSITIVE (la minoritaria en K2: la PR-AUC del azar es su prevalencia).
Salida: results/catalog_reference/a_b_vs_k2.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.data import CLEAN_COLS, clean, load_split
from exoplanet.features import catalog_features, curve_features

SEED = 42
OUT = Path("results/catalog_reference")
VIEWS = {"flux_local": 81, "flux_global": 201, "flux_odd": 201, "flux_even": 201}
RAW = ["star_id", "mission", "radius", "teff", "logg", "mass", "cdpp_3hr"]
# Variables disponibles en las tres misiones (ni mass, ni metallicity, ni cdpp_3hr, ni log_rho_ratio)
FEATURES = ["log_period", "log_duty_cycle", "ingress_frac", "log_rp_implied", "teff", "logg", "odd_even_z"]
THRESHOLD = 0.5  # fijo (class_weight="balanced"): no se ajusta en el dominio destino


def load_vetting(mission: str, splits: tuple[str, ...]) -> pd.DataFrame:
    """Filas limpias PLANET/FP de una misión con las variables comunes y el objetivo y (1 = FP)."""
    cols = list(dict.fromkeys(RAW + CLEAN_COLS + list(VIEWS)))
    df = pd.concat([load_split(s, mission, cols).assign(split=s) for s in splits], ignore_index=True)
    df = clean(df)
    df = df[df.disposition != "NO_SIGNAL"].reset_index(drop=True)
    v = {k: np.stack([np.frombuffer(b, np.float32) for b in df[k]]).reshape(len(df), n) for k, n in VIEWS.items()}
    f = pd.concat([catalog_features(df),
                   curve_features(v["flux_local"], v["flux_odd"], v["flux_even"], v["flux_global"])], axis=1)
    X = pd.DataFrame({"log_period": np.log10(df.period_days), "log_duty_cycle": np.log10(f.duty_cycle),
                      "ingress_frac": f.ingress_frac, "log_rp_implied": np.log10(f.rp_implied),
                      "teff": df.teff, "logg": df.logg, "odd_even_z": f.odd_even_z})[FEATURES]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.assign(y=(df.disposition == "FALSE_POSITIVE").astype(int), star_id=df.star_id, split=df.split)


def make_model() -> object:
    # imputer y scaler se ajustan solo con los datos de entrenamiento de cada modelo
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))


def shift_prior(p: np.ndarray, pi_t: float, pi_s: float = 0.5) -> np.ndarray:
    """Corrige probabilidades de un modelo entrenado con prior pi_s a un prior objetivo pi_t (Bayes).
    class_weight="balanced" equivale a entrenar con prior 0.5. No cambia el ranking (ROC/PR-AUC), solo el umbral efectivo."""
    odds = p / (1 - p) * (pi_t / (1 - pi_t)) / (pi_s / (1 - pi_s))
    return odds / (1 + odds)


def estimate_prior(p: np.ndarray, pi_s: float = 0.5, n_iter: int = 100) -> float:
    """Estima el prior de la clase positiva en el destino SIN etiquetas (EM de Saerens et al. 2002)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi_t = pi_s
    for _ in range(n_iter):
        pi_t = shift_prior(p, pi_t, pi_s).mean()
    return float(pi_t)


def metrics(y: np.ndarray, p: np.ndarray, n_boot: int = 1000) -> dict:
    """ROC-AUC, PR-AUC (FP) y macro-F1 con IC 95 % bootstrap sobre filas."""
    fns = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score,
           "macro_f1": lambda y, p: f1_score(y, p >= THRESHOLD, average="macro")}
    rng = np.random.default_rng(SEED)
    boots = {k: [] for k in fns}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for k, fn in fns.items():
            boots[k].append(fn(y[i], p[i]))
    out = {}
    for k, fn in fns.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = f"{fn(y, p):.3f} [{lo:.3f},{hi:.3f}]"
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k2 = load_vetting("k2", ("train", "val"))
    print(f"K2 (train+val, limpio): n={len(k2)}, FP={k2.y.sum()} ({k2.y.mean():.1%})")
    rows = []

    # Dummy: predice siempre PLANET; el azar tiene AUC 0.5 y PR-AUC = prevalencia
    rows.append({"modelo": "dummy (siempre PLANET)", "evaluado en": "K2",
                 **metrics(k2.y.values, np.zeros(len(k2)))})
    rows[-1]["roc_auc"] = "0.500 (por definición)"
    rows[-1]["pr_auc"] = f"{k2.y.mean():.3f} (prevalencia)"

    # Techo realista: solo K2, CV agrupada por estrella (cada fila fuera de su pliegue de entrenamiento)
    oof = np.zeros(len(k2))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(k2[FEATURES], k2.y, k2.star_id):
        m = make_model().fit(k2.iloc[tr][FEATURES], k2.y.iloc[tr])
        oof[te] = m.predict_proba(k2.iloc[te][FEATURES])[:, 1]
    rows.append({"modelo": "K2 solo (CV 5 pliegues)", "evaluado en": "K2", **metrics(k2.y.values, oof)})

    for name, srcs in (("A: Kepler", ["kepler"]), ("B: TESS", ["tess"]), ("C: Kepler+TESS", ["kepler", "tess"])):
        tr = pd.concat([load_vetting(s, ("train",)) for s in srcs], ignore_index=True)
        model = make_model().fit(tr[FEATURES], tr.y)
        for s in srcs:  # referencia en la propia misión fuente (val)
            va = load_vetting(s, ("val",))
            rows.append({"modelo": f"{name} (referencia)", "evaluado en": f"{s} val",
                         **metrics(va.y.values, model.predict_proba(va[FEATURES])[:, 1])})
        rows.append({"modelo": name, "evaluado en": "K2",
                     **metrics(k2.y.values, model.predict_proba(k2[FEATURES])[:, 1])})
        print(f"{name}: train n={len(tr)} FP={tr.y.mean():.1%}")
        if name.startswith("A"):
            p = model.predict_proba(k2[FEATURES])[:, 1]
            pi_em = estimate_prior(p)
            print(f"Prior de FP en K2: estimado por EM sin etiquetas = {pi_em:.3f}; real = {k2.y.mean():.3f}")
            for tag, pi in (("EM, sin etiquetas", pi_em), ("prior real, oráculo", k2.y.mean())):
                rows.append({"modelo": f"A + prior ({tag}, prior={pi:.2f})", "evaluado en": "K2",
                             **metrics(k2.y.values, shift_prior(p, pi))})

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "a_b_vs_k2.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
