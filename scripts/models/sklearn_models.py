"""Modelos de sklearn de 3 clases dentro de cada misión (fase 2): regresión logística y kNN, con las métricas de #51.

Dos variantes de entrada (decisión 2026-11-02), las mismas para todos los modelos:
  A · variables físicas de la curva (legacy/features.py): profundidad y SNR con signo (log con signo), ingress_frac,
      odd_even_z y secondary_z. ingress_frac (sin caída) y odd_even_z (par/impar no válido) se imputan con la mediana
      de train, sin indicador de "faltaba" (delataría el número de tránsitos, un metadato del proceso).
  B · vistas local (81) y global (201) normalizadas por estrella al estilo AstroNet: se resta la mediana y se divide
      por la profundidad (mediana - mínimo de la local), así la caída vale ~-1; se añade log10 de esa profundidad.
Un modelo por misión; imputación y StandardScaler ajustados solo en train; hiperparámetros elegidos en val por macro-F1.
  logreg: class_weight="balanced", C en 1e-4..100.
  knn:    k en 1..51 y pesos uniformes o por distancia (euclídea); sin pesos de clase (kNN no los admite).
  knn_pca: solo vistas (B). PCA ajustada en train antes del kNN, con 5..80 componentes: en 283 dimensiones el NO_SIGNAL
           normalizado es ruido puro y queda lejos de todo (maldición de la dimensionalidad, research/06).
  hgb:    HistGradientBoosting con class_weight="balanced" y parada temprana (15 % de train, sin tocar val). Recibe los
          datos ya imputados, así no puede usar el patrón de NaN (número de tránsitos) como atajo.
El test no se toca. Solo estrellas con vista válida.

Uso:  python scripts/models/sklearn_models.py [--models logreg knn]   (PYTHONPATH="src;legacy" en Windows)
Salida en results/models/, por modelo: <modelo>_val.csv (métricas en val con IC 95 %), <modelo>_grid.csv (macro-F1 por
      hiperparámetro) y <modelo>_val_confusion.csv; además logreg_coef_A.csv (coeficientes estandarizados de A).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet.data import CLASSES, MISSIONS
from exoplanet.evaluation import evaluate_by_mission, point_metrics
from exoplanet.views import load_sample, load_views, normalize
from features import curve_features

OUT = Path("results/models")
SEED = 0
K_GRID = (1, 3, 5, 7, 11, 15, 21, 31, 51)
ONLY_VARIANTS = {"knn_pca": ("B_vistas",)}   # con 5 variables físicas la PCA no tiene sentido
MODELS = {
    "logreg": ([{"C": C} for C in np.logspace(-4, 2, 13)],
               lambda p: LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000, random_state=SEED)),
    "knn": ([{"k": k, "weights": w} for k in K_GRID for w in ("uniform", "distance")],
            lambda p: KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"])),
    "knn_pca": ([{"pca": n, "k": k, "weights": w} for n in (5, 10, 20, 40, 80) for k in K_GRID for w in ("uniform", "distance")],
                lambda p: make_pipeline(PCA(n_components=p["pca"], random_state=SEED),
                                        KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"]))),
    "hgb": ([{"lr": lr, "leaves": n, "min_leaf": m, "l2": l2} for lr in (0.03, 0.1) for n in (7, 15, 31)
             for m in (10, 30) for l2 in (0.0, 1.0)],
            lambda p: HistGradientBoostingClassifier(
                learning_rate=p["lr"], max_leaf_nodes=p["leaves"], min_samples_leaf=p["min_leaf"], l2_regularization=p["l2"],
                max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                class_weight="balanced", random_state=SEED)),
}


def symlog(x: np.ndarray) -> np.ndarray:
    """log10 con signo: comprime varios órdenes de magnitud y conserva si hay caída (+) o no (-)."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def physical_features(v: dict) -> pd.DataFrame:
    f = curve_features(v["local"], v["odd"], v["even"], v["glob"], mask_no_dip=False)
    return f.assign(depth_local_ppm=symlog(f.depth_local_ppm), snr_local=symlog(f.snr_local))


def fit(model: str, params: dict, X: np.ndarray, y: np.ndarray):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MODELS[model][1](params)).fit(X, y)


def proba(model, X: np.ndarray) -> np.ndarray:
    """Probabilidades con columnas en el orden de CLASSES (sklearn las ordena alfabéticamente)."""
    order = [list(model.classes_).index(c) for c in CLASSES]
    return model.predict_proba(X)[:, order]


def run(model: str, qc: pd.DataFrame, inputs: dict, feat_names: list[str]) -> None:
    grid_params = MODELS[model][0]
    y, val = qc.disposition.to_numpy(), (qc.split == "val").to_numpy()
    grid, tables, confs, coefs = [], [], {}, []
    for name, X in inputs.items():
        if name not in ONLY_VARIANTS.get(model, inputs):
            continue
        p_val, chosen = np.zeros((val.sum(), len(CLASSES))), {}
        for m in MISSIONS:
            tr, va = (qc.mission == m).to_numpy() & ~val, (qc.mission == m).to_numpy() & val
            scores = [point_metrics(y[va], proba(fit(model, p, X[tr], y[tr]), X[va]))["macro_f1"] for p in grid_params]
            grid += [{"variante": name, "mision": m, **p, "macro_f1_val": s} for p, s in zip(grid_params, scores)]
            best = grid_params[int(np.argmax(scores))]
            fitted = fit(model, best, X[tr], y[tr])
            p_val[(qc.mission[val] == m).to_numpy()] = proba(fitted, X[va])
            chosen[m] = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items())
            if model == "logreg" and name == "A_fisicas":
                lr = fitted[-1]
                coefs += [{"mision": m, "clase": c, **dict(zip(feat_names, lr.coef_[list(lr.classes_).index(c)]))} for c in CLASSES]
            print(f"{model} {name} {m}: {chosen[m]} macro-F1 val={max(scores):.3f}", flush=True)
        table, conf = evaluate_by_mission(qc.mission[val].to_numpy(), y[val], p_val)
        tables.append(table.assign(variante=name, hiperparametros=table.mision.map(chosen)))
        confs.update({(name, m): c for m, c in conf.items()})

    out = pd.concat(tables, ignore_index=True)
    out = out[["variante", "mision", "hiperparametros"] + [c for c in out if c not in ("variante", "mision", "hiperparametros")]]
    out.round(3).to_csv(OUT / f"{model}_val.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid).round(4).to_csv(OUT / f"{model}_grid.csv", index=False, encoding="utf-8")
    pd.concat(confs, names=["variante", "mision", "real"]).to_csv(OUT / f"{model}_val_confusion.csv", encoding="utf-8")
    if coefs:
        pd.DataFrame(coefs).round(3).to_csv(OUT / "logreg_coef_A.csv", index=False, encoding="utf-8")
    cols = ["variante", "mision", "macro_f1", "macro_f1_lo", "macro_f1_hi", "pr_auc_macro"] + [f"f1_{c}" for c in CLASSES]
    print("\n" + out[cols].round(3).to_string(index=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    qc = load_sample()
    v = load_views(qc.star_id, ("local", "odd", "even", "glob"))
    feats = physical_features(v)
    inputs = {"A_fisicas": feats.to_numpy(), "B_vistas": np.hstack(normalize(v["local"], v["glob"]))}
    OUT.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        run(model, qc, inputs, list(feats.columns))


if __name__ == "__main__":
    main()
