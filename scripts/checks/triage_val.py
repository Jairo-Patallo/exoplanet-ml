"""Prueba barata de triaje en modo búsqueda: CANDIDATO (PLANET o FP) frente a SIN_SEÑAL (NO_SIGNAL), solo la curva.

Estrellas de val con vista BLS válida (results/sample/qc_bls.csv). Puntuaciones comparadas:
  SDE           la de BLS (exoplanet.bls), sin ajustar nada
  CNN           P(PLANET) + P(FP) de la CNN sobre la vista plegada con BLS: receta final (final_val_bls_pred.csv) y
                con negativos difíciles (hard_val_bls_pred.csv)
  SDE + CNN     regresión logística con log(SDE) y logit(P candidato), predicción fuera de pliegue (5 folds
                estratificados por misión y clase): ningún punto se puntúa con un modelo que lo vio
Métrica principal: recall de candidatos con el umbral que deja pasar el 1 % de los SIN_SEÑAL (y el 5 %). ROC-AUC.
La etiqueta es de la estrella: un candidato al que BLS dio un periodo erróneo sigue contando como candidato.

Varios picos (si existe hard_val_peaks_pred.csv, de build_bls_peaks.py + hard_negatives.py peaks): cada estrella se
pliega con sus K picos de BLS; la regresión SDE + CNN se ajusta con el pico 1 de las estrellas de los otros folds, se
aplica a cada pico y la estrella toma la puntuación máxima. Se compara con usar solo el pico 1 (mismo modelo).

Uso:  PYTHONPATH=src .venv/Scripts/python scripts/checks/triage_val.py
Salida: results/models/triage_val.csv y triage_peaks_val.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "results/models"
FPRS = (0.01, 0.05)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def oof_logreg(X: np.ndarray, y: np.ndarray, strata: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y))
    for fit, ev in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strata):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[fit], y[fit])
        out[ev] = m.predict_proba(X[ev])[:, 1]
    return out


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    r = {"n_candidato": int(y.sum()), "n_sin_senal": int((~y).sum()), "roc_auc": roc_auc_score(y, s)}
    for f in FPRS:
        thr = np.quantile(s[~y], 1 - f)
        r[f"recall_fpr{int(f * 100)}"] = (s[y] > thr).mean()
    return r


def main() -> None:
    q = pd.read_csv("results/sample/qc_bls.csv")
    q = q[q.error.isna() & q["valid"].eq(True) & (q.split == "val")][["star_id", "bls_sde"]]
    d = q.merge(pd.read_csv(f"{OUT}/final_val_bls_pred.csv"), on="star_id")
    h = pd.read_csv(f"{OUT}/hard_val_bls_pred.csv").set_index("star_id")
    d["cnn_final"] = d.p_PLANET + d.p_FALSE_POSITIVE
    d["cnn_hard"] = (h.p_PLANET + h.p_FALSE_POSITIVE).reindex(d.star_id).to_numpy()
    y = (d.disposition != "NO_SIGNAL").to_numpy()
    strata = (d.mission + "_" + d.disposition).to_numpy()
    sde = np.log(d.bls_sde.clip(lower=1e-3)).to_numpy()
    scores = {"SDE": d.bls_sde.to_numpy(), "CNN (final)": d.cnn_final.to_numpy(), "CNN (negativos difíciles)": d.cnn_hard.to_numpy(),
              "SDE + CNN (final)": oof_logreg(np.c_[sde, logit(d.cnn_final)], y, strata),
              "SDE + CNN (negativos difíciles)": oof_logreg(np.c_[sde, logit(d.cnn_hard)], y, strata)}
    rows = []
    for name, s in scores.items():
        for mission in ["kepler", "k2", "tess", "todas"]:
            m = np.ones(len(d), bool) if mission == "todas" else (d.mission == mission).to_numpy()
            rows.append({"puntuacion": name, "mision": mission, **metrics(y[m], s[m])})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(f"{OUT}/triage_val.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


def peaks() -> None:
    d = pd.read_csv(f"{OUT}/hard_val_peaks_pred.csv")
    d["x_sde"], d["x_cnn"] = np.log(d.bls_sde.clip(lower=1e-3)), logit((d.p_PLANET + d.p_FALSE_POSITIVE).to_numpy())
    st = d[d.pico == 0].reset_index(drop=True)                 # estrellas con vista válida del pico 1
    d = d[d.star_id.isin(st.star_id)]
    y = (st.disposition != "NO_SIGNAL").to_numpy()
    strata = (st.mission + "_" + st.disposition).to_numpy()
    k = int(d.pico.max()) + 1
    best = pd.Series(np.nan, index=st.star_id)
    first = pd.Series(np.nan, index=st.star_id)
    cnn_max = d.groupby("star_id").x_cnn.max().reindex(st.star_id)
    for fit, ev in StratifiedKFold(5, shuffle=True, random_state=0).split(st, strata):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(st.loc[fit, ["x_sde", "x_cnn"]], y[fit])
        e = d[d.star_id.isin(st.star_id[ev])]
        sc = pd.Series(m.predict_proba(e[["x_sde", "x_cnn"]])[:, 1], index=e.index)
        best[st.star_id[ev]] = sc.groupby(e.star_id).max().reindex(st.star_id[ev]).to_numpy()
        first[st.star_id[ev]] = sc[e.pico == 0].groupby(e.star_id[e.pico == 0]).max().reindex(st.star_id[ev]).to_numpy()
    scores = {"SDE (pico 1)": st.bls_sde.to_numpy(), "CNN, pico 1": st.x_cnn.to_numpy(), f"CNN, mejor de {k} picos": cnn_max.to_numpy(),
              "SDE + CNN, pico 1": first.to_numpy(), f"SDE + CNN, mejor de {k} picos": best.to_numpy()}
    rows = []
    for name, s in scores.items():
        for mission in ["kepler", "k2", "tess", "todas"]:
            msk = np.ones(len(st), bool) if mission == "todas" else (st.mission == mission).to_numpy()
            rows.append({"puntuacion": name, "mision": mission, **metrics(y[msk], s[msk])})
    out = pd.DataFrame(rows)
    hit = d[d.match == "igual"].groupby("star_id").pico.min().reindex(st.star_id)   # primer pico con el periodo correcto
    print(f"\ncandidatos con el periodo correcto en el pico 1: {(hit[y] == 0).mean():.3f}; en alguno de los {k}: {hit[y].notna().mean():.3f}")
    out.round(3).to_csv(f"{OUT}/triage_peaks_val.csv", index=False, encoding="utf-8")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    if Path(f"{OUT}/hard_val_peaks_pred.csv").exists():
        peaks()
