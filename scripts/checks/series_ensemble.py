"""Conjuntos de semillas de la CNN sin plegar y combinación con BLS (research/12, 2026-11-06).

Por variante (results/models/series_runs/<tag>_seed*.csv/.npz, de series_cnn.py): media de las probabilidades de sus
semillas. Métricas en val:
  - todas las estrellas con curva: ROC-AUC, recall de candidatos con 1 % y 5 % de falsas alarmas, ruido que pasa con
    recall de candidatos del 90/95/99 %;
  - tránsitos inyectados en las curvas SIN_SEÑAL de val (los mismos en todas las ejecuciones): detectados por SNR con el
    umbral del 1 % de falsas alarmas reales;
  - estrellas con vista BLS: la variante sola y combinada con SDE + CNN plegada con negativos difíciles (regresión
    logística fuera de pliegue, triage_val.oof_logreg), frente a la línea base SDE + CNN plegada.

Uso:  PYTHONPATH="src;scripts/checks" .venv/Scripts/python scripts/checks/series_ensemble.py
Salida: results/models/series_ensemble.csv y series_ensemble_snr.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from triage_val import logit, metrics, oof_logreg

OUT, RUNS = "results/models", Path("results/models/series_runs")
SNR_BINS = [3, 5, 7, 10, 15, 20, 30, 50]


def noise_at_recall(y: np.ndarray, s: np.ndarray, rec: float) -> float:
    return float(np.mean(s[~y] >= np.quantile(s[y], 1 - rec)))


def summary(y: np.ndarray, s: np.ndarray) -> dict:
    return {**metrics(y, s), **{f"ruido_a_recall{int(r * 100)}": noise_at_recall(y, s, r) for r in (0.9, 0.95, 0.99)}}


def main() -> None:
    tags = sorted({f.stem.rsplit("_seed", 1)[0] for f in RUNS.glob("*.csv")})
    q = pd.read_csv("results/sample/qc_bls.csv")
    q = q[q.error.isna() & q["valid"].eq(True) & (q.split == "val")][["star_id", "bls_sde"]]
    h = pd.read_csv(f"{OUT}/hard_val_bls_pred.csv")
    rows, snr_rows, base_done = [], [], False
    for tag in tags:
        files = sorted(RUNS.glob(f"{tag}_seed*.csv"))
        preds = [pd.read_csv(f) for f in files]
        d = preds[0][["star_id", "mission", "disposition"]].assign(p=np.mean([p.p_candidato for p in preds], axis=0))
        y, s = (d.disposition != "NO_SIGNAL").to_numpy(), d.p.to_numpy()
        rows.append({"variante": tag, "semillas": len(files), "conjunto": "val completo", **summary(y, s)})
        for p in preds:                                  # cada semilla por separado, para ver la dispersión
            rows.append({"variante": tag, "semillas": 1, "conjunto": "val completo (una semilla)",
                         **summary(y, p.p_candidato.to_numpy())})

        thr = np.quantile(s[~y], 0.99)
        inj = [np.load(f.with_suffix(".npz")) for f in files]
        snr, p_inj = inj[0]["snr"], np.mean([z["p"] for z in inj], axis=0)
        det = pd.DataFrame({"snr": pd.cut(snr, SNR_BINS), "det": p_inj > thr}).groupby("snr", observed=True).det.mean()
        snr_rows.append({"variante": tag, "semillas": len(files), **{str(k): v for k, v in det.items()}})

        c = q.merge(h, on="star_id").merge(d[["star_id", "p"]], on="star_id")
        yc, st = (c.disposition != "NO_SIGNAL").to_numpy(), (c.mission + "_" + c.disposition).to_numpy()
        sde, cnn = np.log(c.bls_sde.clip(lower=1e-3)).to_numpy(), logit((c.p_PLANET + c.p_FALSE_POSITIVE).to_numpy())
        if not base_done:
            rows.append({"variante": "SDE + CNN plegada (línea base)", "conjunto": "estrellas BLS", **summary(yc, oof_logreg(np.c_[sde, cnn], yc, st))})
            base_done = True
        rows.append({"variante": tag, "semillas": len(files), "conjunto": "estrellas BLS", **summary(yc, c.p.to_numpy())})
        rows.append({"variante": f"{tag} + SDE + CNN plegada", "semillas": len(files), "conjunto": "estrellas BLS",
                     **summary(yc, oof_logreg(np.c_[sde, cnn, logit(c.p.to_numpy())], yc, st))})

    pd.set_option("display.width", 250)
    out, snr_out = pd.DataFrame(rows).round(3), pd.DataFrame(snr_rows).round(3)
    out.to_csv(f"{OUT}/series_ensemble.csv", index=False, encoding="utf-8")
    snr_out.to_csv(f"{OUT}/series_ensemble_snr.csv", index=False, encoding="utf-8")
    print(out[out.conjunto != "val completo (una semilla)"].drop(columns=["n_candidato", "n_sin_senal"]).to_string(index=False))
    one = out[out.conjunto == "val completo (una semilla)"].groupby("variante")[["roc_auc", "recall_fpr1"]].agg(["mean", "std"])
    print("\nsemillas sueltas (media y desviación):\n" + one.round(3).to_string())
    print("\ninyectados detectados con el umbral del 1 %, por SNR:\n" + snr_out.to_string(index=False))


if __name__ == "__main__":
    main()
