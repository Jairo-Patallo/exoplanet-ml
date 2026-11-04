"""¿Dónde falla la CNN en TESS cuando no lo ha visto (LOMO)? Diagnóstico para decidir entre otro plegado o un modelo
más complejo (2026-11-04). Solo análisis: no se usa para elegir nada.

Cruza las predicciones por estrella del fold con TESS excluido (lomo_cnn_tess_pred.csv) con la calidad de cada curva:
número de tránsitos (qc.csv), señal/ruido y profundidad (exoplanet.features), y mide el macro-F1 y el acierto por clase
en cada tramo. Si el error se concentra en pocos tránsitos o poca señal, el problema son los datos (más sectores); si es
plano, es el modelo o la física.

Uso:  python scripts/checks/tess_errors.py   (PYTHONPATH=src)
Salida: results/models/tess_errors.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from exoplanet.data import CLASSES
from exoplanet.features import curve_features
from exoplanet.views import load_views

PRED = "results/models/lomo_cnn_tess_pred.csv"
OUT = "results/models/tess_errors.csv"


def main() -> None:
    p = pd.read_csv(PRED)
    qc = pd.read_csv("results/sample/qc.csv")[["star_id", "n_transits", "period_days", "duration_hrs"]]
    p = p.merge(qc, on="star_id")
    v = load_views(p.star_id, ("local", "odd", "even", "glob"))
    f = curve_features(v["local"], v["odd"], v["even"], v["glob"], mask_no_dip=False)
    p["snr"], p["depth_ppm"] = f.snr_local.to_numpy(), f.depth_local_ppm.to_numpy()
    p["pred"] = np.asarray(CLASSES)[p[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)]
    p["ok"] = p.pred == p.disposition
    bins = {
        "n_transits": [0, 2, 4, 8, 20, 1e9],
        "snr": [-1e9, 7, 15, 30, 100, 1e9],
        "period_days": [0, 2, 5, 10, 27, 1e9],
    }
    rows = []
    for var, edges in bins.items():
        p["tramo"] = pd.cut(p[var], edges)
        for t, g in p.groupby("tramo", observed=True):
            pos = g[g.disposition != "NO_SIGNAL"]
            rows.append({"variable": var, "tramo": str(t), "n": len(g), "n_positivos": len(pos),
                         "macro_f1": f1_score(g.disposition, g.pred, labels=list(CLASSES), average="macro", zero_division=0),
                         **{f"acierto_{c}": g.ok[g.disposition == c].mean() for c in CLASSES}})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(out.to_string(index=False))
    fp = p[p.disposition == "FALSE_POSITIVE"]
    print("\nFP de TESS mal clasificados:", (~fp.ok).sum(), "de", len(fp), "| como PLANET:", (fp.pred == "PLANET").sum(),
          "| como NO_SIGNAL:", (fp.pred == "NO_SIGNAL").sum())
    print("SNR mediana de FP acertados / tomados por PLANET:", round(fp.snr[fp.ok].median(), 1), "/",
          round(fp.snr[fp.pred == "PLANET"].median(), 1))


if __name__ == "__main__":
    main()
