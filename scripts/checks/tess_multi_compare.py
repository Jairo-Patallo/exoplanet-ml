"""1 sector frente a varios en TESS (fase 6, opción A): comparación en val y control del atajo.

Predicciones: results/models/final_val_pred.csv (1 sector) y multi_val_pred.csv (multisector), misma receta, mismas
5 semillas y mismas estrellas de val.
1. Macro-F1 por misión con los dos modelos; en TESS, diferencia con IC 95 % por bootstrap emparejado (por estrella),
   en todas las estrellas y por separado en las que ganan sectores (pareja usada con k >= 2) y en las que no.
   Si mejorara igual donde no hay sectores nuevos, sospecharíamos de un efecto no debido a ver más tránsitos.
2. Atajo: en las estrellas de TESS de train, ¿separa la clase solo el número de sectores o el ruido de la vista
   global (desviación robusta fuera del tránsito)? ROC-AUC con las vistas de 1 sector y con las multisector (~0,5 = sin atajo).

Uso:  PYTHONPATH=src .venv/Scripts/python scripts/checks/tess_multi_compare.py
Salida: results/models/tess_multi_compare.csv y tess_multi_shortcut.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from exoplanet.data import CLASSES, DATA
from exoplanet.views import SAMPLE_VIEWS, load_sample

OUT, MULTI = "results/models", DATA / "sample" / "views_tess_multi"
N_BOOT = 2000


def labels(p: pd.DataFrame) -> np.ndarray:
    return np.array(CLASSES)[p[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)]


def macro_f1(y, yhat) -> float:
    return f1_score(y, yhat, labels=CLASSES, average="macro", zero_division=0)


def paired_diff(y, a, b, rng) -> tuple[float, float, float]:
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    d = np.array([macro_f1(y[i], b[i]) - macro_f1(y[i], a[i]) for i in idx])
    return macro_f1(y, b) - macro_f1(y, a), *np.percentile(d, [2.5, 97.5])


def noise(star_ids, views_dir) -> np.ndarray:
    out = []
    for s in star_ids:
        with np.load(views_dir / f"{s}.npz") as f:
            g = f["glob"]
        g = g[np.isfinite(g)]
        side = np.r_[g[:60], g[-60:]] if len(g) > 120 else g          # lejos del tránsito (centrado en la vista global)
        out.append(1.4826 * np.median(np.abs(side - np.median(side))))
    return np.array(out)


def main() -> None:
    rng = np.random.default_rng(0)
    a, b = pd.read_csv(f"{OUT}/final_val_pred.csv"), pd.read_csv(f"{OUT}/multi_val_pred.csv")
    assert (a.star_id.to_numpy() == b.star_id.to_numpy()).all()
    qc = pd.read_csv("results/sample/qc_tess_multi.csv")
    used = set(qc.star_id[qc.usada.eq(True)])
    ya, yb, y = labels(a), labels(b), a.disposition.to_numpy()
    rows = []
    for m in ["kepler", "k2", "tess"]:
        k = (a.mission == m).to_numpy()
        rows.append({"grupo": m, "n": int(k.sum()), "macro_f1_1sector": macro_f1(y[k], ya[k]), "macro_f1_multi": macro_f1(y[k], yb[k])})
    for name, k in [("tess, todas", a.mission.eq("tess")), ("tess, ganan sectores", a.star_id.isin(used)),
                    ("tess, sin sectores nuevos", a.mission.eq("tess") & ~a.star_id.isin(used))]:
        k = k.to_numpy()
        d, lo, hi = paired_diff(y[k], ya[k], yb[k], rng)
        rows.append({"grupo": name, "n": int(k.sum()), "macro_f1_1sector": macro_f1(y[k], ya[k]), "macro_f1_multi": macro_f1(y[k], yb[k]),
                     "diferencia": d, "ic_lo": lo, "ic_hi": hi,
                     "f1_fp_1sector": f1_score(y[k], ya[k], labels=["FALSE_POSITIVE"], average="macro", zero_division=0),
                     "f1_fp_multi": f1_score(y[k], yb[k], labels=["FALSE_POSITIVE"], average="macro", zero_division=0)})
    res = pd.DataFrame(rows).round(3)
    res.to_csv(f"{OUT}/tess_multi_compare.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))

    tr = load_sample(("train",))
    tr = tr[tr.mission == "tess"].reset_index(drop=True)
    k_used = tr.star_id.map(qc.set_index("star_id").k.where(qc.set_index("star_id").usada.eq(True))).fillna(1).to_numpy()
    feats = {"n_sectores": k_used, "ruido_1sector": noise(tr.star_id, SAMPLE_VIEWS), "ruido_multi": noise(tr.star_id, MULTI)}
    sc = []
    for name, (pos, neg) in {"planeta vs FP": ("PLANET", "FALSE_POSITIVE"), "candidato vs sin señal": (None, "NO_SIGNAL")}.items():
        k = tr.disposition.isin([pos, neg]).to_numpy() if pos else np.ones(len(tr), bool)
        yy = (tr.disposition[k] == pos).to_numpy() if pos else (tr.disposition != neg).to_numpy()
        sc.append({"tarea": name, **{f: round(roc_auc_score(yy, -v[k] if f.startswith("ruido") else v[k]), 3) for f, v in feats.items()}})
    sc = pd.DataFrame(sc)
    sc.to_csv(f"{OUT}/tess_multi_shortcut.csv", index=False, encoding="utf-8")
    print("\natajo (ROC-AUC solo con esa variable; 0,5 = no separa):\n" + sc.to_string(index=False))


if __name__ == "__main__":
    main()
