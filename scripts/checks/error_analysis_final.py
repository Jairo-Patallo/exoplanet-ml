"""Análisis de errores del modelo final (fase 4, research/11-analisis-errores.md).

Predicciones: results/models/final_val_pred.csv (receta entrenada solo con train; para decidir mejoras) y
final_test_pred.csv (modelo final; solo descriptivo, para la model card). No se ajusta nada con el test.

1. Tipo de FP. Kepler: motivos del Robovetter (koi.csv). TESS: comentarios de ExoFOP (data/labels/exofop_toi.csv)
   clasificados por palabras clave en binaria (visible), contaminación por otra estrella (invisible en la curva),
   no tránsito (artefacto, ruido) y demasiado grande (radio; invisible sin la profundidad ni la estrella). Un FP puede
   tener varios motivos. K2 no publica motivo.
2. NO_SIGNAL frente a PLANET: tasas de confusión, ROC-AUC de p_PLANET entre esas dos clases y precisión de PLANET si
   hubiera r NO_SIGNAL por planeta (reponderando las tasas medidas). Con la efeméride de BLS (final_val_bls_pred.csv)
   los NO_SIGNAL son negativos difíciles: picos de ruido con forma de tránsito.

Uso:  PYTHONPATH=src .venv/Scripts/python scripts/checks/error_analysis_final.py
Si existe hard_val*_pred.csv (scripts/models/hard_negatives.py), compara también ese modelo.
Salida: results/models/error_fp_tipo.csv y error_nosignal.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from exoplanet.data import DATA
from exoplanet.labels import nearest_period

OUT = "results/models"
TESS_TYPES = {
    "binaria": r"\bs?eb\d?\b|binary|v-shaped|v shaped|(?<!no )secondary|odd[- /]even|\bsb\d?\b",
    "contaminación": r"\bneb\b|centroid|offset|aperture correlation|crowd|another star|stars in (?:the )?pixel|\bnpc\b|true source|correct source|nearby|blend|event is on|target is on|phantom|real stars",
    "demasiado grande": r"too (?:big|large)|large for",
    "no tránsito": r"artifact|systematic|asteroid|scattered light|momentum dump|variability|noise|low snr",
}
KEPLER_TYPES = {"binaria": "koi_fpflag_ss", "contaminación": ["koi_fpflag_co", "koi_fpflag_ec"], "no tránsito": "koi_fpflag_nt"}
RATIOS = (1, 10, 100)


def fp_types(fp: pd.DataFrame) -> pd.DataFrame:
    """Columnas booleanas por motivo para los FP de Kepler y TESS (NaN si no se encontró el candidato)."""
    out = pd.DataFrame(index=fp.index, columns=list(TESS_TYPES), dtype=object)
    kep = fp.mission == "kepler"
    koi = pd.read_csv(DATA / "labels" / "koi.csv")
    kepid = fp.star_id[kep].str.removeprefix("KIC_").astype(int)
    flags = {f: nearest_period(koi, "kepid", kepid, fp.period_days[kep], f, "koi_period")
             for f in ["koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_fpflag_nt"]}
    for t, f in KEPLER_TYPES.items():
        out.loc[kep, t] = pd.concat([flags[c] for c in np.atleast_1d(f)], axis=1).eq(1).any(axis=1)
    tess = fp.mission == "tess"
    toi = pd.read_csv(DATA / "labels" / "exofop_toi.csv").rename(columns={"TIC ID": "tid", "Period (days)": "per"})
    tid = fp.star_id[tess].str.removeprefix("TIC_").astype(int)
    com = nearest_period(toi, "tid", tid, fp.period_days[tess], "Comments", "per").fillna("").str.lower()
    for t, rx in TESS_TYPES.items():
        out.loc[tess, t] = com.str.contains(rx, regex=True)
    return out


def by_type(pred: pd.DataFrame) -> pd.DataFrame:
    fp = pred[pred.disposition == "FALSE_POSITIVE"].copy()
    ty = fp_types(fp)
    groups = {**{t: ty[t] == True for t in ty},
              "solo contaminación (invisible)": (ty["contaminación"] == True) & (ty["binaria"] != True) & (ty["no tránsito"] != True),
              "algún binaria/no tránsito (visible)": (ty["binaria"] == True) | (ty["no tránsito"] == True),
              "sin motivo reconocido": ty.ne(True).all(axis=1), "todos": pd.Series(True, index=fp.index)}
    rows = []
    for mission, g in fp.groupby("mission"):
        for lab, m in groups.items():
            p = g.loc[m[g.index], "pred"]
            if len(p):
                rows.append({"mision": mission, "motivo": lab, "n": len(p), "acierta_FP": (p == "FALSE_POSITIVE").mean(),
                             "como_PLANET": (p == "PLANET").mean(), "como_NO_SIGNAL": (p == "NO_SIGNAL").mean()})
    return pd.DataFrame(rows)


def planet_vs_nosignal(pred: pd.DataFrame, views: str) -> list[dict]:
    rows = []
    for mission, g in pred.groupby("mission"):
        pl, ns = g[g.disposition == "PLANET"], g[g.disposition == "NO_SIGNAL"]
        tpr, f_ns = (pl.pred == "PLANET").mean(), (ns.pred == "PLANET").mean()
        both = pd.concat([pl, ns])
        r = {"mision": mission, "vistas": views, "n_planet": len(pl), "n_nosignal": len(ns),
             "planet_detectado": tpr, "planet_como_NO_SIGNAL": (pl.pred == "NO_SIGNAL").mean(),
             "nosignal_como_PLANET": f_ns, "nosignal_como_FP": (ns.pred == "FALSE_POSITIVE").mean(),
             "roc_auc_planet_vs_ns": roc_auc_score(both.disposition == "PLANET", both.p_PLANET) if len(pl) and len(ns) else np.nan}
        r.update({f"precision_planet_r{k}": tpr / (tpr + k * f_ns) if tpr + f_ns > 0 else np.nan for k in RATIOS})
        rows.append(r)
    return rows


def load(name: str) -> pd.DataFrame:
    p = pd.read_csv(f"{OUT}/{name}")
    p["pred"] = p[["p_PLANET", "p_FALSE_POSITIVE", "p_NO_SIGNAL"]].to_numpy().argmax(1)
    p["pred"] = np.array(["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"])[p.pred]
    qc = pd.read_csv("results/sample/qc.csv", usecols=["star_id", "period_days"]).drop_duplicates("star_id")
    return p.merge(qc, on="star_id", how="left")


def main() -> None:
    pd.set_option("display.width", 200)
    runs = [("final", "val", "final_val_pred.csv"), ("final", "test", "final_test_pred.csv")]
    models = ["final"] + (["hard"] if Path(f"{OUT}/hard_val_pred.csv").exists() else [])   # hard: con negativos difíciles
    runs += [("hard", "val", "hard_val_pred.csv")] if "hard" in models else []
    types = pd.concat([by_type(load(f)).assign(modelo=m, conjunto=s) for m, s, f in runs])
    types.round(3).to_csv(f"{OUT}/error_fp_tipo.csv", index=False, encoding="utf-8")
    print(types.round(3).to_string(index=False))

    rows = []
    for m in models:
        cat, bls = load(f"{m}_val_pred.csv"), load(f"{m}_val_bls_pred.csv")
        rows += [dict(r, modelo=m) for r in planet_vs_nosignal(cat, "catálogo") + planet_vs_nosignal(bls, "BLS (negativos difíciles)")]
    ns = pd.DataFrame(rows)
    ns.round(3).to_csv(f"{OUT}/error_nosignal.csv", index=False, encoding="utf-8")
    print("\n" + ns.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
