"""Análisis de errores por tipo de falso positivo en Kepler (2026-11-04).

El Robovetter marca cada FP de Kepler con uno o varios motivos (tabla KOI cumulative, scripts/data/fetch_labels.py):
  nt  no parece un tránsito          -> visible en la curva
  ss  eclipse estelar (binaria)      -> visible en la curva (secundario, par/impar, forma en V)
  co  centroide desplazado           -> NO visible en la curva de la estrella (sí con el centroide)
  ec  coincidencia de efemérides     -> NO visible: la señal es de otra estrella y la curva es idéntica a un planeta
Se cargan las CNN guardadas (versión conjunta, semilla 0; solo curva y curva + centroide), se predice en val estrella a
estrella y se mide, por motivo, qué fracción de FP acierta cada modelo y con qué los confunde. Cada FP se cruza con su KOI
por kepid y periodo más cercano.

Uso (WSL):  PYTHONPATH=src ~/venvs/exoplanet/bin/python scripts/checks/error_analysis_fp.py
Salida: results/models/error_analysis_fp_kepler.csv (por motivo y modelo) y error_analysis_val_predictions.csv.
"""
import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.evaluation import evaluate_by_mission
from exoplanet.labels import nearest_period
from exoplanet.training import inputs, predict
from exoplanet.views import load_sample

MODELS = {"solo_curva": ("cnn_gpu_conjunta_todas.pt", False), "curva_centroide": ("cnn_centroid_gpu_conjunta_todas.pt", True)}
FLAGS = {"koi_fpflag_nt": "nt (no tránsito)", "koi_fpflag_ss": "ss (binaria)", "koi_fpflag_co": "co (centroide)",
         "koi_fpflag_ec": "ec (efemérides)"}
OUT = "results/models/error_analysis_fp_kepler.csv"


def load_model(name: str, channels: int) -> AstroCNN:
    ckpt = torch.load(DATA / "models" / name, map_location="cpu")
    model = AstroCNN(ckpt["config"]["base"], ckpt["config"]["dropout"], torch.zeros(1), torch.ones(1), channels)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main() -> None:
    val = load_sample(["val"])
    preds = val[["star_id", "mission", "disposition", "period_days"]].copy()
    for name, (file, centroid) in MODELS.items():
        p = predict(load_model(file, 2 if centroid else 1), inputs(ViewDataset(val, centroid=centroid)))
        preds[name] = np.asarray(CLASSES)[p.argmax(1)]
        media = evaluate_by_mission(val.mission.to_numpy(), val.disposition.to_numpy(), p, n_boot=50)[0]
        print(f"{name}: macro-F1 medio recalculado {media.set_index('mision').loc['media', 'macro_f1']:.3f}")

    koi = pd.read_csv(DATA / "labels" / "koi.csv")
    kep = preds[preds.mission == "kepler"].copy()
    kepid = kep.star_id.str.removeprefix("KIC_").astype(int)
    for f in FLAGS:
        kep[f] = nearest_period(koi, "kepid", kepid, kep.period_days, f, "koi_period")
    preds.merge(kep[["star_id", *FLAGS]], on="star_id", how="left").to_csv(
        "results/models/error_analysis_val_predictions.csv", index=False, encoding="utf-8")

    fp = kep[kep.disposition == "FALSE_POSITIVE"]
    print(f"\nFP de Kepler en val: {len(fp)}; sin KOI encontrado: {fp[list(FLAGS)].isna().all(axis=1).sum()}")
    only_invisible = (fp.koi_fpflag_nt != 1) & (fp.koi_fpflag_ss != 1) & ((fp.koi_fpflag_co == 1) | (fp.koi_fpflag_ec == 1))
    groups = {**{lab: fp[f] == 1 for f, lab in FLAGS.items()},
              "solo co/ec (invisibles en la curva)": only_invisible,
              "algún nt/ss (visibles)": (fp.koi_fpflag_nt == 1) | (fp.koi_fpflag_ss == 1), "todos": fp.star_id.notna()}
    rows = []
    for lab, m in groups.items():
        for name in MODELS:
            pr = fp.loc[m, name]
            rows.append({"motivo": lab, "modelo": name, "n": int(m.sum()), "acierta_FP": round((pr == "FALSE_POSITIVE").mean(), 3),
                         "como_PLANET": round((pr == "PLANET").mean(), 3), "como_NO_SIGNAL": round((pr == "NO_SIGNAL").mean(), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
