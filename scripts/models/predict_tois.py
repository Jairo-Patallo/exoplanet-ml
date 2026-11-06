"""Predicciones congeladas v1 en los TOIs abiertos (fase 7, #87; decisión 13 de CONTEXTO.md, 2026-11-06).

Modelo: el final de la fase 4 (data/models/final_seed{0..4}.pt, sin profundidad, media de las 5 semillas) con la
temperatura de TESS de data/models/final_temperature.json (p_T = softmax(log p / T), scripts/models/temperature.py).
Entrada: las vistas de scripts/data/build_toi_views.py (solo las válidas). Además de las 3 probabilidades se da
P(planeta | señal) = p_PLANET / (p_PLANET + p_FALSE_POSITIVE), la pregunta que importa en un TOI (auditoría A7).
El manifiesto guarda el SHA-256 de todo lo que fija la predicción, para demostrar después qué se predijo y cuándo.

Uso (WSL):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/predict_tois.py [--version v1]
Salida: results/toi/predictions_<version>.csv y results/toi/manifest_<version>.json
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet.cnn import AstroCNN
from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.training import DEVICE, inputs, predict

MODELS, OUT = DATA / "models", Path("results/toi")
SEEDS = (0, 1, 2, 3, 4)
QC, EXOFOP, VIEWS = OUT / "qc_toi.csv", Path("data/labels/exofop_toi.csv"), DATA / "toi" / "views"
EXOMINER = DATA / "external" / "exominerpp" / "exominerpp_unk_s1-s67.csv"   # Zenodo 10.5281/zenodo.15466293 (CC-BY-4.0)
TFOP_COLS = ["Master", "SG1A", "SG1B", "SG2"]                                # prioridades del TFOP (1 = la más alta)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def references(out: pd.DataFrame) -> pd.DataFrame:
    """Añade la puntuación de ExoMiner++ (TCE de la misma TIC con periodo a ±1 %; si hay varias ejecuciones, la de más
    tránsitos) y las prioridades del TFOP de ExoFOP, para congelarlas junto a las nuestras."""
    em = pd.read_csv(EXOMINER, usecols=["TIC ID", "Sector Run", "ExoMiner Score", "Orbital Period [day]",
                                        "Number of transits observed"])
    em["star_id"] = "TIC_" + em["TIC ID"].astype(str)
    m = out[["toi", "star_id", "period_days"]].merge(em, on="star_id")
    m = m[(m["Orbital Period [day]"] / m.period_days - 1).abs() < 0.01]
    m = m.sort_values("Number of transits observed").drop_duplicates("toi", keep="last")
    out = out.merge(m[["toi", "ExoMiner Score", "Sector Run"]].rename(
        columns={"ExoMiner Score": "exominerpp_score", "Sector Run": "exominerpp_sector_run"}), on="toi", how="left")
    e = pd.read_csv(EXOFOP, comment="#", low_memory=False)
    e["toi"] = e["TOI"].map(lambda x: f"{x:.2f}")
    return out.merge(e[["toi"] + TFOP_COLS].rename(columns={c: f"tfop_{c}" for c in TFOP_COLS}), on="toi", how="left")


def temper(p: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-7, 1)) / T
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    qc = pd.read_csv(QC, dtype={"toi": str})
    stars = qc[qc["valid"].eq(True)].reset_index(drop=True)
    ds = ViewDataset(stars.assign(star_id=stars.toi, disposition="PLANET"), views_dir=VIEWS)   # clase: relleno, no se usa
    x = {**inputs(ds), "log_depth": torch.zeros_like(ds.log_depth)}
    probas, weights = [], {}
    for s in SEEDS:
        path = MODELS / f"final_seed{s}.pt"
        ck = torch.load(path, map_location="cpu")
        m = AstroCNN(ck["config"]["base"], ck["config"]["dropout"], torch.zeros(1), torch.ones(1))
        m.load_state_dict(ck["state_dict"])
        probas.append(predict(m.to(DEVICE).eval(), x, amp=True))
        weights[path.name] = sha256(path)
    raw = np.mean(probas, axis=0)
    temp_file = MODELS / "final_temperature.json"
    T = json.loads(temp_file.read_text())["temperatura_por_mision"]["tess"]
    p = temper(raw, T)
    out = stars[["toi", "star_id", "period_days", "epoch_bjd", "duration_hrs", "tmag", "depth_ppm", "n_transits"]].assign(
        **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)},
        p_planet_given_signal=p[:, 0] / (p[:, 0] + p[:, 1]),
        pred=np.asarray(CLASSES)[p.argmax(axis=1)])
    out = references(out)
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / f"predictions_{args.version}.csv"
    out.round(5).to_csv(pred_path, index=False, encoding="utf-8")
    views = hashlib.sha256()
    for t in sorted(stars.toi):
        views.update((VIEWS / f"{t}.npz").read_bytes())
    # core.autocrlf=true: desde WSL, el fin de línea CRLF del checkout de Windows no cuenta como cambio
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-c", "core.autocrlf=true", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    manifest = {"version": args.version, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "git_commit": commit, "git_dirty": dirty, "n_tois": len(out),
                "model": "CNN dos ramas solo curva, sin profundidad, media de 5 semillas (research/10)",
                "temperature_tess": T, "sha256": {"weights": weights, "temperature": sha256(temp_file),
                                                    "views_concat": views.hexdigest(), "qc": sha256(QC),
                                                    "exofop": sha256(EXOFOP), "exominerpp": sha256(EXOMINER), "predictions": sha256(pred_path)},
                "pred_counts": out.pred.value_counts().to_dict(),
                "with_exominerpp": int(out.exominerpp_score.notna().sum())}
    (OUT / f"manifest_{args.version}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_tois", "pred_counts", "with_exominerpp", "git_commit", "git_dirty")}, ensure_ascii=False))
    print(out.p_planet_given_signal.describe().round(3).to_string())


if __name__ == "__main__":
    main()
