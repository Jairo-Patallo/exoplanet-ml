"""Detector sin BLS (fase 4, research/12, decisión de Jairo 2026-11-06): CNN 1D sobre la curva SIN PLEGAR.

Tarea binaria de triaje: CANDIDATO (PLANET o FP: hay una señal real) frente a SIN_SEÑAL (NO_SIGNAL), sin efeméride.
Entrada (scripts/data/build_series.py): curva aplanada a ciegas en bloques de 30 min, hasta 9 216 bloques, normalizada
por estrella en unidades de su ruido (mediana y MAD) y comprimida con signo·log1p; canal 2 = máscara de datos.
Entrenamiento con train:
  - estrellas reales: candidatos = 1, NO_SIGNAL = 0;
  - tránsitos inyectados al vuelo en las curvas NO_SIGNAL (en cada época, una copia nueva de cada una), etiqueta 1:
    periodo log-uniforme en [0,5 d, mitad de la curva], duración física (13 h·(P/año)^(1/3)·U(0,5; 1,5), 1–15 h),
    forma trapezoidal, SNR total log-uniforme en [5, 50]. La misma estrella aparece con y sin señal: el modelo no
    puede usar la estrella como atajo.
  Parada temprana con el 15 % de las estrellas de train (reales, sin inyección), por pérdida.
Evaluación en val (estrellas reales; nunca se inyecta para elegir nada): recall de candidatos con el umbral que deja
pasar el 1 % de los SIN_SEÑAL, comparado con SDE + CNN (triage_val.py, 0,47) en las mismas estrellas; y recall de
tránsitos inyectados en las curvas SIN_SEÑAL de val por tramos de SNR, con ese mismo umbral.

Variantes: --width 2 duplica los canales; --snr-min 3 entrena también con tránsitos más débiles. Las inyecciones de
evaluación usan siempre la misma semilla (12345), así que los conjuntos de semillas se comparan con los mismos tránsitos.

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/series_cnn.py [--seed 0] [--tag base]
                 [--width 1] [--snr-min 5] [--smoke]
Salida: data/models/series_<tag>_seed<s>.pt y results/models/series_runs/<tag>_seed<s>.csv (probabilidad de val) y
        .npz (inyecciones de val: SNR y probabilidad). Conjuntos y combinaciones: scripts/checks/series_ensemble.py
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

from exoplanet.data import DATA
from exoplanet.training import DEVICE

SERIES, OUT, MODELS = DATA / "sample" / "series", "results/models", DATA / "models"
L, BIN_DAYS = 9216, 1 / 48
SNR_RANGE, LR, WD, BATCH, MAX_EPOCHS, PATIENCE = (5.0, 50.0), 1e-3, 1e-4, 64, 40, 6
SNR_BINS = [5, 7, 10, 15, 20, 30, 50]


def load(stars: pd.DataFrame) -> np.ndarray:
    """(n, L) flujo relativo con NaN donde no hay datos (también el relleno final)."""
    x = np.full((len(stars), L), np.nan, np.float32)
    for i, s in enumerate(stars.star_id):
        a = np.load(SERIES / f"{s}.npy")
        x[i, :len(a)] = a
    return x


def inject(y: np.ndarray, rng: np.random.Generator, snr_range=SNR_RANGE) -> tuple[np.ndarray, float]:
    """Inserta un tránsito periódico trapezoidal en una curva (flujo relativo); devuelve la curva y su SNR total."""
    valid = np.flatnonzero(np.isfinite(y))
    span = (valid[-1] - valid[0] + 1) * BIN_DAYS
    p = np.exp(rng.uniform(np.log(0.5), np.log(max(span / 2, 0.6))))
    dur = np.clip(13 / 24 * (p / 365) ** (1 / 3) * rng.uniform(0.5, 1.5), 1 / 24, 15 / 24)
    t = np.arange(len(y)) * BIN_DAYS
    dt = np.abs((t - rng.uniform(0, p) + p / 2) % p - p / 2)        # distancia al centro del tránsito más cercano
    ingress = dur * rng.uniform(0.1, 0.3)
    shape = np.clip((dur / 2 - dt) / ingress, 0, 1)
    sigma = 1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y)))
    n_in = max(np.nansum(np.isfinite(y) & (shape > 0.5)), 1)
    snr = np.exp(rng.uniform(*np.log(snr_range)))
    return y * (1 - snr * sigma / np.sqrt(n_in) * shape), snr


def features(y: np.ndarray) -> np.ndarray:
    """(n, 2, L): flujo en unidades de ruido (signo·log1p) y máscara de datos."""
    med = np.nanmedian(y, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(y - med), axis=1, keepdims=True) + 1e-9
    z = (y - med) / mad
    ok = np.isfinite(z)
    z = np.where(ok, np.sign(z) * np.log1p(np.abs(z)), 0.0)
    return np.stack([z, ok], axis=1).astype(np.float32)


class SeriesCNN(nn.Module):
    def __init__(self, mult: int = 1, width=(16, 32, 64, 64, 128, 128, 128)):
        super().__init__()
        width = [c * mult for c in width]
        layers, c_in = [], 2
        for c in width:                                   # 9 216 -> 72 bloques tras 7 reducciones por 2
            layers += [nn.Conv1d(c_in, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(),
                       nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c_in, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.amax(-1), h.mean(-1)], dim=1)).squeeze(1)


@torch.no_grad()
def predict(model: nn.Module, y: np.ndarray, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(y), 256):
        x = torch.from_numpy(features(y[i:i + 256])).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
            out.append(torch.sigmoid(model(x).float()).cpu().numpy())
    return np.concatenate(out)


def recall_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.01) -> tuple[float, float]:
    thr = np.quantile(s[~y], 1 - fpr)
    return float((s[y] > thr).mean()), float(thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--snr-min", type=float, default=SNR_RANGE[0])
    args = ap.parse_args()
    amp, rng = not args.no_amp, np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    q = pd.read_csv("results/sample/qc_series.csv")
    q = q[q.error.isna() & (q.n_bins >= 500)].reset_index(drop=True)
    if args.smoke:
        q = q.groupby(["split", "disposition"], group_keys=False).apply(lambda g: g.sample(min(len(g), 100), random_state=0)).reset_index(drop=True)
    tr, va = q[q.split == "train"].reset_index(drop=True), q[q.split == "val"].reset_index(drop=True)
    y_tr = (tr.disposition != "NO_SIGNAL").to_numpy()
    fit_i, stop_i = train_test_split(np.arange(len(tr)), test_size=0.15, stratify=y_tr, random_state=args.seed)
    x_tr = load(tr)
    neg = fit_i[~y_tr[fit_i]]                              # curvas SIN_SEÑAL donde se inyecta
    print(f"train {len(fit_i)} (+{len(neg)} inyectadas por época), parada {len(stop_i)}, val {len(va)}", flush=True)

    model = SeriesCNN(args.width).to(DEVICE)
    train_snr = (args.snr_min, SNR_RANGE[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state, waited = np.inf, None, 0
    for epoch in range(2 if args.smoke else MAX_EPOCHS):
        t0 = time.time()
        model.train()
        idx = np.r_[fit_i, -1 - np.arange(len(neg))]          # >= 0: estrella real; < 0: copia con tránsito inyectado
        rng.shuffle(idx)
        lab = np.r_[y_tr[fit_i], np.ones(len(neg), bool)]
        pos_w = torch.tensor((~lab).sum() / lab.sum(), device=DEVICE)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for b in np.array_split(idx, max(len(idx) // BATCH, 1)):
            ys, yb = [], []
            for i in b:
                if i >= 0:
                    ys.append(x_tr[i]), yb.append(y_tr[i])
                else:
                    ys.append(inject(x_tr[neg[-1 - i]], rng, train_snr)[0]), yb.append(True)
            xb = features(np.stack(ys))
            flip = rng.random(len(xb)) < 0.5                     # aumento: inversión temporal
            xb[flip] = xb[flip, :, ::-1]
            xb, yb = torch.from_numpy(np.ascontiguousarray(xb)).to(DEVICE), torch.tensor(yb, dtype=torch.float32, device=DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
                loss = loss_fn(model(xb).float(), yb)
            loss.backward()
            opt.step()
        p_stop = predict(model, x_tr[stop_i], amp)
        y_stop = y_tr[stop_i]
        stop_loss = -np.mean(y_stop * np.log(p_stop + 1e-7) + (~y_stop) * np.log(1 - p_stop + 1e-7))
        print(f"época {epoch}: pérdida parada {stop_loss:.4f}, AUC parada {roc_auc_score(y_stop, p_stop):.3f}, "
              f"recall@1% {recall_at_fpr(y_stop, p_stop)[0]:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if stop_loss < best:
            best, best_state, waited = stop_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    del x_tr

    x_va = load(va)
    y_va = (va.disposition != "NO_SIGNAL").to_numpy()
    p_va = predict(model, x_va, amp)
    rec, thr = recall_at_fpr(y_va, p_va)
    print(f"\nval: AUC {roc_auc_score(y_va, p_va):.3f}, recall de candidatos con 1 % de falsas alarmas {rec:.3f}", flush=True)
    for m in ["kepler", "k2", "tess"]:
        k = (va.mission == m).to_numpy()
        print(f"  {m}: AUC {roc_auc_score(y_va[k], p_va[k]):.3f}, recall@1% (umbral propio) {recall_at_fpr(y_va[k], p_va[k])[0]:.3f}")
    base = pd.read_csv(f"{OUT}/triage_val.csv")
    print("línea base SDE + CNN (todas, estrellas con vista BLS):",
          base[(base.puntuacion == "SDE + CNN (negativos difíciles)") & (base.mision == "todas")][["roc_auc", "recall_fpr1"]].to_dict("records"))

    # sensibilidad: tránsitos inyectados en las curvas SIN_SEÑAL de val, con el umbral del 1 % de falsas alarmas
    neg_va, rng_va = np.flatnonzero(~y_va), np.random.default_rng(12345)
    inj, snr = zip(*[inject(x_va[i], rng_va, (3.0, 50.0)) for i in np.repeat(neg_va, 1 if args.smoke else 5)])
    p_inj, snr = predict(model, np.stack(inj), amp), np.array(snr)
    by_snr = pd.DataFrame({"snr": pd.cut(snr, [3] + SNR_BINS), "detectado": p_inj > thr}).groupby("snr", observed=True).detectado.agg(["size", "mean"])
    print("\ntránsitos inyectados en val, detectados con el umbral del 1 %:\n" + by_snr.round(3).to_string())
    if args.smoke:
        return
    name = f"{args.tag}_seed{args.seed}"
    torch.save({"state_dict": model.state_dict(), "seed": args.seed, "width": args.width, "snr_min": args.snr_min},
               MODELS / f"series_{name}.pt")
    runs = Path(OUT) / "series_runs"
    runs.mkdir(exist_ok=True)
    va[["star_id", "mission", "disposition"]].assign(p_candidato=p_va.round(5)).to_csv(runs / f"{name}.csv", index=False, encoding="utf-8")
    np.savez_compressed(runs / f"{name}.npz", snr=snr, p=p_inj)


if __name__ == "__main__":
    main()
