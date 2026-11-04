"""Variables candidatas para mejorar el modelo: correlación entre ellas y capacidad de separar clases (2026-11-04).

Solo train. Candidatas calculables con la curva o la efeméride (portables); los metadatos del proceso (n_transits,
n_points...) quedan fuera porque delatan cómo se fabricaron los datos. Dos paneles:
  1. Correlación de Spearman entre candidatas (¿son redundantes?), en una sola misión para no mezclar misiones.
  2. Separación: AUC de cada variable para FP vs PLANET y para NO_SIGNAL vs positivos, calculada dentro de cada misión
     y promediada (sin efecto Simpson). Se muestra |AUC - 0,5| · 2: 0 = no separa, 1 = separa perfectamente.

Uso:  python scripts/checks/feature_candidates.py   (PYTHONPATH=src)
Salida: results/eda/feature_candidates.png y feature_candidates.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

from exoplanet.data import MISSIONS
from exoplanet.features import LOC_OUT, curve_features, robust_sigma
from exoplanet.views import load_centroid, load_sample, load_views

OUT = "results/eda/feature_candidates"
BLUE, ORANGE, GRAY, RED, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#f0efec", "#e34948", "#1f1f1e", "#6b6b66", "#fcfcfb"
LABELS = {
    "depth": "Profundidad (log)", "snr": "Señal/ruido (log)", "ingress_frac": "Forma V/U", "odd_even_z": "Par/impar",
    "secondary_z": "Eclipse secundario", "sec_ratio": "Secundario / primario", "asymmetry": "Asimetría del tránsito",
    "duty": "Duración / periodo", "log_period": "Periodo (log)", "noise": "Ruido fuera del tránsito",
    "centroid": "Centroide en el tránsito",
}


def candidates(stars: pd.DataFrame) -> pd.DataFrame:
    v = load_views(stars.star_id, ("local", "odd", "even", "glob"))
    duty = (stars.duration_hrs / 24 / stars.period_days).to_numpy()
    f = curve_features(v["local"], v["odd"], v["even"], v["glob"], duty=duty, mask_no_dip=False)
    sl = lambda x: np.sign(x) * np.log10(1 + np.abs(x))
    depth = 1 - np.median(v["local"][:, 39:42], axis=1)
    sec_depth = np.median(v["glob"][:, 20:180], axis=1) - np.median(v["glob"][:, np.r_[0:6, 195:201]], axis=1)
    left, right = v["local"][:, 35:40].mean(axis=1), v["local"][:, 41:46].mean(axis=1)
    return pd.DataFrame({
        "depth": sl(f.depth_local_ppm), "snr": sl(f.snr_local), "ingress_frac": f.ingress_frac,
        "odd_even_z": sl(f.odd_even_z), "secondary_z": sl(f.secondary_z),
        "sec_ratio": np.clip(sec_depth / np.where(depth > 0, depth, np.nan), -2, 2),
        "asymmetry": (left - right) / np.where(depth > 0, depth, np.nan),
        "duty": np.log10(duty), "log_period": np.log10(stars.period_days.to_numpy()),
        "noise": np.log10(robust_sigma(v["local"][:, LOC_OUT]) * 1e6 + 1e-3),
        "centroid": load_centroid(stars.star_id)["local"][:, 40],
    })


def separation(x: pd.Series, y: np.ndarray, mission: pd.Series) -> float:
    aucs = []
    for m in MISSIONS:
        ok = (mission == m).to_numpy() & x.notna().to_numpy()
        if len(set(y[ok])) == 2:
            aucs.append(abs(roc_auc_score(y[ok], x[ok]) - 0.5) * 2)
    return float(np.mean(aucs))


def main() -> None:
    stars = load_sample(["train"])
    X = candidates(stars)
    d = stars.disposition
    pos = d.isin(["PLANET", "FALSE_POSITIVE"]).to_numpy()
    sep = pd.DataFrame({
        "FP vs PLANET": {c: separation(X.loc[pos, c], (d[pos] == "FALSE_POSITIVE").to_numpy(), stars.mission[pos]) for c in X},
        "NO_SIGNAL vs positivos": {c: separation(X[c], (d == "NO_SIGNAL").to_numpy(), stars.mission) for c in X},
    })
    corr = X[stars.mission == "kepler"].corr(method="spearman")
    sep.round(3).to_csv(f"{OUT}.csv", encoding="utf-8")
    print(sep.round(3).sort_values("FP vs PLANET", ascending=False).to_string())

    order = sep.sort_values("FP vs PLANET", ascending=False).index.tolist()
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1]}, facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, GRAY, RED])
    c = corr.loc[order, order].to_numpy()
    im = a1.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    names = [LABELS[k] for k in order]
    a1.set_xticks(range(len(order)), names, rotation=45, ha="right")
    a1.set_yticks(range(len(order)), names)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and abs(c[i, j]) >= 0.5:
                a1.text(j, i, f"{c[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK)
    a1.set_title("Correlación entre candidatas (Spearman, train de Kepler)", loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("−1 opuestas · 0 independientes · +1 redundantes", color=MUTED)

    y = np.arange(len(order))
    h = 0.38
    a2.barh(y - h / 2, sep.loc[order, "FP vs PLANET"], h - 0.04, color=BLUE, label="FP vs PLANET")
    a2.barh(y + h / 2, sep.loc[order, "NO_SIGNAL vs positivos"], h - 0.04, color=ORANGE, label="NO_SIGNAL vs positivos")
    for yi, k in zip(y, order):
        a2.text(sep.loc[k, "FP vs PLANET"] + 0.01, yi - h / 2, f"{sep.loc[k, 'FP vs PLANET']:.2f}", va="center", fontsize=8, color=MUTED)
    a2.set_yticks(y, names)
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.set_xlabel("Separación |AUC − 0,5| · 2  (0 = no separa · 1 = perfecta), media de misiones")
    a2.set_title("¿Cuánto separa cada variable las clases?", loc="left", fontsize=11, color=INK)
    a2.grid(axis="x", color="#e4e3df", linewidth=0.6)
    a2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        a2.spines[s].set_visible(False)
    a2.legend(loc="lower right", frameon=False)
    a1.set_facecolor(SURFACE), a2.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=140, facecolor=SURFACE)


if __name__ == "__main__":
    main()
