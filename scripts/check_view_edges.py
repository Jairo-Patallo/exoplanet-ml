"""Comprobación de los bordes de las vistas propias (tarea #10).

En el dataset de HF los bins 0-4 y 76-80 de la vista local tenían artefactos. Aquí se mide si las vistas de
exoplanet.folding los heredan: por estrella, desviación de los bins de borde frente al ruido de las alas (bins 5-29 y
51-75 de la local; 0-4/196-200 frente a 5-79/121-195 en la global), en unidades de ese ruido (MAD).

Uso:  python scripts/check_view_edges.py   (PYTHONPATH=src)
Salida: results/sample/view_edges.csv (por misión y clase) y results/sample/view_edges.png (vista local media).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanet.data import DATA

VIEWS = DATA / "sample" / "views"
OUT = "results/sample/view_edges"
EDGES = {"local": (np.r_[0:5, 76:81], np.r_[5:30, 51:76]), "glob": (np.r_[0:5, 196:201], np.r_[5:80, 121:196])}


def edge_z(view: np.ndarray, edge: np.ndarray, wing: np.ndarray) -> float:
    """Desviación mediana de los bins de borde respecto a la mediana de las alas, en MAD de las alas."""
    w = view[wing]
    mad = 1.4826 * np.median(np.abs(w - np.median(w))) or 1e-9
    return float(np.median(view[edge] - np.median(w)) / mad)


def main() -> None:
    qc = pd.read_csv("results/sample/qc.csv")
    qc = qc[qc["valid"].fillna(False).astype(bool)]
    rows, means = [], {}
    for r in qc.itertuples():
        v = np.load(VIEWS / f"{r.star_id}.npz")
        rows.append({"mission": r.mission, "disposition": r.disposition,
                     **{f"z_{k}": edge_z(v[k], *EDGES[k]) for k in EDGES}})
        means.setdefault((r.mission, r.disposition), []).append(v["local"])
    z = pd.DataFrame(rows)
    summary = z.groupby(["mission", "disposition"]).agg(
        n=("z_local", "size"),
        z_local_median=("z_local", "median"), pct_local_abs_z_gt3=("z_local", lambda s: 100 * (s.abs() > 3).mean()),
        z_glob_median=("z_glob", "median"), pct_glob_abs_z_gt3=("z_glob", lambda s: 100 * (s.abs() > 3).mean()),
    ).round(2).reset_index()
    summary.to_csv(f"{OUT}.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, mission in zip(axes, ["kepler", "k2", "tess"]):
        for (m, cls), vs in sorted(means.items()):
            if m == mission:
                ax.plot(np.median(vs, axis=0), label=f"{cls} (n={len(vs)})")
        for b in (4.5, 75.5):
            ax.axvline(b, color="grey", ls=":", lw=1)
        ax.set(title=f"{mission}: vista local mediana", xlabel="bin")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=120)


if __name__ == "__main__":
    main()
