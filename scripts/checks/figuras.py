"""Figuras de la documentación del proyecto (fases 1–7), a partir de los CSV de results/ (no entrena nada).

Colores fijos: misión (Kepler azul, K2 naranja, TESS verde agua) y clase (PLANET violeta, FP ámbar, NO_SIGNAL gris).
Los resultados de val vienen de versiones de datos algo distintas (research/06, nota de versión); las comparaciones finas
deben hacerse dentro de la misma tabla.

Uso:  PYTHONPATH=src python scripts/checks/figuras.py
Salida: docs/figuras/*.png
"""
import ast
import re
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exoplanet.data import CLASSES
from exoplanet.evaluation import point_metrics
from exoplanet.views import load_views, normalize

R, OUT = Path("results"), Path("docs/figuras")
MIS = ["kepler", "k2", "tess"]
MIS_LABEL = {"kepler": "Kepler", "k2": "K2", "tess": "TESS", "media": "Media"}
MIS_COLOR = {"kepler": "#2a78d6", "k2": "#eb6834", "tess": "#1baf7a"}
CLS_LABEL = {"PLANET": "Planeta", "FALSE_POSITIVE": "Falso positivo", "NO_SIGNAL": "Sin señal"}
CLS_COLOR = {"PLANET": "#6250d6", "FALSE_POSITIVE": "#eda100", "NO_SIGNAL": "#8a8986"}
INK, MUTED, GRID = "#1b1b1a", "#52514e", "#e6e4de"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9.5, "axes.edgecolor": "#b9b7b0", "axes.labelcolor": INK,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": "white"})
INDEX: list[tuple[str, str]] = []


def save(fig, name: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    INDEX.append((name, caption))
    print("ok", name, flush=True)


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / name)


def bar_labels(ax, bars, fmt="{:.2f}", inside=False):
    for b in bars:
        w = b.get_height()
        ax.annotate(fmt.format(w), (b.get_x() + b.get_width() / 2, w), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points", color=INK)


def legend_below(ax, ncol=3, y=-0.14):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def top_labels(ax, bars, tops):
    for b, t in zip(bars, tops):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, t), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points", color=INK)


def mission_legend(ax, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MIS_COLOR[m], label=MIS_LABEL[m]) for m in MIS], loc=loc, ncol=3)


# ----------------------------------------------------------------------------------------------------- fase 1: datos
def f_muestra():
    q = csv("sample/qc.csv")
    q = q[q["valid"].eq(True)]
    t = q.groupby(["mission", "disposition"]).size().unstack().reindex(MIS)[list(CLASSES)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x, w = np.arange(3), 0.26
    for i, c in enumerate(CLASSES):
        bars = ax.bar(x + (i - 1) * (w + 0.01), t[c], w, color=CLS_COLOR[c], label=CLS_LABEL[c])
        bar_labels(ax, bars, "{:.0f}")
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.set_ylabel("Estrellas con vista válida")
    ax.set_title("Muestra completa: estrellas por misión y clase")
    ax.legend(loc="upper right")
    save(fig, "f01_muestra", f"Estrellas con vista válida por misión y clase ({len(q):,} en total). NO_SIGNAL se fabricó 1:1 con los positivos.")


def f_galeria():
    q = csv("sample/qc.csv")
    q = q[q["valid"].eq(True) & (q.split == "train")]
    rng = np.random.default_rng(3)
    for key, n_bins, title, fname, cap in (
            ("local", 81, "Vista local (±4 duraciones)", "f02_galeria_local", "Vista local de una estrella al azar por misión y clase (normalizada: la caída vale −1)."),
            ("glob", 201, "Vista global (órbita completa)", "f03_galeria_global", "Vista global de las mismas estrellas: fase −0,5 a 0,5, tránsito en el centro.")):
        fig, axes = plt.subplots(3, 3, figsize=(8.5, 6), sharex=True, sharey=True)
        for i, c in enumerate(CLASSES):
            for j, m in enumerate(MIS):
                s = q[(q.mission == m) & (q.disposition == c)].sample(1, random_state=int(rng.integers(1e6)))
                v = load_views(s.star_id, ("local", "glob"))
                loc, glo, _ = normalize(v["local"], v["glob"])
                y = (loc if key == "local" else glo)[0]
                ax = axes[i, j]
                ax.plot(np.linspace(-4, 4, n_bins) if key == "local" else np.linspace(-0.5, 0.5, n_bins), y,
                        color=CLS_COLOR[c], lw=1.4)
                ax.set_ylim(-1.9, 1.6)
                if i == 0:
                    ax.set_title(MIS_LABEL[m], loc="center")
                if j == 0:
                    ax.set_ylabel(CLS_LABEL[c])
                ax.text(0.98, 0.04, s.star_id.iloc[0], transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
        fig.supxlabel("Duraciones desde el centro del tránsito" if key == "local" else "Fase orbital")
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
        save(fig, fname, cap)


def f_atajos():
    s = csv("sample/shortcut_negatives_views.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x, w = np.arange(3), 0.36
    for i, c in enumerate(["PLANET", "FALSE_POSITIVE"]):
        v = s[s.clase == c].set_index("mission").reindex(MIS)["auc_cv_vista_sin_transito"]
        bar_labels(ax, ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=CLS_COLOR[c], label=f"{CLS_LABEL[c]} frente a sin señal"))
    ax.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax.text(2.45, 0.51, "azar", fontsize=8, color=MUTED)
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC con el tránsito tapado")
    ax.set_title("Test de atajo: ¿se distingue la clase sin mirar el tránsito?")
    ax.legend(loc="upper left")
    save(fig, "f04_atajos", "Boosting sobre la vista global con el 20 % central tapado. Planeta ≈ 0,55–0,60 (sesgo de detección aceptado); FP de K2 0,84 es física (secundarios, variabilidad).")


def f_fuga_mision():
    s = csv("leak/mission_leak_own_views.csv")
    s["auc"] = s.auc_ovr_macro.str.extract(r"^([\d.]+)").astype(float)
    s[["lo", "hi"]] = s.auc_ovr_macro.str.extract(r"\[([\d.]+),([\d.]+)\]").astype(float)
    s["label"] = s.subconjunto + " · " + s.entradas.str.replace("variables derivadas de la curva", "variables").str.replace(" (local+global)", "")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(s) + 1))
    y = np.arange(len(s))[::-1]
    ax.errorbar(s.auc, y, xerr=[s.auc - s.lo, s.hi - s.auc], fmt="o", color="#2a78d6", ms=6, capsize=3, lw=1)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y, s.label)
    ax.set_xlim(0.45, 1)
    ax.set_xlabel("ROC-AUC al adivinar la misión (0,5 = imposible)")
    ax.set_title("La misión se adivina desde la curva (fuga de misión)")
    save(fig, "f05_fuga_mision", "Clasificador que intenta adivinar la misión con las vistas propias (IC 95 %). Motivó la ablación de la profundidad (#43) y la evaluación LOMO.")


# --------------------------------------------------------------------------------------------- fase 2: modelos
MODELS = [  # (etiqueta, archivo, columna de filtro, valor)
    ("Dummy", "dummy_val.csv", None, None),
    ("Logística · físicas", "logreg_val.csv", "variante", "A_fisicas"),
    ("Logística · vistas", "logreg_val.csv", "variante", "B_vistas"),
    ("kNN · físicas", "knn_val.csv", "variante", "A_fisicas"),
    ("PCA + kNN · vistas", "knn_pca_val.csv", "variante", "B_vistas"),
    ("Boosting · físicas", "hgb_val.csv", "variante", "A_fisicas"),
    ("Boosting · vistas", "hgb_val.csv", "variante", "B_vistas"),
    ("MLP", "mlp_val.csv", None, None),
    ("MOMENT-small congelado", "moment_small_val.csv", None, None),
    ("MOMENT-large congelado", "moment_large_val.csv", None, None),
    ("MOMENT ajustado", "moment_ft_val.csv", None, None),
    ("CNN por misión", "cnn_val.csv", "entrenamiento", "por_mision"),
    ("CNN conjunta", "cnn_val.csv", "entrenamiento", "conjunta"),
    ("CNN conjunta · conjunto de 3", "cnn_gpu_val.csv", "entrenamiento", "conjunta_ensemble"),
    ("CNN + centroide", "cnn_centroid_gpu_val.csv", "entrenamiento", "conjunta"),
]


def model_rows(fname, col, val):
    d = csv(f"models/{fname}")
    if col:
        d = d[d[col] == val]
    return d.set_index("mision")


def f_modelos():
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y = np.arange(len(MODELS))[::-1]
    for yi, (lab, f, c, v) in zip(y, MODELS):
        d = model_rows(f, c, v)
        vals = [d.loc[m, "macro_f1"] for m in MIS]
        ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, lw=3, zorder=1)
        for m, val in zip(MIS, vals):
            ax.scatter(val, yi + {"kepler": 0.12, "k2": 0, "tess": -0.12}[m], color=MIS_COLOR[m], s=36, zorder=3,
                       edgecolor="white", lw=0.8)
        mean = d.loc["media", "macro_f1"] if "media" in d.index else np.mean(vals)
        ax.scatter(mean, yi, marker="|", color=INK, s=180, zorder=4)
        ax.annotate(f"{mean:.2f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [m[0] for m in MODELS])
    ax.set_xlim(0.15, 0.95)
    ax.set_xlabel("Macro-F1 en val (3 clases)")
    ax.set_title("Fase 2: todos los modelos dentro de cada misión")
    mission_legend(ax, "lower left")
    ax.text(1.0, 1.01, "media", transform=ax.transAxes, fontsize=8, color=MUTED, ha="left")
    save(fig, "f06_comparacion_modelos", "Macro-F1 en val por misión (puntos) y media (raya). Seis familias se quedan en 0,78–0,85: el límite está en la información de la curva, no en el modelo.")


def f_prauc_clases():
    rows = []
    for lab, f, c, v in MODELS[1:]:
        d = model_rows(f, c, v)
        rows.append([d.loc[m, [f"pr_auc_{k}" for k in CLASSES]].to_numpy(float) for m in MIS])
    a = np.array(rows).mean(axis=1)           # media de misiones
    fig, ax = plt.subplots(figsize=(6.2, 6))
    im = ax.imshow(a, cmap="Blues", vmin=0.5, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(a.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if a[i, j] > 0.82 else INK)
    ax.set_xticks(range(3), [CLS_LABEL[c] for c in CLASSES])
    ax.set_yticks(range(len(MODELS) - 1), [m[0] for m in MODELS[1:]])
    ax.set_title("PR-AUC por clase (media de misiones)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "f07_prauc_por_clase", "PR-AUC one-vs-rest por clase, media de las tres misiones, en val. Sin señal es fácil; planeta frente a FP es lo difícil.")


CONF = [  # (etiqueta, archivo, columna de filtro, valor, nombre)
    ("Logística · vistas", "logreg_val_confusion.csv", "variante", "B_vistas", "logistica"),
    ("PCA + kNN · vistas", "knn_pca_val_confusion.csv", "variante", "B_vistas", "knn_pca"),
    ("Boosting · vistas", "hgb_val_confusion.csv", "variante", "B_vistas", "boosting"),
    ("MLP", "mlp_val_confusion.csv", None, None, "mlp"),
    ("MOMENT ajustado", "moment_ft_val_confusion.csv", None, None, "moment"),
    ("CNN por misión", "cnn_val_confusion.csv", "entrenamiento", "por_mision", "cnn"),
    ("CNN + centroide", "cnn_centroid_gpu_val_confusion.csv", "entrenamiento", "conjunta", "cnn_centroide"),
]


def confusion_panel(axes, getter, title, suptitle_fig):
    for ax, m in zip(axes, MIS):
        c = getter(m)
        pct = c / c.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Purples", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{c[i, j]:.0f}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if pct[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["Pl.", "FP", "S/S"])
        ax.set_yticks(range(3), ["Pl.", "FP", "S/S"])
        ax.set_title(MIS_LABEL[m], loc="center")
        ax.set_xlabel("Predicha")
    axes[0].set_ylabel("Real")
    suptitle_fig.suptitle(title, x=0.01, ha="left", fontweight="bold")


def f_confusiones():
    for lab, f, c, v, name in CONF:
        d = csv(f"models/{f}")
        if c:
            d = d[d[c] == v]
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 3))
        getter = lambda m: d[d.mision == m].set_index("real").loc[[f"real_{k}" for k in CLASSES], [f"pred_{k}" for k in CLASSES]].to_numpy(float)
        confusion_panel(axes, getter, f"Matriz de confusión en val: {lab}", fig)
        save(fig, f"f08_confusion_{name}", f"{lab}: filas = clase real, columnas = predicha (Pl. planeta, FP falso positivo, S/S sin señal); recuento y % por fila.")


def f_centroide():
    base = model_rows("cnn_gpu_val.csv", "entrenamiento", "conjunta")
    cen = model_rows("cnn_centroid_gpu_val.csv", "entrenamiento", "conjunta")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x, w = np.arange(3), 0.36
    bar_labels(ax, ax.bar(x - w / 2 - 0.01, [base.loc[m, "macro_f1"] for m in MIS], w, color="#b9b7b0", label="Solo curva"))
    bar_labels(ax, ax.bar(x + w / 2 + 0.01, [cen.loc[m, "macro_f1"] for m in MIS], w, color="#2a78d6", label="Curva + centroide"))
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 en val")
    ax.set_title("El centroide ayuda en Kepler y empeora K2 y TESS")
    legend_below(ax, 2)
    save(fig, "f09_centroide", "CNN conjunta (semilla 0) con y sin el canal de centroide. Kepler (píxel de 4″) mejora; TESS (21″) y K2 (apuntado inestable) empeoran.")


def f_variantes_cnn():
    var = [("Base (solo curva)", "cnn_gpu_val.csv"), ("+ par/impar", "cnn_oddeven_gpu_val.csv"),
           ("+ diagnósticos", "cnn_feats_gpu_val.csv"), ("+ ambas", "cnn_oddeven_feats_gpu_val.csv"),
           ("+ efeméride", "cnn_ephem_gpu_val.csv")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x, w = np.arange(len(var)), 0.24
    for i, m in enumerate(MIS):
        vals = []
        for lab, f in var:
            d = csv(f"models/{f}")
            d = d[d.entrenamiento == ("conjunta_ensemble" if (d.entrenamiento == "conjunta_ensemble").any() else "conjunta")]
            vals.append(d.set_index("mision").loc[m, "macro_f1"])
        ax.bar(x + (i - 1) * (w + 0.01), vals, w, color=MIS_COLOR[m], label=MIS_LABEL[m])
    ax.set_xticks(x, [v[0] for v in var])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro-F1 en val")
    ax.set_title("Variantes de la CNN: ninguna supera claramente a la base")
    legend_below(ax, 3)
    save(fig, "f10_variantes_cnn", "Variantes de la CNN conjunta (conjunto de 3 semillas cuando existe). Las diferencias están dentro del ruido de val.")


def f_errores_fp_kepler():
    d = csv("models/error_analysis_fp_kepler.csv")
    d = d[d.modelo == "solo_curva"].set_index("motivo")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(d))[::-1]
    left = np.zeros(len(d))
    for col, lab, color in (("acierta_FP", "Acierta FP", CLS_COLOR["FALSE_POSITIVE"]), ("como_PLANET", "Lo llama planeta", CLS_COLOR["PLANET"]),
                            ("como_NO_SIGNAL", "Lo llama sin señal", CLS_COLOR["NO_SIGNAL"])):
        ax.barh(y, d[col], left=left, color=color, label=lab, height=0.6, edgecolor="white", lw=1)
        left += d[col].to_numpy()
    ax.set_yticks(y, [f"{i} (n={n})" for i, n in zip(d.index, d.n)])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Kepler: qué hace la CNN con cada tipo de falso positivo")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    save(fig, "f11_errores_fp_kepler", "FP de Kepler en val por motivo del Robovetter. Los de contaminación (co/ec, invisibles en la curva) son el techo físico del modelo solo curva.")


# --------------------------------------------------------------------------------------------- fase 3: LOMO
def f_lomo():
    rows = {"Dummy": csv("models/lomo_dummy.csv"), "Logística (LOMO, val)": csv("models/lomo_logreg.csv"),
            "CNN (LOMO, val)": csv("models/lomo_cnn_nodepth_fixed.csv"), "CNN (LOMO, test)": csv("models/lomo_cnn_nodepth_test.csv")}
    final = csv("models/final_test.csv").set_index("mision")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = list(rows) + ["CNN viendo la misión (test)"]
    colors = ["#d3d1c7", "#b4b2a9", "#85b7eb", "#2a78d6", "#0c447c"]
    x, w = np.arange(3), 0.16
    for i, (lab, col) in enumerate(zip(labels, colors)):
        vals = [final.loc[m, "pr_auc_macro"] for m in MIS] if i == 4 else rows[lab].set_index("excluida").loc[MIS, "pr_auc_macro"].to_numpy()
        bars = ax.bar(x + (i - 2) * (w + 0.008), vals, w, color=col, label=lab)
        if i >= 3:
            bar_labels(ax, bars)
    ax.set_xticks(x, [f"{MIS_LABEL[m]} excluida" for m in MIS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PR-AUC macro")
    ax.set_title("Portabilidad: evaluar en un telescopio que el modelo no ha visto")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "f12_lomo", "LOMO: entrenar con dos misiones y evaluar en la tercera. La pérdida frente a ver la misión es pequeña en Kepler y K2; TESS es difícil incluso viéndola.")


def f_adaptacion_tess():
    items = [("AdaBN", 0.7397 - 0.7639, None), ("Recorte Kepler/K2 a 27 d", 0.753 - 0.7639, None),
             ("Pseudo-etiquetas", 0.7626 - 0.7639, (-0.007, 0.004)), ("2 sectores (mismas estrellas)", 0.8141 - 0.7804, (0.018, 0.048))]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(items))[::-1]
    for yi, (lab, d, ic) in zip(y, items):
        ax.barh(yi, d, color="#1baf7a" if d > 0 else "#b9b7b0", height=0.55)
        if ic:
            ax.errorbar(d, yi, xerr=[[d - ic[0]], [ic[1] - d]], color=INK, capsize=3, lw=1)
        end = (ic[1] if d > 0 else ic[0]) if ic else d
        ax.annotate(f"{d:+.3f}", (end, yi), xytext=(6 if d > 0 else -6, 0), textcoords="offset points", va="center",
                    ha="left" if d > 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xlim(-0.045, 0.065)
    ax.set_xlabel("Cambio de PR-AUC macro en TESS (LOMO)")
    ax.set_title("Adaptación a TESS: solo ayuda dar más datos de TESS")
    save(fig, "f13_adaptacion_tess", "Intentos de mejorar TESS sin sus etiquetas (fase 3). Barras de error: IC 95 % bootstrap cuando se midió.")


def f_curva_aprendizaje():
    d = csv("models/learning_curve_tess.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = d.n_tess_etiquetadas.replace(0, 10)
    for col, lab, color in (("pr_auc_macro", "PR-AUC macro", "#2a78d6"), ("macro_f1", "Macro-F1", "#eb6834")):
        ax.plot(x, d[col], "-o", color=color, lw=2, ms=5, label=lab)
    ax.set_xscale("log")
    ax.set_xticks(x, ["0"] + [str(v) for v in d.n_tess_etiquetadas[1:]])
    ax.minorticks_off()
    ax.set_xlabel("Estrellas de TESS etiquetadas añadidas (escala log)")
    ax.set_ylabel("Métrica en val de TESS")
    ax.set_title("¿Cuántas etiquetas de TESS hacen falta?")
    ax.legend()
    save(fig, "f14_curva_aprendizaje_tess", "Kepler + K2 más una fracción del train de TESS. Pocas etiquetas apenas ayudan: pesan poco frente a ~10 000 de Kepler y K2.")


# --------------------------------------------------------------------------------------------- fase 4: final
def f_test_final():
    d = csv("models/final_test.csv").set_index("mision")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x, w = np.arange(3), 0.36
    for i, (col, lab, color) in enumerate((("macro_f1", "Macro-F1", "#6250d6"), ("pr_auc_macro", "PR-AUC macro", "#b9b7b0"))):
        v = d.loc[MIS, col]
        err = [v - d.loc[MIS, f"{col}_lo"], d.loc[MIS, f"{col}_hi"] - v]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=color, label=lab, yerr=err, capsize=3, ecolor=INK, error_kw={"lw": 0.8})
        top_labels(ax, bars, d.loc[MIS, f"{col}_hi"])
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.set_ylim(0, 1.05)
    ax.set_title("Modelo final en test (evaluado una sola vez)")
    legend_below(ax, 2)
    save(fig, "f15_test_final", "CNN solo curva sin profundidad, conjunto de 5 semillas. IC 95 % bootstrap por estrella.")
    c = csv("models/final_test_confusion.csv").rename(columns={"Unnamed: 0": "mision", "Unnamed: 1": "real"})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3))
    confusion_panel(axes, lambda m: c[c.mision == m].set_index("real").loc[[f"real_{k}" for k in CLASSES], [f"pred_{k}" for k in CLASSES]].to_numpy(float),
                    "Matriz de confusión en test: modelo final", fig)
    save(fig, "f16_confusion_final_test", "Modelo final en test. El fallo dominante es FP de TESS predicho como planeta (34 de 89).")


def f_bls():
    d = csv("models/eval_bls.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, col, title in ((axes[0], "macro_f1", "Macro-F1"), (axes[1], "f1_NO_SIGNAL", "F1 de sin señal")):
        x, w = np.arange(3), 0.36
        for i, (ef, color) in enumerate((("catálogo", "#2a78d6"), ("BLS", "#eb6834"))):
            v = d[d.efemeride == ef].set_index("mision").loc[MIS, col]
            bar_labels(ax, ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=color, label=f"Efeméride de {ef}"))
        ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
        ax.set_title(title)
    axes[0].set_ylim(0, 1.08)
    legend_below(axes[0], 2)
    fig.suptitle("El mismo modelo con la efeméride del catálogo o la de BLS", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.8)
    save(fig, "f17_bls", "Con la efeméride de BLS, los NO_SIGNAL plegados en un pico de ruido parecen tránsitos: el modelo es un clasificador de señales ya detectadas (decisión 7).")


def f_negativos_dificiles():
    files = {("Base", "catálogo"): "final_val_pred.csv", ("Negativos difíciles", "catálogo"): "hard_val_pred.csv",
             ("Base", "BLS"): "final_val_bls_pred.csv", ("Negativos difíciles", "BLS"): "hard_val_bls_pred.csv"}
    res = {}
    for k, f in files.items():
        p = csv(f"models/{f}")
        res[k] = {m: point_metrics(g.disposition.to_numpy(), g[[f"p_{c}" for c in CLASSES]].to_numpy())["macro_f1"] for m, g in p.groupby("mission")}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)
    for ax, ef in zip(axes, ("catálogo", "BLS")):
        x, w = np.arange(3), 0.36
        for i, (mod, color) in enumerate((("Base", "#b9b7b0"), ("Negativos difíciles", "#6250d6"))):
            bar_labels(ax, ax.bar(x + (i - 0.5) * (w + 0.02), [res[(mod, ef)][m] for m in MIS], w, color=color, label=mod))
        ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
        ax.set_title(f"Efeméride de {ef}")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro-F1 en val")
    legend_below(axes[0], 2)
    fig.suptitle("Negativos difíciles: más robusto con BLS, algo peor con catálogo", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.8)
    save(fig, "f18_negativos_dificiles", "Añadir al entrenamiento NO_SIGNAL plegados en su pico de BLS. Decisión 13: base para TOIs, negativos difíciles solo para búsqueda.")


def f_errores_tipo_final():
    d = csv("models/error_fp_tipo.csv")
    d = d[(d.modelo == "final") & (d.conjunto == "val") & (d.motivo != "todos") & (d.mision != "k2")]
    d["label"] = d.mision.map(MIS_LABEL) + " · " + d.motivo + " (n=" + d.n.astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(d) + 1.2))
    y = np.arange(len(d))[::-1]
    left = np.zeros(len(d))
    for col, lab, color in (("acierta_FP", "Acierta FP", CLS_COLOR["FALSE_POSITIVE"]), ("como_PLANET", "Lo llama planeta", CLS_COLOR["PLANET"]),
                            ("como_NO_SIGNAL", "Lo llama sin señal", CLS_COLOR["NO_SIGNAL"])):
        ax.barh(y, d[col], left=left, color=color, label=lab, height=0.6, edgecolor="white", lw=1)
        left += d[col].to_numpy()
    ax.set_yticks(y, d.label)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Modelo final: tipos de falso positivo en val")
    legend_below(ax, 3, -0.06)
    save(fig, "f19_errores_tipo_final", "Kepler: motivos del Robovetter; TESS: comentarios de ExoFOP (etiquetas ruidosas). Las binarias de TESS se escapan como planeta un 31 %.")


# --------------------------------------------------------------------------------------------- fase 5: búsqueda
def f_triaje():
    t = csv("models/triage_val.csv")
    t = t[t.mision == "todas"]
    s = csv("models/series_ensemble.csv")
    s = s[s.conjunto == "estrellas BLS"]
    rows = list(zip(t.puntuacion, t.recall_fpr1)) + [(f"CNN sin plegar ({v})", r) if "+" not in v else (v.replace("base", "CNN sin plegar"), r)
                                                    for v, r in zip(s.variante, s.recall_fpr1) if v.startswith("base")]
    rows = [r for r in rows if r[0] != "SDE + CNN plegada (línea base)"]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    best = int(np.argmax(vals))
    ax.barh(y, vals, color=["#2a78d6" if i == best else "#b9b7b0" for i in range(len(rows))], height=0.6)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Recall de candidatos con un 1 % de falsas alarmas")
    ax.set_title("Triaje sin efeméride: candidato frente a sin señal")
    save(fig, "f20_triaje", "Val, todas las misiones. La combinación de la CNN sin plegar con SDE y la CNN plegada llega a 0,56 y toca techo (fase 5).")


def f_snr():
    d = csv("models/series_ensemble_snr.csv")
    cols = [c for c in d.columns if c.startswith("(")]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for (v, s), color in zip(d[["variante", "semillas"]].itertuples(index=False), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.plot(range(len(cols)), d[(d.variante == v)].iloc[0][cols].astype(float), "-o", color=color, lw=2, ms=4,
                label=f"{v} ({int(s)} semillas)")
    ax.set_xticks(range(len(cols)), [c.strip("(]").replace(", ", "–") for c in cols])
    ax.set_xlabel("SNR del tránsito inyectado")
    ax.set_ylabel("Detectados con 1 % de falsas alarmas")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("Sensibilidad del detector sin plegar")
    ax.legend(loc="upper left")
    save(fig, "f21_sensibilidad_snr", "Tránsitos inyectados en curvas sin señal de val. Por encima de SNR 30 se detecta ~90 %.")


# --------------------------------------------------------------------------------------------- fases 6 y 7
def f_multisector():
    d = csv("models/tess_multi_compare.csv")
    d = d[d.diferencia.notna()]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    y = np.arange(len(d))[::-1]
    ax.errorbar(d.diferencia, y, xerr=[d.diferencia - d.ic_lo, d.ic_hi - d.diferencia], fmt="o", color="#1baf7a", capsize=3, ms=6)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(d.grupo, d.n)])
    ax.set_xlabel("Diferencia de macro-F1 (multisector − 1 sector)")
    ax.set_title("Fase 6: varios sectores al reentrenar, sin efecto detectable")
    save(fig, "f22_multisector", "IC 95 % bootstrap. Las estrellas sin sectores nuevos bajan lo mismo que las que ganan: es variación entre entrenamientos (auditoría A1).")


def f_transitos():
    d = csv("models/transit_feats_check.csv")
    d = d[(d.source == "lomo") & (d.subset == "todas") & d.metric.isin(["auc_cnn", "auc_feats_logreg", "auc_cnn_plus_feats"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x, w = np.arange(3), 0.26
    for i, (met, lab, color) in enumerate((("auc_feats_logreg", "Solo rasgos por tránsito", "#b9b7b0"), ("auc_cnn", "CNN", "#2a78d6"),
                                           ("auc_cnn_plus_feats", "CNN + rasgos", "#6250d6"))):
        v = d[d.metric == met].set_index("mission").reindex(MIS)
        bars = ax.bar(x + (i - 1) * (w + 0.01), v.value, w, color=color, label=lab, yerr=[v.value - v.lo, v.hi - v.value],
                      capsize=2, ecolor=INK, error_kw={"lw": 0.7})
        top_labels(ax, bars, v.hi)
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("ROC-AUC planeta frente a FP")
    ax.set_title("Fase 7: los tránsitos individuales no añaden nada a la CNN")
    legend_below(ax, 3)
    save(fig, "f23_transitos_individuales", "Predicciones LOMO fuera de muestra. |z| par/impar, dispersión y tránsito más desviado (research/15). Ganancia en TESS +0,000.")


def f_tois():
    p = pd.read_csv(R / "toi/predictions_v1.csv", dtype={"toi": str})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(p[~p.efemeride_incierta].p_planet_given_signal, bins, color="#6250d6", alpha=0.85, label="Efeméride fiable")
    ax.hist(p[p.efemeride_incierta].p_planet_given_signal, bins, color="#eda100", alpha=0.9, label="Efeméride incierta")
    ax.set_xlabel("P(planeta | señal)")
    ax.set_ylabel("TOIs")
    ax.set_title("v1: 3 536 TOIs abiertos")
    ax.legend(fontsize=8)
    ax = axes[1]
    e = p[p.exominerpp_score.notna() & ~p.efemeride_incierta]
    ax.scatter(e.exominerpp_score, e.p_planet_given_signal, s=5, color="#2a78d6", alpha=0.35, lw=0)
    ax.set_xlabel("Puntuación de ExoMiner++")
    ax.set_ylabel("Nuestro P(planeta | señal)")
    ax.set_title(f"Frente a ExoMiner++ (n={len(e)})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax = axes[2]
    p["bin"] = pd.cut(p.drift, [-0.01, 0.1, 0.25, 0.5, 1, np.inf], labels=["≤0,1", "0,1–\n0,25", "0,25–\n0,5", "0,5–1", ">1"])
    g = p.groupby("bin", observed=True).pred.value_counts(normalize=True).unstack().fillna(0)
    left = np.zeros(len(g))
    for c in CLASSES:
        ax.bar(range(len(g)), g[c], bottom=left, color=CLS_COLOR[c], label=CLS_LABEL[c], edgecolor="white", lw=1)
        left += g[c].to_numpy()
    ax.set_xticks(range(len(g)), g.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel("Deriva de la efeméride (σ_t / duración)")
    ax.set_title("Clase predicha según la deriva")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    save(fig, "f24_tois_v1", "Predicciones congeladas v1 (etiqueta v1 en GitHub). Con deriva > 0,25 la efeméride no sirve y aumentan NO_SIGNAL y FP (research/16).")


def f_esquema_lomo():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.grid(False)
    ax.axis("off")
    for r, held in enumerate(MIS):
        y = 2 - r
        ax.text(-0.15, y + 0.35, f"Pliegue {r + 1}", ha="right", va="center", fontsize=9.5, color=INK)
        for c, m in enumerate(MIS):
            ev = m == held
            ax.add_patch(matplotlib.patches.FancyBboxPatch((c * 1.25, y), 1.1, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor="#e6f1fb" if not ev else "#faeeda", edgecolor="#185fa5" if not ev else "#854f0b", lw=0.8))
            ax.text(c * 1.25 + 0.55, y + 0.47, MIS_LABEL[m], ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#0c447c" if not ev else "#633806")
            ax.text(c * 1.25 + 0.55, y + 0.18, "evaluación\n(una sola vez)" if ev else "entrenamiento\ny selección", ha="center",
                    va="center", fontsize=7, color="#185fa5" if not ev else "#854f0b", linespacing=1.1)
    ax.text(3.9, 1.9, "Bucle interno (solo con las dos\nmisiones de entrenamiento):\nentrenar en A → validar en B\ny al revés; se elige la\nconfiguración con mejor PR-AUC", fontsize=8.5,
            va="center", color=MUTED)
    ax.set_xlim(-0.9, 5.6)
    ax.set_ylim(-0.1, 3)
    ax.set_title("Protocolo LOMO anidado: la misión evaluada no participa en ninguna decisión")
    save(fig, "f25_esquema_lomo", "Cada pliegue deja fuera una misión, elige el modelo solo con las otras dos y la evalúa una única vez.")


def f_etiquetas():
    d = csv("labels/strength_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, cls in zip(axes, ["PLANET", "FALSE_POSITIVE"]):
        t = d[d.disposition == cls].set_index("mission").reindex(MIS)
        left = np.zeros(3)
        for col, lab, color in (("fuerte", "Fuerte (se evalúa)", "#2a78d6"), ("debil", "Débil (candidato)", "#85b7eb"),
                                ("excluida", "Excluida", "#d3d1c7")):
            ax.barh(range(3), t[col], left=left, color=color, label=lab, height=0.6, edgecolor="white", lw=1)
            left += t[col].to_numpy()
        ax.set_yticks(range(3), [MIS_LABEL[m] for m in MIS])
        ax.invert_yaxis()
        ax.set_title(CLS_LABEL[cls])
        ax.set_xlabel("Objetos del dataset")
    legend_below(axes[0], 3, -0.25)
    fig.suptitle("Fuerza de las etiquetas por misión", x=0.01, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.8)
    save(fig, "f26_etiquetas", "Solo las etiquetas fuertes (confirmados y FP claros) se usan para evaluar; los candidatos son etiquetas débiles.")


def f_atajos_evolucion():
    steps = ["Al azar,\ntodos los sectores", "1 segmento\nen K2 y TESS", "Emparejados por\nsegmento y brillo", "+ efeméride del\nmismo segmento"]
    vals = {"kepler": [0.64, 0.64, 0.64, 0.64], "k2": [0.90, 0.87, 0.85, 0.84], "tess": [0.83, 0.70, 0.51, 0.55]}   # research/05
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for m in MIS:
        ax.plot(range(4), vals[m], "-o", color=MIS_COLOR[m], lw=2, ms=6, label=MIS_LABEL[m])
        ax.annotate(f"{vals[m][-1]:.2f}", (3, vals[m][-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax.text(0, 0.47, "0,5 = sin atajo", fontsize=8, color=MUTED, va="top")
    ax.set_xticks(range(4), steps, fontsize=8.5)
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("ROC-AUC sin mirar el tránsito")
    ax.set_title("Cómo se fue quitando el atajo de los negativos")
    legend_below(ax, 3, -0.3)
    save(fig, "f27_atajos_evolucion", "Capacidad de distinguir los negativos fabricados sin mirar el tránsito, en cada versión de su construcción (research/05).")


def f_bls_periodo():
    q = csv("sample/qc_bls.csv")
    q = q[(q.split == "val") & q.disposition.isin(["PLANET", "FALSE_POSITIVE"]) & q.match.notna()]
    q["cat"] = q.match.map(lambda s: "Periodo correcto" if s == "igual" else ("Otro periodo" if s == "otro" else "Alias (½, 2×, ⅓…)"))
    g = q.groupby(["mission", "disposition"]).cat.value_counts(normalize=True).unstack().fillna(0)
    rows = [(m, c) for m in MIS for c in ["PLANET", "FALSE_POSITIVE"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for cat, color in (("Periodo correcto", "#1baf7a"), ("Alias (½, 2×, ⅓…)", "#eda100"), ("Otro periodo", "#b9b7b0")):
        v = np.array([g.loc[r, cat] if r in g.index and cat in g.columns else 0 for r in rows])
        ax.barh(y, v, left=left, color=color, label=cat, height=0.6, edgecolor="white", lw=1)
        left += v
    n = q.groupby(["mission", "disposition"]).size()
    ax.set_yticks(y, [f"{MIS_LABEL[m]} · {CLS_LABEL[c]} (n={n.get((m, c), 0)})" for m, c in rows])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_title("¿Encuentra BLS el periodo correcto?")
    legend_below(ax, 3, -0.18)
    save(fig, "f28_bls_periodo", "Periodo de BLS frente al del catálogo en val. BLS acierta ~2 de cada 3; los FP dan más alias de ½ (binarias con dos eclipses parecidos).")


def f_calibracion_par_impar():
    f = csv("sample/transit_feats.csv")
    f = f[f.get("error", pd.Series(index=f.index, dtype=object)).isna()]
    f["out"] = f.z_odd_even.abs() > 3
    g = f.groupby(["mission", "disposition"]).out.mean().unstack().reindex(MIS)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    x, w = np.arange(3), 0.36
    for i, c in enumerate(["PLANET", "FALSE_POSITIVE"]):
        bar_labels(ax, ax.bar(x + (i - 0.5) * (w + 0.02), g[c], w, color=CLS_COLOR[c], label=CLS_LABEL[c]), "{:.1%}")
    ax.axhline(0.145, color="#e34948", lw=1, ls="--")
    ax.text(2.45, 0.15, "planetas con el test\nantiguo (14,5 %)", fontsize=7.5, color="#a32d2d", va="bottom", ha="right")
    ax.axhline(0.003, color=INK, lw=0.8, ls=":")
    ax.text(-0.45, 0.008, "ideal 0,3 %", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, [MIS_LABEL[m] for m in MIS])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_ylim(0, 0.36)
    ax.set_ylabel("Estrellas con |z| par/impar > 3")
    ax.set_title("El nuevo test par/impar está mucho mejor calibrado")
    legend_below(ax, 2)
    save(fig, "f29_calibracion_par_impar", "Fracción de estrellas con diferencia par/impar significativa usando el ajuste tránsito a tránsito con ruido rojo (research/15).")


def write_index() -> None:
    lines = ["# Figuras del proyecto", "", "Generadas con `scripts/checks/figuras.py` a partir de `results/` (no entrena nada).", ""]
    lines += [f"## {n}\n\n![{n}]({n}.png)\n\n{c}\n" for n, c in INDEX]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    for f in (f_muestra, f_galeria, f_atajos, f_fuga_mision, f_modelos, f_prauc_clases, f_confusiones, f_centroide,
              f_variantes_cnn, f_errores_fp_kepler, f_lomo, f_adaptacion_tess, f_curva_aprendizaje, f_test_final, f_bls,
              f_negativos_dificiles, f_errores_tipo_final, f_triaje, f_snr, f_multisector, f_transitos, f_tois,
              f_esquema_lomo, f_etiquetas, f_atajos_evolucion, f_bls_periodo, f_calibracion_par_impar):
        try:
            f()
        except Exception as e:
            print("FALLO", f.__name__, type(e).__name__, e, flush=True)
    write_index()
