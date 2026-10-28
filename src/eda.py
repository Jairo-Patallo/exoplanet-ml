"""Análisis exploratorio con propósito (Fase 1 de la hoja de ruta) + exploración de las variables de la Fase 2.

Uso:  python src/eda.py
Solo usa TRAIN para explorar; VAL únicamente para el test de atajos. TEST no se toca.
Salida: figuras y tablas en results/eda/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from features import LOC_VALID, catalog_features, curve_features

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
SEED = 42
OUT = Path("results/eda")
VIEWS = {"flux_local": 81, "flux_global": 201, "flux_odd": 201, "flux_even": 201}
STELLAR = ["teff", "logg", "radius", "mass", "metallicity", "kepmag", "cdpp_3hr"]
TRANSIT = ["period_days", "duration_hrs", "depth_ppm", "planet_radius_earth"]
TAB = ["star_id", "mission", "disposition", "label_source"] + TRANSIT + STELLAR
COLORS = {"PLANET": "#1f77b4", "FALSE_POSITIVE": "#d62728", "NO_SIGNAL": "#7f7f7f"}
rng = np.random.default_rng(SEED)


def load(split: str, views: bool) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{split}.parquet", columns=TAB + (list(VIEWS) if views else []))
    for v, n in (VIEWS.items() if views else []):
        df[v] = list(np.stack([np.frombuffer(b, np.float32) for b in df[v]]).reshape(-1, n))
    return df


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. Calidad e integridad
def integrity(df: pd.DataFrame) -> None:
    section("1. Integridad: valores imposibles y composición")
    print(pd.crosstab(df.mission, df.disposition, margins=True))
    sig = df[df.disposition != "NO_SIGNAL"]
    checks = {
        "period_days <= 0": sig.period_days <= 0,
        "duration_hrs <= 0": sig.duration_hrs <= 0,
        "depth_ppm <= 0": sig.depth_ppm <= 0,
        "depth_ppm >= 1e6 (caída >= 100 %)": sig.depth_ppm >= 1e6,
        "duración > periodo": sig.duration_hrs / 24 > sig.period_days,
        "Rp catálogo > 100 R_T": sig.planet_radius_earth > 100,
        "vistas plegadas todo cero": sig.flux_global.map(lambda a: not a.any()),
    }
    t = pd.DataFrame({k: v.groupby(sig.disposition).sum() for k, v in checks.items()}).T
    print("\nFilas PLANET/FP con valores físicamente imposibles o sospechosos:\n", t)
    print("\nPLANET con Rp > 100 R_T por label_source:\n",
          sig[(sig.disposition == "PLANET") & (sig.planet_radius_earth > 100)].label_source.value_counts())


# ---------------------------------------------------------------- 2. Valores faltantes
def missingness(df: pd.DataFrame) -> None:
    section("2. Tasa de NaN por misión y por clase (%)")
    cols = STELLAR + TRANSIT
    by_m = df[cols].isna().groupby(df.mission).mean().mul(100).round(1)
    by_c = df[cols].isna().groupby(df.disposition).mean().mul(100).round(1)
    print(by_m, "\n\n", by_c)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    for ax, t, title in [(axes[0], by_m, "por misión"), (axes[1], by_c, "por clase")]:
        im = ax.imshow(t.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
        ax.set(xticks=range(len(cols)), yticks=range(len(t)), title=f"% NaN {title}")
        ax.set_xticklabels(cols, rotation=45, ha="right"), ax.set_yticklabels(t.index)
        for (i, j), v in np.ndenumerate(t.values):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "02_missingness.png", dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------- 3. Test de atajos
def shortcut_test(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    """Si un modelo que solo ve QUÉ falta (o la misión) supera al baseline, hay un atajo."""
    section("3. Test de atajos: entrenar en train, evaluar en val (macro-F1)")
    tasks = {"3 clases": lambda d: d, "vetting (PLANET vs FP)": lambda d: d[d.disposition != "NO_SIGNAL"]}
    inputs = {
        "dummy (clase mayoritaria)": None,
        "solo máscara isna() estelar": lambda d: d[STELLAR].isna().astype(int),
        "solo misión (one-hot)": lambda d: pd.get_dummies(d.mission).reindex(columns=["kepler", "k2", "tess"], fill_value=0),
    }
    rows = []
    for tname, f in tasks.items():
        a, b = f(tr), f(va)
        for iname, g in inputs.items():
            if g is None:
                m, Xa, Xb = DummyClassifier(strategy="most_frequent"), np.zeros((len(a), 1)), np.zeros((len(b), 1))
            else:
                m, Xa, Xb = LogisticRegression(max_iter=1000), g(a), g(b)
            pred = m.fit(Xa, a.disposition).predict(Xb)
            rows.append((tname, iname, f1_score(b.disposition, pred, average="macro")))
    print(pd.DataFrame(rows, columns=["tarea", "entrada", "macro-F1 val"]).round(3).to_string(index=False))


# ---------------------------------------------------------------- 4. Separabilidad univariante
def auc_ci(y: np.ndarray, x: np.ndarray, n_boot: int = 200) -> tuple[float, float, float]:
    """AUC = P(X_planet > X_fp) (estadístico U de Mann-Whitney normalizado) + IC 95 % bootstrap percentil."""
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 20:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, x)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], x[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def separability(v: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    section("4. Separabilidad PLANET vs FP: AUC [IC95 %] global y POR MISIÓN (efecto Simpson)")
    print("AUC = 0.5 -> la variable no separa; |AUC-0.5| grande -> separa. AUC<0.5: PLANET tiene valores MENORES.")
    y = (v.disposition == "PLANET").astype(int).values
    rows = {}
    for f in feats:
        row = {}
        for name, mask in [("todas", np.ones(len(v), bool))] + [(m, (v.mission == m).values) for m in ["kepler", "k2", "tess"]]:
            auc, lo, hi = auc_ci(y[mask], v[f].values[mask].astype(float))
            row[name] = "—" if np.isnan(auc) else f"{auc:.2f} [{lo:.2f},{hi:.2f}]"
        rows[f] = row
    t = pd.DataFrame(rows).T
    print(t.to_string())
    t.to_csv(OUT / "04_auc_by_mission.csv")
    return t


def histograms(v: pd.DataFrame, feats: list[str]) -> None:
    """Histogramas por clase en escala log cuando la variable abarca varios órdenes de magnitud."""
    n = len(feats)
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(16, 3.2 * np.ceil(n / 4)))
    for ax, f in zip(axes.flat, feats):
        x = v[f].astype(float)
        pos = x[x > 0]
        logx = len(pos) > 0.95 * x.notna().sum() and pos.quantile(0.99) / pos.quantile(0.01) > 100
        x = np.log10(x.where(x > 0)) if logx else x
        lo, hi = x.quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 50)
        for c in ["PLANET", "FALSE_POSITIVE"]:
            ax.hist(x[v.disposition == c].dropna().clip(lo, hi), bins=bins, density=True, alpha=0.5,
                    color=COLORS[c], label=c)
        ax.set_title(f"log10({f})" if logx else f, fontsize=9)
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "04_histograms_vetting.png", dpi=110)


# ---------------------------------------------------------------- 5. Reglas físicas
def physical_rules(v: pd.DataFrame) -> None:
    section("5. Reglas físicas a mano (positivo = FALSE_POSITIVE), evaluadas en train")
    y_fp = (v.disposition == "FALSE_POSITIVE").values
    rules = {
        "δ > 2 % (regla original)": v.depth_ppm > 2e4,
        "Rp implícito > 22 R_T": v.rp_implied > 22,
        "log ρ_tr/ρ_cat < -1 (densidad 10x menor)": v.log_rho_ratio < -1,
        "par/impar z > 3": v.odd_even_z > 3,
        "secundario z > 3": v.secondary_z > 3,
        "forma τ/T > 0.35 (V) y SNR_local>10": (v.ingress_frac > 0.35) & (v.snr_local > 10),
    }
    rows = []
    for name, r in rules.items():
        pred = r.fillna(False).astype(bool).values   # NaN -> no se marca (la regla no se puede aplicar)
        tp, fp = (pred & y_fp).sum(), (pred & ~y_fp).sum()
        rows.append((name, pred.mean(), tp / max(tp + fp, 1), tp / y_fp.sum(), f1_score(y_fp, pred)))
    print(pd.DataFrame(rows, columns=["regla", "% marcado", "precision FP", "recall FP", "F1 FP"]).round(3).to_string(index=False))
    print(f"(Referencia: proporción de FP en vetting train = {y_fp.mean():.3f}; un clasificador al azar tiene precision ≈ esa cifra)")
    # Calibración de los z: si el modelo de ruido es correcto, en planetas (sin diferencia par/impar ni
    # secundario) |z| ~ |N(0,1)| -> mediana teórica 0.674 y P(|z|>3) = 0.0027.
    p = v[v.disposition == "PLANET"]
    for f in ["odd_even_z", "secondary_z"]:
        z = p[f].abs().dropna()
        print(f"Calibración {f} en PLANET: mediana |z| = {z.median():.2f} (teórico 0.67), "
              f"P(|z|>3) = {(z > 3).mean():.3f} (teórico 0.003)")


# ---------------------------------------------------------------- 6. Correlaciones
def correlations(v: pd.DataFrame, feats: list[str]) -> None:
    section("6. Correlación de Spearman (rango; robusta a escalas log y outliers)")
    c = v[feats].astype(float).corr(method="spearman")
    pairs = c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
    print("Pares con |ρ_s| > 0.7 (redundancia):\n", pairs[pairs.abs() > 0.7].sort_values(key=abs, ascending=False).round(2).to_string())
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(feats)), yticks=range(len(feats)))
    ax.set_xticklabels(feats, rotation=60, ha="right", fontsize=8), ax.set_yticklabels(feats, fontsize=8)
    fig.colorbar(im, shrink=0.8)
    fig.savefig(OUT / "06_spearman.png", dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- 7. Forma de las curvas + PCA
def curve_shapes(v: pd.DataFrame) -> None:
    section("7. Forma del tránsito: perfiles medios normalizados y PCA de flux_local")
    # Solo señales claras: con SNR baja la "forma" es casi todo ruido
    s = v[v.snr_local > 10]
    L = np.stack(s.flux_local.values)[:, LOC_VALID]
    shape = (1 - L) / s.depth_local_ppm.values[:, None] * 1e6   # 0 fuera, ≈1 en el fondo: compara FORMA, no profundidad
    print(f"Curvas con SNR_local > 10: {len(s)} ({s.disposition.value_counts().to_dict()})")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for c in ["PLANET", "FALSE_POSITIVE"]:
        m = (s.disposition == c).values
        q = np.percentile(shape[m], [25, 50, 75], axis=0)
        x = np.arange(LOC_VALID.start, LOC_VALID.stop)
        axes[0].plot(x, q[1], color=COLORS[c], label=f"{c} (mediana, n={m.sum()})")
        axes[0].fill_between(x, q[0], q[2], color=COLORS[c], alpha=0.2)
    axes[0].axvspan(35, 45, color="orange", alpha=0.1, label="ventana T")
    axes[0].set(title="flux_local normalizado: (1-f)/δ", xlabel="bin", ylabel="déficit relativo"), axes[0].legend(fontsize=7)
    axes[0].invert_yaxis()

    # PCA = SVD de la matriz centrada. Se ajusta solo con estas curvas de train.
    pca = PCA(n_components=5, random_state=SEED).fit(shape)
    Z = pca.transform(shape)
    print("Varianza explicada por PC1..PC5:", np.round(pca.explained_variance_ratio_, 3))
    y = (s.disposition == "PLANET").astype(int).values
    for i in range(3):
        print(f"  AUC de PC{i + 1} para PLANET vs FP: {roc_auc_score(y, Z[:, i]):.3f}")
    for c in ["FALSE_POSITIVE", "PLANET"]:
        m = (s.disposition == c).values
        axes[1].scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.3, color=COLORS[c], label=c)
    lim = np.percentile(Z[:, :2], [1, 99], axis=0)
    axes[1].set(xlim=lim[:, 0], ylim=lim[:, 1], title="PCA de la forma (PC1 vs PC2)", xlabel="PC1", ylabel="PC2")
    axes[1].legend(fontsize=7, markerscale=5)
    for i in range(3):
        axes[2].plot(x, pca.components_[i], label=f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%})")
    axes[2].set(title="Componentes principales (patrones de forma)", xlabel="bin"), axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "07_curve_shapes_pca.png", dpi=110)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load("train", views=True)
    va = load("val", views=False)
    print(f"train: {len(tr)} filas | val: {len(va)} filas (val solo para el test de atajos)")

    integrity(tr)
    missingness(tr)
    shortcut_test(tr, va)

    # Tarea de vetting: PLANET vs FP con vistas plegadas válidas (Fase 0 de la hoja de ruta)
    v = tr[(tr.disposition != "NO_SIGNAL") & tr.flux_global.map(lambda a: a.any())].copy()
    st = {k: np.stack(v[k].values) for k in VIEWS}
    v = pd.concat([v, catalog_features(v),
                   curve_features(st["flux_local"], st["flux_odd"], st["flux_even"], st["flux_global"]).set_index(v.index)], axis=1)
    derived = ["rp_implied", "log_rp_ratio", "log_rho_ratio", "duty_cycle", "snr_expected",
               "depth_local_ppm", "snr_local", "ingress_frac", "odd_even_z", "secondary_z"]
    print(f"\nVetting (train, con vistas): {len(v)} filas -> {v.disposition.value_counts().to_dict()}")
    print("Coherencia catálogo vs curva: mediana |log10(Rp_implícito/Rp_catálogo)| =",
          f"{v.log_rp_ratio.abs().median():.3f};  Spearman(depth_ppm, depth_local) =",
          f"{v[['depth_ppm', 'depth_local_ppm']].corr('spearman').iloc[0, 1]:.3f}")

    feats = TRANSIT + STELLAR + derived
    separability(v, feats)
    histograms(v, feats)
    physical_rules(v)
    correlations(v, feats)
    curve_shapes(v)
    v[["star_id", "mission", "disposition"] + derived].to_csv(OUT / "derived_features_train.csv", index=False)
    print(f"\nFiguras y tablas en {OUT}/")


if __name__ == "__main__":
    main()
