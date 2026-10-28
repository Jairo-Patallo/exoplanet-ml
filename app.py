"""App Streamlit para explorar las curvas de luz. Uso: streamlit run app.py (tras scripts/preprocess.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROC = Path("data/processed")
COLORS = {"PLANET": "#1f77b4", "FALSE_POSITIVE": "#d62728", "NO_SIGNAL": "#7f7f7f"}
st.set_page_config(page_title="Curvas de luz de exoplanetas", layout="wide")


@st.cache_data
def load_meta() -> pd.DataFrame:
    return pd.read_parquet(PROC / "meta.parquet")


@st.cache_resource
def load_curves() -> dict:
    # mmap: flux_raw pesa ~190 MB y solo leemos la fila seleccionada
    return {v: np.load(PROC / f"curves_{v}.npy", mmap_mode="r")
            for v in ["flux_raw", "flux_global", "flux_local", "flux_odd", "flux_even"]}


if not (PROC / "meta.parquet").exists():
    st.error("Faltan datos. Ejecuta primero: python scripts/preprocess.py")
    st.stop()

meta, C = load_meta(), load_curves()
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Explorador de una estrella", "Comparar clases", "Variables"])

# ---------- 1. Resumen ----------
with tab1:
    st.subheader("Split test: composición")
    c1, c2 = st.columns(2)
    c1.dataframe(meta["disposition"].value_counts().rename("estrellas"))
    c2.dataframe(pd.crosstab(meta["mission"], meta["disposition"]))
    st.caption(f"{len(meta)} estrellas. `NO_SIGNAL` solo existe en Kepler y no tiene curvas plegadas (ceros). "
               f"{int((meta.folded_empty & (meta.disposition != 'NO_SIGNAL')).sum())} estrellas TESS/K2 "
               "tampoco tienen vistas plegadas por faltar periodo/duración/época.")

# ---------- 2. Explorador ----------
with tab2:
    f1, f2, f3 = st.columns(3)
    cls = f1.selectbox("Clase", ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"])
    mis = f2.selectbox("Misión", ["todas", "kepler", "tess", "k2"])
    sub = meta[(meta.disposition == cls) & ((meta.mission == mis) | (mis == "todas"))]
    if cls != "NO_SIGNAL":
        sub = sub[~sub.folded_empty]
    if sub.empty:
        st.warning("No hay estrellas con ese filtro.")
        st.stop()
    seed = f3.number_input("Semilla del azar", 0, 9999, 0)
    pick = st.radio("Estrella", ["aleatoria", "elegir de la lista"], horizontal=True)
    if pick == "aleatoria":
        idx = int(sub.sample(1, random_state=int(seed)).index[0])
    else:
        sel = st.selectbox("star_id", sub.star_id.tolist())
        idx = int(sub.index[sub.star_id == sel][0])
    row = meta.loc[idx]
    st.markdown(f"**{row.star_id}** · {row.mission} · `{row.disposition}`")

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot(C["flux_raw"][idx], ".", ms=1, color=COLORS[cls])
    ax.set(title="Curva de luz completa (flux_raw, sin plegar)", xlabel="índice temporal (20 000 puntos)", ylabel="flujo normalizado")
    st.pyplot(fig)

    if not row.folded_empty:
        a, b, c = st.columns(3)
        for col, key, title in [(a, "flux_global", "Plegada global (fase completa)"),
                                (b, "flux_local", "Vista local (zoom al tránsito)")]:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.plot(C[key][idx], color=COLORS[cls])
            ax.set(title=title, xlabel="fase (bins)", ylabel="flujo")
            col.pyplot(fig)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(C["flux_odd"][idx], label="impares", color="tab:orange")
        ax.plot(C["flux_even"][idx], label="pares", color="tab:green", alpha=.8)
        ax.set(title="Tránsitos impares vs pares", xlabel="fase (bins)")
        ax.legend()
        c.pyplot(fig)
        st.caption("En un planeta ambas curvas deben coincidir; si la profundidad difiere claramente, sugiere una binaria eclipsante.")
    st.json({k: (None if pd.isna(v) else v) for k, v in row[
        ["period_days", "duration_hrs", "depth_ppm", "planet_radius_earth", "teff", "radius", "kepmag", "cdpp_3hr"]].items()})

# ---------- 3. Comparar clases ----------
with tab3:
    view = st.radio("Vista", ["flux_local", "flux_global"], horizontal=True)
    mission = st.selectbox("Misión (comparación)", ["todas", "kepler", "tess", "k2"], key="m3")
    base = meta[~meta.folded_empty & (meta.disposition != "NO_SIGNAL")]
    if mission != "todas":
        base = base[base.mission == mission]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for cl in ["PLANET", "FALSE_POSITIVE"]:
        ids = base.index[base.disposition == cl]
        if len(ids) == 0:
            continue
        M = np.asarray(C[view][ids])
        med, lo, hi = np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)
        x = np.arange(M.shape[1])
        ax.plot(x, med, color=COLORS[cl], label=f"{cl} (n={len(ids)})")
        ax.fill_between(x, lo, hi, color=COLORS[cl], alpha=.2)
    ax.set(title=f"Mediana y rango intercuartílico · {view}", xlabel="fase (bins)", ylabel="flujo normalizado")
    ax.legend()
    st.pyplot(fig)
    st.caption("Las curvas dependen mucho de la profundidad de cada estrella; la mediana muestra la forma típica, no un caso concreto.")

    st.markdown("**Galería aleatoria (vista local)**")
    n = st.slider("Ejemplos por clase", 2, 8, 4)
    gseed = st.number_input("Semilla galería", 0, 9999, 1)
    for cl in ["PLANET", "FALSE_POSITIVE"]:
        ids = base[base.disposition == cl].sample(min(n, (base.disposition == cl).sum()), random_state=int(gseed)).index
        cols = st.columns(len(ids))
        for col, i in zip(cols, ids):
            fig, ax = plt.subplots(figsize=(2.6, 2))
            ax.plot(C["flux_local"][i], color=COLORS[cl])
            ax.set_title(f"{cl[:4]} · {meta.mission[i]}", fontsize=8)
            ax.tick_params(labelsize=6)
            col.pyplot(fig)

# ---------- 4. Variables ----------
with tab4:
    var = st.selectbox("Variable", ["period_days", "duration_hrs", "depth_ppm", "planet_radius_earth",
                                    "teff", "radius", "kepmag", "cdpp_3hr", "raw_std_ppm", "odd_even_diff_ppm"])
    logx = st.checkbox("Escala logarítmica en X", value=var in ("period_days", "depth_ppm", "odd_even_diff_ppm", "cdpp_3hr", "raw_std_ppm"))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    vals = meta[var].dropna()
    vals = vals[vals > 0] if logx else vals
    bins = np.geomspace(vals.quantile(.01), vals.quantile(.99), 40) if logx else np.linspace(vals.quantile(.01), vals.quantile(.99), 40)
    for cl in ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]:
        v = meta.loc[meta.disposition == cl, var].dropna()
        if len(v):
            ax.hist(v, bins=bins, alpha=.5, color=COLORS[cl], label=f"{cl} (n={len(v)})", density=True)
    if logx:
        ax.set_xscale("log")
    ax.set(title=f"Distribución de {var} por clase", xlabel=var, ylabel="densidad")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(meta.groupby("disposition")[var].describe().round(2))
