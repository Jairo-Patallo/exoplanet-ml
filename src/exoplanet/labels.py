"""Armonización de etiquetas entre misiones (tarea #33).

El dataset de HF une criterios distintos bajo PLANET / FALSE_POSITIVE. Aquí se cruza con la disposición original
(TESS: TOI, K2: archivo de candidatos, Kepler: label_source) y se asigna una *fuerza* a cada fila:

  fuerte    -> etiqueta clara (planeta confirmado, falso positivo claro). Es la que se usa en test.
  debil     -> candidato sin confirmar (PC / CANDIDATE / Robovetter PC). Solo entra en entrenamiento, como variante.
  excluida  -> ambigua, contradictoria con el archivo actual o de otra naturaleza (APC, FA...). No se usa.

Mapeo versionado y razonado en research/04-armonizacion-etiquetas.md. Cualquier combinación no listada es `excluida`.
"""
import numpy as np
import pandas as pd

from exoplanet.data import DATA, TARGET

# (misión, clase del dataset, disposición original) -> fuerza. En Kepler la "disposición original" es label_source.
STRENGTH = {
    ("kepler", "PLANET", "kepler_confirmed"): "fuerte",
    ("kepler", "PLANET", "kepler_robovetter"): "debil",
    ("kepler", "FALSE_POSITIVE", "kepler_robovetter"): "fuerte",
    ("tess", "PLANET", "CP"): "fuerte",
    ("tess", "PLANET", "KP"): "fuerte",
    ("tess", "PLANET", "PC"): "debil",
    ("tess", "FALSE_POSITIVE", "FP"): "fuerte",
    ("k2", "PLANET", "CONFIRMED"): "fuerte",
    ("k2", "PLANET", "CANDIDATE"): "debil",
    ("k2", "FALSE_POSITIVE", "FALSE POSITIVE"): "fuerte",
}


def nearest_period(rows: pd.DataFrame, key: str, ids: pd.Series, periods: pd.Series, col: str, per_col: str) -> pd.Series:
    """Para cada fila (id, periodo): el valor de `col` del candidato con ese id y periodo más cercano (NaN si no hay)."""
    by_id = {k: g for k, g in rows.groupby(key)}
    out = []
    for i, p in zip(ids, periods):
        g = by_id.get(i)
        if g is None:
            out.append(np.nan)
        elif np.isnan(p) or g[per_col].isna().all():
            out.append(g[col].iloc[0])
        else:
            out.append(g.loc[(g[per_col] - p).abs().idxmin(), col])
    return pd.Series(out, index=ids.index)


def external_disposition(df: pd.DataFrame, toi: pd.DataFrame | None = None, k2: pd.DataFrame | None = None) -> pd.Series:
    """Disposición original de cada fila. `df` necesita star_id, mission, label_source y period_days.
    Kepler -> label_source; TESS -> tfopwg_disp del TOI de periodo más cercano; K2 -> disposition del candidato más cercano."""
    out = pd.Series("sin cruce", index=df.index, dtype=object)
    k = df.mission == "kepler"
    out[k] = df.label_source[k]

    t = df.mission == "tess"
    if t.any():
        toi = pd.read_csv(DATA / "labels" / "toi.csv") if toi is None else toi
        tid = df.star_id[t].str.replace("TIC_", "", regex=False).astype(np.int64)
        out[t] = nearest_period(toi, "tid", tid, df.period_days[t], "tfopwg_disp", "pl_orbper").fillna("sin TOI")

    s = df.mission == "k2"
    if s.any():
        k2 = pd.read_csv(DATA / "labels" / "k2pandc.csv") if k2 is None else k2
        k2 = k2.assign(epic=k2.epic_hostname.str.replace("EPIC ", "", regex=False).str.strip())
        epic = df.star_id[s].str.replace("EPIC_", "", regex=False)
        out[s] = nearest_period(k2, "epic", epic, df.period_days[s], "disposition", "pl_orbper").fillna("sin candidato")
    return out


def label_strength(df: pd.DataFrame, **external) -> pd.Series:
    """Fuerza de cada fila PLANET / FALSE_POSITIVE; NO_SIGNAL queda `excluida` aquí (sus negativos se generan aparte)."""
    ext = external_disposition(df, **external)
    keys = zip(df.mission, df[TARGET], ext)
    return pd.Series([STRENGTH.get(k, "excluida") for k in keys], index=df.index)
