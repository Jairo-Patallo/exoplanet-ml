"""Plegado de una curva en las vistas del modelo (tarea #30). Función pura sobre arrays: no usa catálogo ni misión.

Entrada: tiempo (días), flujo ya aplanado y normalizado a 1 fuera de tránsito, y la efeméride (periodo, época, duración, en días).
Vistas (mediana por bin, flujo normalizado a 1; el tránsito cae en el bin central de cada vista):
  local   81 bins en [-4, 4] duraciones                    -> tránsito en el bin 40
  global  201 bins en fase [-0.5, 0.5]                     -> tránsito en el bin 100
  odd/even  como global, solo con los tránsitos impares / pares (n = round((t - t0) / P))
Bins vacíos: se interpolan si son <= MAX_EMPTY de la vista (y se guarda la fracción); si son más, la vista no es válida.
"""
from dataclasses import dataclass, field

import numpy as np

N_LOCAL, N_GLOBAL = 81, 201
LOCAL_HALF_WIDTH = 4.0      # en duraciones
MAX_EMPTY = 0.10
MIN_DURATION_CADENCES = 2


@dataclass
class Views:
    local: np.ndarray
    glob: np.ndarray
    odd: np.ndarray
    even: np.ndarray
    qc: dict = field(default_factory=dict)


def bin_median(x: np.ndarray, y: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    """Mediana de y en n bins iguales de [lo, hi]; NaN en los bins sin puntos."""
    idx = np.floor((x - lo) / (hi - lo) * n).astype(int)
    ok = (idx >= 0) & (idx < n)
    idx, y = idx[ok], y[ok]
    out = np.full(n, np.nan)
    order = np.argsort(idx, kind="stable")
    idx, y = idx[order], y[order]
    bins, start = np.unique(idx, return_index=True)
    for b, chunk in zip(bins, np.split(y, start[1:])):
        out[b] = np.median(chunk)
    return out


def fill_gaps(view: np.ndarray) -> tuple[np.ndarray, float]:
    """Interpola linealmente los bins vacíos. Devuelve (vista, fracción de bins que estaban vacíos)."""
    empty = np.isnan(view)
    frac = float(empty.mean())
    if frac == 0.0 or frac == 1.0:
        return view, frac
    x = np.arange(len(view))
    return np.where(empty, np.interp(x, x[~empty], view[~empty]), view), frac


def fold_views(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> Views:
    """Pliega la curva en las cuatro vistas y calcula variables de calidad (nuisance) del plegado."""
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    cadence = float(np.median(np.diff(time)))
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period          # días, tránsito en 0
    n_transit = np.round((time - t0) / period).astype(int)
    in_transit = np.abs(phase) < 0.5 * duration

    views, empties = {}, {}
    raw = {"local": bin_median(phase / duration, flux, -LOCAL_HALF_WIDTH, LOCAL_HALF_WIDTH, N_LOCAL),
           "glob": bin_median(phase / period, flux, -0.5, 0.5, N_GLOBAL)}
    for name, parity in (("odd", 1), ("even", 0)):
        m = n_transit % 2 == parity
        raw[name] = bin_median((phase / period)[m], flux[m], -0.5, 0.5, N_GLOBAL)
    for name, v in raw.items():
        views[name], empties[name] = fill_gaps(v)

    qc = {"n_points": len(time), "span_days": float(np.ptp(time)), "data_days": float(len(time) * cadence),
          "cadence_days": cadence,
          "n_transits": int(np.unique(n_transit[in_transit]).size), "n_points_in_transit": int(in_transit.sum()),
          **{f"frac_empty_{k}": v for k, v in empties.items()}}
    qc["duration_ok"] = bool(duration >= MIN_DURATION_CADENCES * cadence)
    qc["valid"] = bool(qc["duration_ok"] and empties["local"] <= MAX_EMPTY and empties["glob"] <= MAX_EMPTY)
    qc["odd_even_valid"] = bool(empties["odd"] <= MAX_EMPTY and empties["even"] <= MAX_EMPTY)
    if not qc["odd_even_valid"]:      # falta una paridad (pocos tránsitos): las vistas par/impar no sirven
        views["odd"] = views["even"] = np.full(N_GLOBAL, np.nan)
    return Views(views["local"].astype(np.float32), views["glob"].astype(np.float32),
                 views["odd"].astype(np.float32), views["even"].astype(np.float32), qc)
