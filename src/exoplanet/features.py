"""Diagnósticos físicos medidos en las vistas plegadas, sin catálogo (auditoría 2026-11-04: antes en legacy/features.py).

Vistas (exoplanet.folding): local 81 bins en ±4 duraciones (tránsito en el bin 40, ~10 bins de ancho); global, par e
impar 201 bins en fase [-0,5, 0,5] (tránsito en el bin 100). En la global el tránsito ocupa duty·201 bins, con
duty = duración/periodo: para P = 10 d y T = 3 h son ~2,5 bins. La versión de legacy usaba siempre 11 bins como "dentro
del tránsito", lo que diluía par/impar y secundario en los tránsitos cortos; con `duty` la ventana se ajusta a cada estrella.
"""
import numpy as np
import pandas as pd

LOC_IN = slice(35, 46)             # |i-40| <= 5: dentro del tránsito en la local
LOC_CORE = slice(39, 42)           # 3 bins centrales (fondo del tránsito)
LOC_OUT = np.r_[5:25, 56:76]       # > 1,5 duraciones del centro
GLB_OUT = np.r_[20:70, 131:181]    # lejos del tránsito y de la fase 0,5
N_GLOBAL, CENTER = 201, 100


def robust_sigma(x: np.ndarray) -> np.ndarray:
    """σ ≈ 1,4826·MAD por fila: ruido gaussiano robusto a outliers y al propio tránsito."""
    med = np.median(x, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=1)


def _band(center: int, half: np.ndarray) -> np.ndarray:
    """Máscara (n, 201) de los bins a <= half[i] del bin `center` (con vuelta en los extremos para la fase 0,5)."""
    d = np.abs(np.arange(N_GLOBAL)[None, :] - center)
    d = np.minimum(d, N_GLOBAL - d)
    return d <= half[:, None]


def _masked_median(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.nanmedian(np.where(mask, v, np.nan), axis=1)


def curve_features(local: np.ndarray, odd: np.ndarray, even: np.ndarray, glob: np.ndarray,
                   duty: np.ndarray | None = None, mask_no_dip: bool = True) -> pd.DataFrame:
    """Profundidad, SNR y forma (local), diferencia par/impar y eclipse secundario (global), en unidades de error.
    duty (n,) = duración/periodo: ventana del tránsito en la global de cada estrella (al menos 1 bin a cada lado).
    Sin duty se usa la ventana fija de 11 bins de la versión antigua (para reproducir resultados previos).
    mask_no_dip=False conserva profundidad y SNR con signo cuando no hay caída (δ <= 0) y deja NaN solo ingress_frac."""
    np.seterr(divide="ignore", invalid="ignore")
    n = len(local)
    half = np.full(n, 5) if duty is None else np.maximum(1, np.ceil(np.asarray(duty) * N_GLOBAL / 2)).astype(int)
    in_tr = _band(CENTER, half)
    if duty is None:                 # ventana fija antigua: fase |φ| > 0,45 (bins 0-9 y 191-200)
        sec = np.zeros((n, N_GLOBAL), bool)
        sec[:, np.r_[0:10, 191:201]] = True
    else:
        sec = _band(0, half)
    sig_loc = robust_sigma(local[:, LOC_OUT])
    depth = 1 - np.median(local[:, LOC_CORE], axis=1)
    n_in = LOC_IN.stop - LOC_IN.start
    feats = {"depth_local_ppm": depth * 1e6,
             "snr_local": depth / (sig_loc / np.sqrt(n_in)),
             "ingress_frac": 1 - (1 - local[:, LOC_IN]).mean(axis=1) / depth}   # caja (U) -> 0; triángulo (V) -> 0,5
    n_tr, n_sec = in_tr.sum(axis=1), sec.sum(axis=1)
    d_odd, d_even = (1 - _masked_median(v, in_tr) for v in (odd, even))
    se_odd, se_even = (1.2533 * robust_sigma(v[:, GLB_OUT]) / np.sqrt(n_tr) for v in (odd, even))
    feats["odd_even_z"] = np.abs(d_odd - d_even) / np.sqrt(se_odd ** 2 + se_even ** 2)
    base = np.median(glob[:, GLB_OUT], axis=1)
    feats["secondary_z"] = (base - _masked_median(glob, sec)) / (1.2533 * robust_sigma(glob[:, GLB_OUT]) / np.sqrt(n_sec))
    out = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    if mask_no_dip:
        out.loc[~(depth > 0)] = np.nan       # sin caída medible las variables no tienen sentido físico
    else:
        out.loc[~(depth > 0), "ingress_frac"] = np.nan
    return out
