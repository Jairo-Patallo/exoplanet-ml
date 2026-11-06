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


FAKE_PHASES = (0.15, 0.25, 0.35, 0.65, 0.75, 0.85)    # "tránsitos falsos": lejos del tránsito (0) y del secundario (0,5)
HALF_WIN, N_TEMPLATE = 1.5, 30                        # plantilla del tránsito: ±1,5 duraciones en bins de 0,1 duraciones


def _amplitudes(phase, epoch, dip, duration, template=None) -> np.ndarray:
    """(n_épocas, 2): amplitud de la plantilla en cada tránsito y su error, ajustando dip = c + a·plantilla por mínimos
    cuadrados en ±HALF_WIN duraciones. Sin `template`, la de cada tránsito es la media de los demás (deja uno fuera),
    para que el tránsito no se compare consigo mismo. a ≈ 1 si el tránsito es igual a la media."""
    b = np.clip(((phase / duration + HALF_WIN) / (2 * HALF_WIN) * N_TEMPLATE).astype(int), 0, N_TEMPLATE - 1)
    S, C = np.bincount(b, dip, N_TEMPLATE), np.bincount(b, minlength=N_TEMPLATE)
    out = []
    for e in np.unique(epoch):
        m = epoch == e
        if template is None:
            Se, Ce = np.bincount(b[m], dip[m], N_TEMPLATE), np.bincount(b[m], minlength=N_TEMPLATE)
            with np.errstate(invalid="ignore", divide="ignore"):
                tmpl = (S - Se) / (C - Ce)
        else:
            tmpl = template
        x, y = tmpl[b[m]], dip[m]
        ok = np.isfinite(x)
        core = ok & (np.abs(phase[m]) < 0.5 * duration)
        if ok.sum() < 8 or core.sum() < 2:
            out.append((np.nan, np.nan))
            continue
        X = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        resid = y[ok] - X @ coef
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        cov = np.linalg.pinv(X.T @ X) * sigma ** 2
        out.append((coef[1], np.sqrt(cov[1, 1])))
    return np.array(out).reshape(-1, 2)


def transit_features(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float) -> dict:
    """Información de los tránsitos individuales que la mediana plegada borra (propuesta P1, paso 1; 2026-11-06).

    A cada tránsito se le ajusta la plantilla plegada de los demás (forma fija, amplitud libre), así que una forma en V
    o la cadencia no crean diferencias falsas. El ruido real (rojo) se mide ajustando la misma plantilla en "tránsitos
    falsos" (FAKE_PHASES de cada órbita): beta = dispersión robusta de sus amplitudes en unidades de su error (1 si el
    ruido fuera blanco; idea del block bootstrap de #9). Con el error corregido (error · beta):
      z_odd_even: amplitud media impar − par (binaria con el periodo a la mitad);
      z_scatter:  χ² de las amplitudes frente a su media (mezclas, sistemáticos), pasado a z;
      z_max:      el tránsito más desviado, pasado a z teniendo en cuenta cuántos hay.
    Los tres valen ~N(0, 1) en un planeta con ruido bien medido, sea cual sea el número de tránsitos: así no delatan
    ese número, que es un metadato del proceso (n_transits se devuelve solo para estratificar)."""
    from scipy.stats import chi2, norm
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    dip = 1 - flux
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    clean = (np.abs(phase) > duration) & (np.abs(np.abs(phase) - 0.5 * period) > duration)
    out = {"n_transits": 0, "n_fakes": 0, "beta": np.nan, "snr": np.nan,
           "z_odd_even": np.nan, "z_scatter": np.nan, "z_max": np.nan}
    win = np.abs(phase) < HALF_WIN * duration
    epoch = np.round((time - t0) / period).astype(int)
    real = _amplitudes(phase[win], epoch[win], dip[win], duration)
    epochs = np.unique(epoch[win])
    ok = np.isfinite(real[:, 0])
    out["n_transits"] = int(ok.sum())
    b = np.clip(((phase[win] / duration + HALF_WIN) / (2 * HALF_WIN) * N_TEMPLATE).astype(int), 0, N_TEMPLATE - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        template = np.bincount(b, dip[win], N_TEMPLATE) / np.bincount(b, minlength=N_TEMPLATE)
    fakes = []
    for f in FAKE_PHASES:
        ph = ((time - t0 - f * period + 0.5 * period) % period) - 0.5 * period
        m = (np.abs(ph) < HALF_WIN * duration) & clean
        if m.any():
            fakes.append(_amplitudes(ph[m], np.round((time[m] - t0 - f * period) / period).astype(int), dip[m],
                                     duration, template))
    fakes = np.concatenate(fakes) if fakes else np.empty((0, 2))
    fakes = fakes[np.isfinite(fakes[:, 0]) & (fakes[:, 1] > 0)]
    out["n_fakes"] = len(fakes)
    if len(fakes) >= 6:
        r = fakes[:, 0] / fakes[:, 1]
        out["beta"] = max(1.0, 1.4826 * np.median(np.abs(r - np.median(r))))
    if ok.sum() < 2 or not np.isfinite(out["beta"]):
        return out
    d, s, ep = real[ok, 0], real[ok, 1] * out["beta"], epochs[ok]
    w = 1 / s ** 2
    mean = np.sum(w * d) / w.sum()
    out["snr"] = mean * np.sqrt(w.sum())
    k = len(d)
    out["z_scatter"] = float(norm.isf(np.clip(chi2.sf(np.sum(((d - mean) / s) ** 2), k - 1), 1e-300, 1 - 1e-15)))
    p_max = 1 - (2 * norm.cdf(np.max(np.abs(d - mean) / s)) - 1) ** k        # P(el máximo de k |N(0,1)| sea mayor)
    out["z_max"] = float(norm.isf(np.clip(p_max, 1e-300, 1 - 1e-15)))
    odd = ep % 2 == 1
    if odd.any() and (~odd).any():
        m_o, m_e = (np.sum(w[m] * d[m]) / w[m].sum() for m in (odd, ~odd))
        out["z_odd_even"] = (m_o - m_e) / np.sqrt(1 / w[odd].sum() + 1 / w[~odd].sum())
    return out
