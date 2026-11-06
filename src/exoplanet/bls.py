"""Efeméride a partir de la curva, sin catálogo: aplanado a ciegas + Box Least Squares (fase 4, #53, 2026-11-05).

1. Aplanado a ciegas: no sabemos dónde están los tránsitos, así que no se pueden enmascarar. Se usa una ventana larga
   (BLIND_WINDOW_DAYS) para que un tránsito de unas horas apenas se deforme, y se recortan los picos hacia arriba.
2. BLS (astropy, Kovács et al. 2002) de MIN_PERIOD a la mitad de la duración de los datos (>= 2 tránsitos):
   - por bandas de periodo, cada una con duraciones de caja razonables (una caja de 12 h no cabe en 0,5 d);
   - rejilla gruesa (resolución de SPAN_CAP = 60 días de datos) y refinado fino alrededor de los mejores picos, para que dos
     trimestres de Kepler separados por años no den rejillas de millones de periodos;
   - el periodograma se aplana con una mediana móvil antes de elegir el pico: con periodos cortos caben más cajas y la
     verosimilitud sube sola (sin esto, BLS elige siempre el periodo mínimo); la tendencia se calcula por tramos de
     log(P), no por número de puntos.
Después, curves.prepare con la efeméride encontrada vuelve a aplanar enmascarando el tránsito, igual que con catálogo.
"""
from dataclasses import dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares

MIN_PERIOD = 0.5                   # días (decisión 2026-11-05)
BLIND_WINDOW_DAYS = 1.5            # > 3 duraciones para tránsitos de hasta ~12 h
BANDS = [(0.5, 2.0, np.array([1, 1.5, 2, 3]) / 24),             # (P mín, P máx, duraciones en días)
         (2.0, 10.0, np.array([1, 2, 3, 4.5, 6]) / 24),
         (10.0, np.inf, np.array([2, 3, 4.5, 6, 9, 12]) / 24)]
SPAN_CAP = 60.0                    # días: resolución de la rejilla gruesa (más fina en el refinado)
OVERSAMPLE = 3
TREND_DEX = 0.05                   # anchura de los tramos de log10(P) para la tendencia del periodograma (12 %)


@dataclass
class Ephemeris:
    period: float
    t0: float
    duration_hrs: float
    depth: float
    sde: float                     # pico del periodograma aplanado en unidades de su dispersión (significancia)
    n_periods: int


def flatten_blind(lc) -> tuple[np.ndarray, np.ndarray]:
    """(tiempo, flujo) aplanado sin máscara de tránsito y sin picos hacia arriba (> 3 sigma)."""
    lc = lc.remove_nans().normalize()
    step = float(np.median(np.diff(lc.time.value)))
    window = int(BLIND_WINDOW_DAYS / step // 2 * 2 + 1)
    flat = lc.flatten(window_length=min(window, (len(lc) // 2) * 2 - 1))
    t, y = flat.time.value, flat.flux.value
    mad = 1.4826 * np.median(np.abs(y - np.median(y)))
    keep = np.isfinite(y) & (y < np.median(y) + 3 * mad)
    return t[keep], y[keep]


def _grid(p_min: float, p_max: float, dur_min: float, span: float) -> np.ndarray:
    df = dur_min / span ** 2 / OVERSAMPLE
    return np.sort(1 / np.arange(1 / p_max, 1 / p_min, df))


def trend(periods: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Tendencia del periodograma: mediana por tramos de TREND_DEX en log10(periodo), interpolada.
    En tramos de log(P) y no de número de puntos: la rejilla es muy densa en periodos cortos y una ventana de puntos
    fija cabe dentro de un solo pico, que la mediana se comería (pasaba en K2: elegía múltiplos x7, x9)."""
    logp = np.log10(periods)
    edges = np.arange(logp.min(), logp.max() + TREND_DEX, TREND_DEX)
    idx = np.digitize(logp, edges)
    centers, meds = [], []
    for k in np.unique(idx):
        sel = idx == k
        centers.append(logp[sel].mean()), meds.append(np.median(power[sel]))
    return np.interp(logp, centers, meds)


def find_ephemeris(t: np.ndarray, y: np.ndarray) -> Ephemeris:
    """Pico más significativo del periodograma BLS de una curva ya aplanada (tiempo en días)."""
    return find_ephemerides(t, y, 1)[0]


def find_ephemerides(t: np.ndarray, y: np.ndarray, k: int = 1, min_sep: float = 0.01) -> list[Ephemeris]:
    """Los k picos más significativos del periodograma, distintos entre sí (periodos separados más de min_sep en
    proporción; los alias P/2, 2P... cuentan como picos distintos), ordenados por SDE. El primero es find_ephemeris."""
    bls = BoxLeastSquares(t, y)
    data_days = len(t) * float(np.median(np.diff(t)))
    p_max_all, span = data_days / 2, min(float(np.ptp(t)), SPAN_CAP)
    periods, power, durs = [], [], {}
    for lo, hi, d in BANDS:
        hi = min(hi, p_max_all)
        if hi <= lo:
            continue
        grid = _grid(lo, hi, d.min(), span)
        res = bls.power(grid, d, objective="likelihood")
        periods.append(grid), power.append(np.asarray(res.power))
        durs.update({float(p): d for p in grid})
    periods, power = np.concatenate(periods), np.concatenate(power)
    flat = power - trend(periods, power)
    sde = (flat - np.median(flat)) / (1.4826 * np.median(np.abs(flat - np.median(flat))) + 1e-12)

    peaks = []                     # índices de los k picos distintos de mayor SDE
    for i in np.argsort(sde)[::-1]:
        if all(abs(periods[i] / periods[j] - 1) > min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) == k:
            break
    out, base = [], float(np.ptp(t))   # refinado fino (resolución de toda la base de tiempo) alrededor de cada pico
    for i in peaks:
        p, d = float(periods[i]), durs[float(periods[i])]
        df = d.min() / base ** 2 / OVERSAMPLE
        step = d.min() / span ** 2 / OVERSAMPLE * 2
        f = np.arange(1 / p - step, 1 / p + step, df)
        res = bls.power(1 / f[f > 0], d, objective="likelihood")
        j = int(np.argmax(res.power))
        out.append(Ephemeris(float(res.period[j]), float(res.transit_time[j]), float(res.duration[j]) * 24,
                             float(res.depth[j]), float(sde[i]), len(periods)))
    return out


def match(p_found: float, p_true: float, tol: float = 0.01) -> str:
    """Relación entre el periodo encontrado y el del catálogo: 'igual', alias '1/2', '2', '1/3', '3', 'otro armónico' (k o 1/k, k = 4..10) u 'otro'."""
    for name, k in (("igual", 1), ("1/2", 0.5), ("2", 2), ("1/3", 1 / 3), ("3", 3)):
        if abs(p_found / (k * p_true) - 1) < tol:
            return name
    for k in range(4, 11):         # armónicos más lejanos
        if abs(p_found / (k * p_true) - 1) < tol or abs(p_found * k / p_true - 1) < tol:
            return "otro armónico"
    return "otro"
