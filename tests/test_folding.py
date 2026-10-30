import inspect

import numpy as np

from exoplanet.folding import N_GLOBAL, N_LOCAL, fold_views

P, T0, DUR = 3.7, 1.2, 0.12          # días
CADENCE = 30 / 1440                  # 30 min


def synthetic(depth_odd=0.01, depth_even=0.01, span=100.0, noise=1e-4, seed=0, gap=None):
    rng = np.random.default_rng(seed)
    t = np.arange(0, span, CADENCE)
    if gap:
        t = t[(t < gap[0]) | (t > gap[1])]
    flux = 1 + noise * rng.standard_normal(len(t))
    n = np.round((t - T0) / P).astype(int)
    inside = np.abs(t - (T0 + n * P)) < DUR / 2
    flux[inside] -= np.where(n[inside] % 2 == 1, depth_odd, depth_even)
    return t, flux


def test_transit_is_centered_in_every_view():
    v = fold_views(*synthetic(), P, T0, DUR)
    assert abs(int(np.argmin(v.local)) - N_LOCAL // 2) <= 1          # bin 40
    for view in (v.glob, v.odd, v.even):
        assert abs(int(np.argmin(view)) - N_GLOBAL // 2) <= 1        # bin 100
    assert abs(np.median(v.local[39:42]) - 0.99) < 1e-3
    assert abs(np.median(v.local) - 1.0) < 1e-3 and abs(np.median(v.glob) - 1.0) < 1e-3
    assert v.qc["valid"] and v.qc["odd_even_valid"]
    assert v.qc["n_transits"] == len(np.arange(T0, 100, P))


def test_odd_even_parity():
    v = fold_views(*synthetic(depth_odd=0.01, depth_even=0.03), P, T0, DUR)
    assert abs(np.min(v.odd) - 0.99) < 2e-3 and abs(np.min(v.even) - 0.97) < 2e-3


def test_small_gaps_are_interpolated_and_reported():
    v = fold_views(*synthetic(gap=(40, 44)), P, T0, DUR)             # un hueco de 4 d
    assert v.qc["valid"] and not np.isnan(v.glob).any()


def test_too_many_empty_bins_is_invalid():
    t, y = synthetic(span=20.0)                                       # 5 tránsitos pero el global tiene huecos
    keep = np.abs(((t - T0 + 0.5 * P) % P) - 0.5 * P) < 0.4           # solo puntos cerca del tránsito
    v = fold_views(t[keep], y[keep], P, T0, DUR)
    assert v.qc["frac_empty_glob"] > 0.10 and not v.qc["valid"]


def test_duration_shorter_than_two_cadences_is_invalid():
    v = fold_views(*synthetic(), P, T0, 1.5 * CADENCE)
    assert not v.qc["duration_ok"] and not v.qc["valid"]


def test_missing_parity_invalidates_only_odd_even():
    t, y = synthetic()
    m = np.round((t - T0) / P).astype(int) % 2 == 0                   # se quitan todos los tránsitos impares
    v = fold_views(t[m], y[m], P, T0, DUR)
    assert not v.qc["odd_even_valid"] and np.isnan(v.odd).all() and np.isnan(v.even).all()


def test_fold_views_takes_no_catalog_or_mission_input():
    params = list(inspect.signature(fold_views).parameters)
    assert params == ["time", "flux", "period", "t0", "duration"]
