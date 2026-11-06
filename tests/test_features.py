import numpy as np

from exoplanet.features import curve_features
from exoplanet.folding import N_GLOBAL, N_LOCAL, fold_centroid


def box_views(n=40, depth=1e-3, odd_factor=1.0, half=2, noise=1e-4, seed=0):
    """Vistas sintéticas: tránsito en caja de semiancho `half` bins en la global; la impar más profunda si odd_factor > 1."""
    rng = np.random.default_rng(seed)
    local = 1 + rng.normal(0, noise, (n, N_LOCAL))
    local[:, 35:46] -= depth
    glob, odd, even = (1 + rng.normal(0, noise, (n, N_GLOBAL)) for _ in range(3))
    sl = slice(100 - half, 100 + half + 1)
    glob[:, sl] -= depth
    even[:, sl] -= depth
    odd[:, sl] -= depth * odd_factor
    return local, odd, even, glob


def test_duty_window_finds_odd_even_difference_in_short_transits():
    views = box_views(odd_factor=2.0, half=2)
    duty = np.full(40, 5 / N_GLOBAL)                      # el tránsito ocupa 5 bins de la global
    z_fixed = curve_features(*views)["odd_even_z"].median()
    z_duty = curve_features(*views, duty=duty)["odd_even_z"].median()
    assert z_duty > 2 * z_fixed, "la ventana ajustada a la duración debe ver mejor la diferencia par/impar"
    assert curve_features(*box_views(odd_factor=1.0), duty=duty)["odd_even_z"].median() < 3


def test_no_dip_keeps_signed_depth_when_not_masked():
    local, odd, even, glob = box_views(depth=-1e-3)        # "caída" negativa: no hay tránsito
    f = curve_features(local, odd, even, glob, mask_no_dip=False)
    assert (f.depth_local_ppm < 0).all() and f.ingress_frac.isna().all()
    assert curve_features(local, odd, even, glob).isna().all().all()


def test_fold_centroid_detects_shift_in_transit():
    rng = np.random.default_rng(1)
    period, t0, dur = 3.0137, 0.5, 0.15            # periodo no múltiplo de la cadencia, como en una estrella real
    t = np.arange(0, 90, 0.02)
    phase = ((t - t0 + period / 2) % period) - period / 2
    in_tr = np.abs(phase) < dur / 2
    dx, dy = rng.normal(0, 1e-3, len(t)), rng.normal(0, 1e-3, len(t))
    loc_quiet, _, ok = fold_centroid(t, dx, dy, period, t0, dur)
    dx_shift = dx + np.where(in_tr, 5e-3, 0)
    loc_shift, _, _ = fold_centroid(t, dx_shift, dy, period, t0, dur)
    assert ok
    assert loc_shift[40] > 10 and abs(loc_quiet[40]) < 5, "un desplazamiento en el tránsito debe dar muchos sigmas"


def box_curve(odd_depth=2e-3, even_depth=2e-3, noise=1e-3, red=0.0, seed=0):
    """Un sector de TESS sintético (27 d a 2 min) con tránsitos en caja; red > 0 añade ruido correlado de ~1 duración."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 27, 2 / 1440)
    period, t0, dur = 2.3, 1.0, 0.12
    y = 1 + rng.normal(0, noise, t.size)
    if red:
        k = int(dur / (2 / 1440))
        y += np.convolve(rng.normal(0, red, t.size), np.ones(k) / np.sqrt(k), mode="same")
    n = np.round((t - t0) / period)
    inside = np.abs(t - t0 - n * period) < dur / 2
    y[inside] -= np.where(n[inside] % 2 == 1, odd_depth, even_depth)
    return t, y, period, t0, dur


def test_transit_features_planet_is_calibrated():
    from exoplanet.features import transit_features
    f = transit_features(*box_curve())
    assert f["n_transits"] >= 10 and 1.0 <= f["beta"] < 1.3
    assert abs(f["z_odd_even"]) < 3 and f["z_scatter"] < 3 and f["z_max"] < 3


def test_transit_features_finds_odd_even_binary():
    from exoplanet.features import transit_features
    assert transit_features(*box_curve(odd_depth=4e-3, even_depth=1e-3))["z_odd_even"] > 5


def test_transit_features_red_noise_inflates_errors():
    from exoplanet.features import transit_features
    assert transit_features(*box_curve(red=1e-3))["beta"] > 1.5
