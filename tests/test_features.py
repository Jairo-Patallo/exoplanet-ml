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
