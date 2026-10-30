"""Plegado propio sobre las 3 curvas reales de MAST (data/raw_lc), comparado con la vista local del dataset de HF."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet.curves import RAW, read_prepare
from exoplanet.data import DATA
from exoplanet.folding import fold_views

STARS = {"KIC_11446443": "kplr011446443", "EPIC_211089792": "ktwo211089792", "TIC_261136679": "tess2018206045859"}
READY = (DATA / "train.parquet").exists() and all(list(RAW.rglob(f"*{k}*.fits")) for k in STARS.values())


@pytest.mark.skipif(not READY, reason="faltan data/train.parquet o las curvas de data/raw_lc")
@pytest.mark.parametrize("star_id", list(STARS))
def test_real_curve_matches_dataset_local_view_and_is_centered(star_id):
    r = pd.read_parquet(DATA / "train.parquet", filters=[("star_id", "==", star_id)],
                        columns=["period_days", "epoch_bjd", "duration_hrs", "flux_local"]).iloc[0]
    P, t0, T = r.period_days, r.epoch_bjd, r.duration_hrs / 24
    paths = sorted(Path(p) for p in RAW.rglob(f"*{STARS[star_id]}*.fits"))
    v = fold_views(*read_prepare(paths, P, t0, T), P, t0, T)

    ref = np.frombuffer(r.flux_local, np.float32)
    mine, theirs = np.median(v.local[39:42]), np.median(ref[39:42])
    assert abs(mine - theirs) < 0.15 * (1 - theirs), f"profundidad central {mine:.5f} frente a {theirs:.5f} del dataset"
    assert abs(int(np.argmin(v.local)) - 40) <= 3
    assert abs(int(np.argmin(v.glob)) - 100) <= 3, "la vista global debe tener el tránsito en el centro"
    assert v.qc["n_transits"] >= 1
