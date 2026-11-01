from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet.negatives import match_negatives, random_ephemerides, segment_ephemerides, segment_of


def test_segment_of_parses_k2_and_tess_names():
    assert segment_of(Path("ktwo201367065-c01_llc.fits"), "k2") == ("EPIC_201367065", 1)
    name = "tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"
    assert segment_of(Path(name), "tess") == ("TIC_25155310", 1)
    assert segment_of(Path("kplr006048255-2011177032512_llc.fits"), "k2") is None


def test_match_negatives_same_segment_nearest_magnitude():
    pos = pd.DataFrame({"star_id": ["P1", "P2"], "split": ["train", "test"], "segment": [1, 2], "mag": [10.0, 12.0]})
    pool = pd.DataFrame({"star_id": ["A", "B", "C", "D", "X"], "segment": [1, 1, 2, 2, 1],
                         "mag": [9.0, 10.2, 12.1, 10.0, 10.0]})
    out = match_negatives(pos, pool, exclude={"X"}, rng=np.random.default_rng(0)).set_index("pair_star_id")
    assert out.loc["P1", "star_id"] == "B" and out.loc["P1", "split"] == "train"   # X está excluida
    assert out.loc["P2", "star_id"] == "C" and out.loc["P2", "segment"] == 2         # D es de otro brillo


def test_match_negatives_never_reuses_a_star():
    pos = pd.DataFrame({"star_id": ["P1", "P2", "P3"], "split": "train", "segment": 1, "mag": 10.0})
    pool = pd.DataFrame({"star_id": ["A", "B"], "segment": 1, "mag": [10.0, 11.0]})
    out = match_negatives(pos, pool, exclude=set(), rng=np.random.default_rng(0))
    assert len(out) == 2 and out.star_id.is_unique


def test_random_ephemerides_keep_real_pairs_and_epoch_within_period():
    pos = pd.DataFrame({"period_days": [1.0, 10.0, 100.0], "duration_hrs": [1.0, 3.0, 9.0]})
    eph = random_ephemerides(pos, 500, np.random.default_rng(0))
    pairs = set(zip(pos.period_days, pos.duration_hrs))
    assert set(zip(eph.period_days, eph.duration_hrs)) <= pairs
    assert ((eph.epoch_bjd >= 0) & (eph.epoch_bjd < eph.period_days)).all()


def test_segment_ephemerides_use_positives_of_the_same_segment():
    train = pd.DataFrame({"segment": [1] * 5 + [2] * 5 + [3], "period_days": [1.0] * 5 + [20.0] * 5 + [99.0],
                          "duration_hrs": 2.0})
    stars = pd.DataFrame({"segment": [1, 2, 2, 3]})
    eph = segment_ephemerides(stars, train, np.random.default_rng(0))
    assert list(eph.period_days[:3]) == [1.0, 20.0, 20.0]
    assert eph.period_days.iloc[3] in {1.0, 20.0, 99.0}            # segmento 3 con 1 positivo: toda la misión
