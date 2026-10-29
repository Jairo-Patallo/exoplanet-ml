import pandas as pd
import pytest

from exoplanet import data

pytestmark = pytest.mark.skipif(not (data.DATA / "train.parquet").exists(), reason="faltan los parquet en data/")
COLS = ["star_id"]


@pytest.fixture(scope="module")
def folds():
    return list(data.lomo_folds(COLS))


def test_each_mission_held_out_once(folds):
    assert sorted(m for m, *_ in folds) == sorted(data.MISSIONS)


def test_missions_and_classes(folds):
    for held_out, fit, tune, dev in folds:
        assert held_out not in set(fit.mission) | set(tune.mission)
        assert set(dev.mission) == {held_out}
        for df in (fit, tune, dev):
            assert set(df[data.TARGET]) == {"PLANET", "FALSE_POSITIVE"}
            assert "flux_global" not in df.columns


def test_no_shared_stars(folds):
    for _, fit, tune, dev in folds:
        assert not set(fit.star_id) & set(tune.star_id)
        assert not set(fit.star_id) & set(dev.star_id)
        assert not set(tune.star_id) & set(dev.star_id)


def test_test_split_never_read(monkeypatch):
    read = []
    original = pd.read_parquet
    monkeypatch.setattr(data.pd, "read_parquet", lambda path, **kw: read.append(path.name) or original(path, **kw))
    list(data.lomo_folds(COLS))
    assert read and "test.parquet" not in read
