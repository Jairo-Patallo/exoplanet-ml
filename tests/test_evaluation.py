import numpy as np
import pandas as pd
import pytest

from exoplanet.data import DATA, MISSIONS, SPLITS
from exoplanet.evaluation import in_domain_split, load_meta, nested_folds, shared_stars


def fake_meta(n_per_cell: int = 4) -> pd.DataFrame:
    rows = [(f"{m}_{s}_{i}", m, "PLANET" if i % 2 else "FALSE_POSITIVE", s)
            for m in MISSIONS for s in SPLITS for i in range(n_per_cell)]
    return pd.DataFrame(rows, columns=["star_id", "mission", "disposition", "split"])


def check_folds(meta: pd.DataFrame, shared=None) -> None:
    folds = nested_folds(meta, shared)
    assert [f.held_out for f in folds] == list(MISSIONS)
    for f in folds:
        held = meta.mission.to_numpy() == f.held_out
        every_fit = np.concatenate([f.fit_idx] + [i for pair in f.inner for i in pair])
        assert not held[every_fit].any(), "la misión excluida aparece en un conjunto de ajuste o validación interna"
        assert held[f.eval_idx].all() and len(f.eval_idx) == held.sum(), "eval_idx debe ser la misión excluida entera"
        assert not set(f.fit_idx) & set(f.eval_idx)
        assert set(meta.mission[f.fit_idx]) == set(f.fit_missions)
        assert len(f.inner) == 2
        for fit, val in f.inner:
            assert not set(fit) & set(val), "estrellas repetidas entre ajuste y validación interna"
            assert set(meta.split[fit]) == {"train"} and set(meta.split[val]) == {"val"}
            assert len(set(meta.mission[fit])) == 1 and len(set(meta.mission[val])) == 1
            assert set(meta.mission[fit]) != set(meta.mission[val]), "el fold interno debe cruzar misiones"
        assert set(meta.split[f.fit_idx]) == {"train", "val"}, "el test de las misiones de ajuste no se usa"
        assert not set(meta.star_id[f.fit_idx]) & set(meta.star_id[f.eval_idx])


def test_nested_folds_synthetic():
    check_folds(fake_meta())


def test_shared_only_dropped_from_fit_of_its_fold():
    meta = fake_meta()
    shared = {"k2": {"kepler_train_0", "tess_val_1"}}   # solo afecta al fold que excluye K2
    for f in nested_folds(meta, shared):
        dropped = shared.get(f.held_out, set())
        assert not dropped & set(meta.star_id[f.fit_idx])
        assert len(f.eval_idx) == (meta.mission == f.held_out).sum(), "shared no debe tocar la misión excluida"
    other = next(f for f in nested_folds(meta, shared) if f.held_out == "tess")
    assert "kepler_train_0" in set(meta.star_id[other.fit_idx]), "no debe quitarse en los demás folds"


def test_shared_stars_synthetic():
    meta = pd.DataFrame({"star_id": ["k1", "k2", "t1", "t2", "e1"],
                         "mission": ["kepler", "kepler", "tess", "tess", "k2"],
                         "ra": [10.0, 50.0, 10.0 + 1 / 3600, 120.0, 200.0],      # t1 a 1" de k1
                         "dec": [20.0, -10.0, 20.0, 30.0, 5.0]})
    shared = shared_stars(meta)
    assert shared["kepler"] == {"t1"} and shared["tess"] == {"k1"} and shared["k2"] == set()


@pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="faltan los datos en data/")
def test_shared_stars_real_data():
    meta = load_meta()
    shared = shared_stars(meta)
    check_folds(meta, shared)
    for f in nested_folds(meta, shared):
        assert not shared[f.held_out] & set(meta.star_id[f.fit_idx])


def test_in_domain_split():
    meta = fake_meta()
    for mission in MISSIONS:
        fit, test = in_domain_split(meta, mission)
        assert set(meta.mission[fit]) == set(meta.mission[test]) == {mission}
        assert set(meta.split[fit]) == {"train", "val"} and set(meta.split[test]) == {"test"}


@pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="faltan los datos en data/")
def test_nested_folds_real_data():
    meta = load_meta()
    assert meta.star_id.is_unique
    check_folds(meta)
