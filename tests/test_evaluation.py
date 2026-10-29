import numpy as np
import pandas as pd
import pytest

from exoplanet.data import DATA, MISSIONS, SPLITS
from exoplanet.evaluation import in_domain_split, load_meta, nested_folds


def fake_meta(n_per_cell: int = 4) -> pd.DataFrame:
    rows = [(f"{m}_{s}_{i}", m, "PLANET" if i % 2 else "FALSE_POSITIVE", s)
            for m in MISSIONS for s in SPLITS for i in range(n_per_cell)]
    return pd.DataFrame(rows, columns=["star_id", "mission", "disposition", "split"])


def check_folds(meta: pd.DataFrame, exclude=frozenset()) -> None:
    folds = nested_folds(meta, exclude)
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


def test_exclude_only_from_fit():
    meta = fake_meta()
    drop = {"kepler_train_0", "tess_val_1"}
    for f in nested_folds(meta, drop):
        assert not drop & set(meta.star_id[f.fit_idx])
        assert len(f.eval_idx) == (meta.mission == f.held_out).sum(), "exclude no debe tocar la misión excluida"


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
