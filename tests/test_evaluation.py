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


def test_point_metrics_match_sklearn_and_perfect_model():
    from sklearn.metrics import average_precision_score, f1_score
    from exoplanet.data import CLASSES
    from exoplanet.evaluation import point_metrics
    rng = np.random.default_rng(0)
    y = rng.choice(CLASSES, 300)
    proba = rng.dirichlet(np.ones(3), 300)
    m = point_metrics(y, proba)
    pred = np.asarray(CLASSES)[proba.argmax(1)]
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(m["pr_auc_FALSE_POSITIVE"], average_precision_score(y == "FALSE_POSITIVE", proba[:, 1]))
    perfect = point_metrics(y, np.stack([(y == c).astype(float) for c in CLASSES], axis=1))
    assert perfect["macro_f1"] == 1 and perfect["pr_auc_macro"] == 1


def test_evaluate_by_mission_mean_and_ci():
    from exoplanet.data import CLASSES
    from exoplanet.evaluation import evaluate_by_mission
    rng = np.random.default_rng(1)
    mission = np.repeat(list(MISSIONS), 100)
    y = rng.choice(CLASSES, 300)
    proba = rng.dirichlet(np.ones(3), 300)
    table, conf = evaluate_by_mission(mission, y, proba, n_boot=200)
    t = table.set_index("mision")
    assert list(t.index) == [*MISSIONS, "media"]
    assert np.isclose(t.loc["media", "macro_f1"], t.loc[list(MISSIONS), "macro_f1"].mean()), "la media pesa igual cada misión"
    assert (t.macro_f1_lo <= t.macro_f1).all() and (t.macro_f1 <= t.macro_f1_hi).all()
    assert all(c.to_numpy().sum() == 100 for c in conf.values())
