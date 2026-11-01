import pandas as pd

from exoplanet.labels import STRENGTH, label_strength


def frame(rows):
    return pd.DataFrame(rows, columns=["star_id", "mission", "disposition", "label_source", "period_days"])


TOI = pd.DataFrame({"tid": [1, 1, 2, 3, 4], "tfopwg_disp": ["CP", "FP", "PC", "APC", "FA"],
                    "pl_orbper": [3.0, 10.0, 5.0, 2.0, 1.0]})
K2 = pd.DataFrame({"epic_hostname": ["EPIC 11", "EPIC 12", "EPIC 12"], "disposition": ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"],
                   "pl_orbper": [4.0, 6.0, 20.0]})


def test_strength_per_mission():
    df = frame([
        ("KIC_1", "kepler", "PLANET", "kepler_confirmed", 1.0),
        ("KIC_2", "kepler", "PLANET", "kepler_robovetter", 1.0),
        ("KIC_3", "kepler", "FALSE_POSITIVE", "kepler_robovetter", 1.0),
        ("KIC_4", "kepler", "FALSE_POSITIVE", "kepler_confirmed", 1.0),     # contradictorio
        ("TIC_1", "tess", "PLANET", "tess_tfopwg", 3.1),                    # elige el TOI de periodo 3.0 -> CP
        ("TIC_2", "tess", "PLANET", "tess_tfopwg", 5.0),                    # PC
        ("TIC_3", "tess", "PLANET", "tess_tfopwg", 2.0),                    # APC
        ("TIC_4", "tess", "FALSE_POSITIVE", "tess_tfopwg", 1.0),            # FA, no FP
        ("TIC_9", "tess", "PLANET", "tess_tfopwg", 1.0),                    # sin TOI
        ("EPIC_11", "k2", "PLANET", "k2_project", 4.0),
        ("EPIC_12", "k2", "PLANET", "k2_project", 6.2),                     # CANDIDATE
        ("EPIC_12", "k2", "FALSE_POSITIVE", "k2_project", 19.0),            # FALSE POSITIVE
    ])
    got = label_strength(df, toi=TOI, k2=K2).tolist()
    assert got == ["fuerte", "debil", "fuerte", "excluida", "fuerte", "debil", "excluida", "excluida", "excluida",
                   "fuerte", "debil", "fuerte"]


def test_unlisted_combinations_are_excluded():
    df = frame([("KIC_1", "kepler", "NO_SIGNAL", "x", float("nan"))])
    assert label_strength(df).tolist() == ["excluida"]


def test_mapping_has_no_strong_label_that_disagrees_with_class():
    for (_, cls, _), strength in STRENGTH.items():
        assert cls in ("PLANET", "FALSE_POSITIVE") and strength in ("fuerte", "debil")
