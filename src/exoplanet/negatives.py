"""Negativos NO_SIGNAL reales (tarea #32): estrellas sin candidato plegadas con una efeméride al azar.

Decisiones (2026-11-01): tantos negativos como positivos en cada misión; en Kepler, las NO_SIGNAL del dataset. En K2 y
TESS, estrellas sin TOI/candidato y fuera del dataset, emparejadas una a una con cada positivo: misma campaña/sector que
el único segmento usado del positivo y magnitud más cercana (el test de atajo detectaba la campaña y el ruido); el
negativo hereda el split de su pareja. Efeméride: pareja (periodo, duración) real sorteada de los positivos de TRAIN
de la misma misión (en K2 y TESS, de la misma campaña/sector) y época al azar dentro de un periodo.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ktwo201367065-c01_llc.fits  /  tess2018206045859-s0001-0000000025155310-0120-s_lc.fits
FILE_PATTERNS = {"k2": re.compile(r"ktwo(\d{9})-c(\d+)"), "tess": re.compile(r"tess\d+-s(\d{4})-(\d{16})-")}


def segment_of(path: Path, mission: str) -> tuple[str, int] | None:
    """(star_id, campaña o sector) a partir del nombre de un FITS de K2 o TESS."""
    m = FILE_PATTERNS[mission].search(Path(path).name)
    if not m:
        return None
    if mission == "k2":
        return f"EPIC_{int(m.group(1))}", int(m.group(2))
    return f"TIC_{int(m.group(2))}", int(m.group(1))


def match_negatives(pos: pd.DataFrame, pool: pd.DataFrame, exclude: set[str], rng: np.random.Generator) -> pd.DataFrame:
    """Un negativo por positivo: estrella de `pool` (star_id, segment, mag) en la MISMA campaña/sector que el segmento
    usado del positivo (pos: star_id, split, segment, mag) y con la magnitud más cercana; sin repetir ni usar `exclude`.
    El negativo hereda el split de su pareja. Positivos sin candidatos en su segmento se quedan sin pareja."""
    pool = pool[~pool.star_id.isin(exclude)].drop_duplicates("star_id")
    used, rows = set(), []
    for p in pos.iloc[rng.permutation(len(pos))].itertuples():
        cand = pool[(pool.segment == p.segment) & ~pool.star_id.isin(used)]
        if cand.empty:
            continue
        dmag = (cand.mag - p.mag).abs()
        best = cand.loc[dmag.idxmin()] if dmag.notna().any() else cand.iloc[int(rng.integers(len(cand)))]
        used.add(best.star_id)
        rows.append({"star_id": best.star_id, "segment": int(p.segment), "mag": best.mag, "split": p.split,
                     "pair_star_id": p.star_id, "dmag": abs(best.mag - p.mag)})
    return pd.DataFrame(rows)


def random_ephemerides(train_pos: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n efemérides: parejas (periodo, duración) reales de los positivos de train y época uniforme en [0, P)."""
    pick = train_pos.iloc[rng.integers(len(train_pos), size=n)]
    period = pick.period_days.to_numpy()
    return pd.DataFrame({"period_days": period, "duration_hrs": pick.duration_hrs.to_numpy(),
                         "epoch_bjd": rng.uniform(0, period)})


def segment_ephemerides(stars: pd.DataFrame, train_pos: pd.DataFrame, rng: np.random.Generator, min_pool: int = 5) -> pd.DataFrame:
    """Como random_ephemerides, pero sorteando entre los positivos de train de la MISMA campaña/sector que cada negativo
    (columna segment en ambos): en K2 el periodo típico cambia con la campaña y, sorteado de toda la misión, la pareja
    (campaña, periodo) delataba a los negativos. Con menos de `min_pool` positivos en el segmento, se usa toda la misión."""
    out = pd.DataFrame(index=stars.index, columns=["period_days", "duration_hrs", "epoch_bjd"], dtype=float)
    for seg, g in stars.groupby("segment"):
        pool = train_pos[train_pos.segment == seg]
        eph = random_ephemerides(pool if len(pool) >= min_pool else train_pos, len(g), rng)
        out.loc[g.index] = eph.to_numpy()
    return out
