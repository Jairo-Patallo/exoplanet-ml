"""Folds del protocolo de evaluación anidado (research/03-protocolo-evaluacion.md).

Todo trabaja sobre `meta` (star_id, mission, disposition, split; una fila por estrella) y devuelve posiciones
de fila (np.ndarray), no copias de datos. La misión excluida solo aparece en `eval_idx`.
"""
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from exoplanet.data import MISSIONS, SPLITS, TARGET, load_split


@dataclass
class Fold:
    held_out: str                   # misión excluida
    fit_missions: tuple[str, ...]   # las otras dos
    fit_idx: np.ndarray             # train+val de las misiones de ajuste: reentrenamiento final
    eval_idx: np.ndarray            # la misión excluida entera: se mira una sola vez
    inner: list[tuple[np.ndarray, np.ndarray]]  # (ajuste, validación): train de una misión -> val de la otra


def load_meta() -> pd.DataFrame:
    """star_id, misión, clase, coordenadas y split de todas las estrellas, sin cargar las curvas."""
    cols = ["star_id", "mission", TARGET, "label_source", "period_days", "ra", "dec"]
    return pd.concat([load_split(s, columns=cols).assign(split=s) for s in SPLITS], ignore_index=True)


def shared_stars(meta: pd.DataFrame, radius_arcsec: float = 2.0) -> dict[str, set[str]]:
    """Por misión excluida M: star_id de las otras misiones a menos de `radius_arcsec` de alguna estrella de M.
    Es la misma estrella física observada por dos misiones (p. ej. TESS reobserva campos de Kepler y K2)."""
    meta = meta.dropna(subset=["ra", "dec"])
    sky = SkyCoord(meta.ra.to_numpy() * u.deg, meta.dec.to_numpy() * u.deg)
    shared = {}
    for held_out in MISSIONS:
        in_m = (meta.mission == held_out).to_numpy()
        found = set()
        for other in (m for m in MISSIONS if m != held_out):
            in_o = (meta.mission == other).to_numpy()
            _, sep, _ = sky[in_o].match_to_catalog_sky(sky[in_m])
            found |= set(meta.star_id.to_numpy()[in_o][(sep < radius_arcsec * u.arcsec)])
        shared[held_out] = found
    return shared


def _rows(meta: pd.DataFrame, missions, splits, exclude=frozenset()) -> np.ndarray:
    mask = meta.mission.isin(missions) & meta.split.isin(splits) & ~meta.star_id.isin(exclude)
    return np.flatnonzero(mask.to_numpy())


def nested_folds(meta: pd.DataFrame, shared: dict[str, set[str]] | None = None) -> list[Fold]:
    """Un fold externo por misión excluida. `shared` (salida de `shared_stars`) quita de los conjuntos de ajuste,
    no de la evaluación, las estrellas que la misión excluida también observa."""
    folds = []
    for held_out in MISSIONS:
        exclude = (shared or {}).get(held_out, frozenset())
        a, b = (m for m in MISSIONS if m != held_out)
        inner = [(_rows(meta, [x], ["train"], exclude), _rows(meta, [y], ["val"], exclude)) for x, y in ((a, b), (b, a))]
        folds.append(Fold(held_out, (a, b), _rows(meta, [a, b], ["train", "val"], exclude),
                          _rows(meta, [held_out], SPLITS), inner))
    return folds


def in_domain_split(meta: pd.DataFrame, mission: str) -> tuple[np.ndarray, np.ndarray]:
    """(train+val, test) de una misión: referencia dentro de dominio para medir la brecha."""
    return _rows(meta, [mission], ["train", "val"]), _rows(meta, [mission], ["test"])
