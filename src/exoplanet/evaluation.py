"""Folds del protocolo de evaluación anidado (research/03-protocolo-evaluacion.md).

Todo trabaja sobre `meta` (star_id, mission, disposition, split; una fila por estrella) y devuelve posiciones
de fila (np.ndarray), no copias de datos. La misión excluida solo aparece en `eval_idx`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exoplanet.data import MISSIONS, SPLITS, TARGET, load_split


@dataclass
class Fold:
    held_out: str                   # misión excluida
    fit_missions: tuple[str, ...]   # las otras dos
    fit_idx: np.ndarray             # train+val de las misiones de ajuste: reentrenamiento final
    eval_idx: np.ndarray            # la misión excluida entera: se mira una sola vez
    inner: list[tuple[np.ndarray, np.ndarray]]  # (ajuste, validación): train de una misión -> val de la otra


def load_meta() -> pd.DataFrame:
    """star_id, misión, clase y split de todas las estrellas, sin cargar las curvas."""
    return pd.concat([load_split(s, columns=["star_id", "mission", TARGET]).assign(split=s) for s in SPLITS],
                     ignore_index=True)


def _rows(meta: pd.DataFrame, missions, splits, exclude=frozenset()) -> np.ndarray:
    mask = meta.mission.isin(missions) & meta.split.isin(splits) & ~meta.star_id.isin(exclude)
    return np.flatnonzero(mask.to_numpy())


def nested_folds(meta: pd.DataFrame, exclude=frozenset()) -> list[Fold]:
    """Un fold externo por misión excluida. `exclude` (star_id) se quita solo de los conjuntos de ajuste,
    p. ej. las estrellas observadas también por la misión excluida (tarea #36)."""
    folds = []
    for held_out in MISSIONS:
        a, b = (m for m in MISSIONS if m != held_out)
        inner = [(_rows(meta, [x], ["train"], exclude), _rows(meta, [y], ["val"], exclude)) for x, y in ((a, b), (b, a))]
        folds.append(Fold(held_out, (a, b), _rows(meta, [a, b], ["train", "val"], exclude),
                          _rows(meta, [held_out], SPLITS), inner))
    return folds


def in_domain_split(meta: pd.DataFrame, mission: str) -> tuple[np.ndarray, np.ndarray]:
    """(train+val, test) de una misión: referencia dentro de dominio para medir la brecha."""
    return _rows(meta, [mission], ["train", "val"]), _rows(meta, [mission], ["test"])
