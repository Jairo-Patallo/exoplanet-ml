"""Folds del protocolo de evaluación anidado y métricas de 3 clases (research/03-protocolo-evaluacion.md, tarea #51).

Todo trabaja sobre `meta` (star_id, mission, disposition, split; una fila por estrella) y devuelve posiciones
de fila (np.ndarray), no copias de datos. La misión excluida solo aparece en `eval_idx`.
"""
from dataclasses import dataclass

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from exoplanet.data import CLASSES, MISSIONS, SPLITS, TARGET, load_split


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


def sample_meta(splits=SPLITS) -> pd.DataFrame:
    """La muestra de la fase 2 (estrellas con vista válida) con coordenadas, para los folds LOMO: ra/dec del dataset de
    partida (positivos y NO_SIGNAL de Kepler) y de negatives.csv (negativos de K2 y TESS)."""
    from exoplanet.views import load_sample     # import local: views no depende de evaluation
    stars = load_sample(splits)
    hf = pd.concat([load_split(s, columns=["star_id", "ra", "dec"]) for s in SPLITS])
    neg = pd.read_csv("results/sample/negatives.csv", usecols=["star_id", "ra", "dec"])
    coords = pd.concat([hf, neg]).drop_duplicates("star_id")
    return stars.merge(coords, on="star_id", how="left")


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


def nested_folds(meta: pd.DataFrame, shared: dict[str, set[str]] | None = None, eval_splits=SPLITS) -> list[Fold]:
    """Un fold externo por misión excluida. `shared` (salida de `shared_stars`) quita de los conjuntos de ajuste,
    no de la evaluación, las estrellas que la misión excluida también observa. `eval_splits`: splits de la misión
    excluida que se evalúan (el protocolo dice todos; sin "test" se puede iterar sin gastar el test)."""
    folds = []
    for held_out in MISSIONS:
        exclude = (shared or {}).get(held_out, frozenset())
        a, b = (m for m in MISSIONS if m != held_out)
        inner = [(_rows(meta, [x], ["train"], exclude), _rows(meta, [y], ["val"], exclude)) for x, y in ((a, b), (b, a))]
        folds.append(Fold(held_out, (a, b), _rows(meta, [a, b], ["train", "val"], exclude),
                          _rows(meta, [held_out], eval_splits), inner))
    return folds


def in_domain_split(meta: pd.DataFrame, mission: str) -> tuple[np.ndarray, np.ndarray]:
    """(train+val, test) de una misión: referencia dentro de dominio para medir la brecha."""
    return _rows(meta, [mission], ["train", "val"]), _rows(meta, [mission], ["test"])


def point_metrics(y: np.ndarray, proba: np.ndarray, classes=CLASSES) -> dict[str, float]:
    """Macro-F1 (principal), F1 por clase, PR-AUC one-vs-rest por clase y su media. La predicción es el argmax de
    `proba` (n, k), cuyas columnas siguen el orden de `classes`; sin ajuste de umbral (decisión #51)."""
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    f1 = f1_score(y, pred, labels=list(classes), average=None, zero_division=0)
    pr = [average_precision_score(y == c, proba[:, i]) if (y == c).any() else np.nan for i, c in enumerate(classes)]
    return {"macro_f1": f1.mean(), **{f"f1_{c}": v for c, v in zip(classes, f1)},
            "pr_auc_macro": np.mean(pr), **{f"pr_auc_{c}": v for c, v in zip(classes, pr)}}


def evaluate_by_mission(mission: np.ndarray, y: np.ndarray, proba: np.ndarray, classes=CLASSES,
                        n_boot: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Métricas por misión y su media (resumen: cada misión pesa igual), con IC 95 % por bootstrap sobre estrellas
    (una fila = una estrella). La media se remuestrea misión a misión con los mismos índices, así su IC es coherente.
    Devuelve (tabla con columnas <métrica>, <métrica>_lo, <métrica>_hi por fila) y la matriz de confusión de cada misión."""
    rng = np.random.default_rng(seed)
    point, boot, conf = {}, {}, {}
    for m in (m for m in MISSIONS if m in set(mission)):
        i = np.flatnonzero(mission == m)
        point[m] = point_metrics(y[i], proba[i], classes)
        boot[m] = pd.DataFrame([point_metrics(y[b], proba[b], classes) for b in rng.choice(i, (n_boot, len(i)))])
        conf[m] = pd.DataFrame(confusion_matrix(y[i], np.asarray(classes)[proba[i].argmax(axis=1)], labels=list(classes)),
                               index=[f"real_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    point["media"] = pd.DataFrame(point).T.mean().to_dict()
    boot["media"] = sum(boot[m] for m in conf) / len(conf)
    rows = []
    for name, p in point.items():
        lo, hi = boot[name].quantile(0.025), boot[name].quantile(0.975)
        rows.append({"mision": name, **{k: v for key in p for k, v in ((key, p[key]), (f"{key}_lo", lo[key]), (f"{key}_hi", hi[key]))}})
    return pd.DataFrame(rows), conf
