"""Curvas de MAST (Kepler, K2, TESS): selección de segmentos, descarga, limpieza y aplanado (tarea #30).

Decisiones aprobadas: flujo PDCSAP con la máscara de calidad por defecto (D1); aplanado de lightkurve con el tránsito
enmascarado (D2); tope de ~4 meses de datos por estrella, eligiendo los segmentos con más tránsitos (D3);
normalización por la mediana fuera de tránsito (D8). El plegado en sí está en `exoplanet.folding`.
"""
from functools import lru_cache
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet.data import DATA

RAW = DATA / "raw_lc"
MAX_DAYS = 100.0            # D3: se acumulan segmentos hasta superar este tiempo
MASK_WIDTH = 3.0            # D2: el tránsito enmascarado mide 3 duraciones
MIN_WINDOW_DAYS = 0.5       # D2: ventana de aplanado >= max(3 duraciones, 0.5 d)
MJD_OFFSET = {"kepler": -54832.5, "k2": -54832.5, "tess": -56999.5}   # tiempo de la misión = MJD + offset (BKJD, BTJD)
SEARCH = {"kepler": dict(author="Kepler", cadence="long"), "k2": dict(author="K2", cadence="long"),
          "tess": dict(author="SPOC", exptime=120)}


def expected_transits(tmin: float, tmax: float, period: float, t0: float) -> int:
    """Número de tránsitos (t0 + n·P) dentro de [tmin, tmax]."""
    return max(0, int(np.floor((tmax - t0) / period) - np.ceil((tmin - t0) / period) + 1))


def select_segments(spans: list[tuple[float, float]], period: float, t0: float, max_days: float = MAX_DAYS) -> list[int]:
    """Índices de los segmentos elegidos: los de más tránsitos esperados (a igualdad, los más largos),
    acumulando hasta superar `max_days`. `spans` y `t0` en el mismo sistema de tiempo."""
    order = sorted(range(len(spans)), key=lambda i: (-expected_transits(*spans[i], period, t0), -(spans[i][1] - spans[i][0])))
    chosen, total = [], 0.0
    for i in order:
        if total > max_days:
            break
        chosen.append(i)
        total += spans[i][1] - spans[i][0]
    return sorted(chosen)


@lru_cache(maxsize=1)
def kepler_quarters() -> pd.DataFrame:
    """Intervalo de cada trimestre de Kepler. MAST da el mismo intervalo (toda la misión) para todos los trimestres,
    así que se usa la tabla medida en las cabeceras de los archivos (scripts/build_kepler_quarters.py)."""
    return pd.read_csv(Path(__file__).with_name("kepler_quarters.csv")).set_index("quarter")


def kepler_span(label: str) -> tuple[float, float]:
    """(inicio, fin) en BKJD de un trimestre, a partir de su etiqueta 'Kepler Quarter NN'."""
    row = kepler_quarters().loc[int(label.split()[-1])]
    return float(row.tstart_bkjd), float(row.tstop_bkjd)


def product_spans(table, mission: str) -> list[tuple[float, float]]:
    """(inicio, fin) de cada producto de una búsqueda, en el tiempo de la misión (el mismo que la época del dataset)."""
    if mission == "kepler":
        return [kepler_span(str(m)) for m in table["mission"]]
    return [(float(a) + MJD_OFFSET[mission], float(b) + MJD_OFFSET[mission]) for a, b in zip(table["t_min"], table["t_max"])]


def target_name(star_id: str) -> str:
    """'KIC_11446443' -> 'KIC 11446443' (también EPIC y TIC)."""
    return star_id.replace("_", " ")


def download_star(star_id: str, mission: str, period: float, t0: float, max_days: float = MAX_DAYS) -> list[Path]:
    """Descarga a data/raw_lc los segmentos elegidos por `select_segments`. Lista vacía si MAST no tiene curvas."""
    res = lk.search_lightcurve(target_name(star_id), **SEARCH[mission])
    if len(res) == 0:
        return []
    chosen = select_segments(product_spans(res.table, mission), period, t0, max_days)
    RAW.mkdir(parents=True, exist_ok=True)
    return [Path(res[i].download(download_dir=str(RAW)).filename) for i in chosen]


def prepare(lc: "lk.LightCurve", period: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Un segmento -> (tiempo, flujo) aplanado (tránsito enmascarado) y normalizado a 1 fuera de tránsito."""
    lc = lc.remove_nans().normalize()
    mask = lc.create_transit_mask(period=period, transit_time=t0, duration=MASK_WIDTH * duration)
    step = float(np.median(np.diff(lc.time.value)))
    window = int(max(MASK_WIDTH * duration, MIN_WINDOW_DAYS) / step // 2 * 2 + 1)       # puntos, impar
    flat = lc.flatten(window_length=min(window, (len(lc) // 2) * 2 - 1), mask=mask)
    t, y = flat.time.value, flat.flux.value
    out = ~flat.create_transit_mask(period=period, transit_time=t0, duration=MASK_WIDTH * duration)
    return t, y / np.median(y[out])


def read_prepare(paths: list[Path], period: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Lee los segmentos (PDCSAP, calidad por defecto), prepara cada uno por separado y los concatena."""
    parts = [prepare(lk.read(p, quality_bitmask="default"), period, t0, duration) for p in paths]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
