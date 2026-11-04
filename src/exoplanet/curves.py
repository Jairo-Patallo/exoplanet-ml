"""Curvas de MAST (Kepler, K2, TESS): selección de segmentos, descarga, limpieza y aplanado (tarea #30).

Decisiones aprobadas: flujo PDCSAP con la máscara de calidad por defecto (D1); aplanado de lightkurve con el tránsito
enmascarado (D2); tope de ~4 meses de datos por estrella, eligiendo los segmentos con más tránsitos (D3);
normalización por la mediana fuera de tránsito (D8). El plegado en sí está en `exoplanet.folding`.
"""
import re
import time
from functools import lru_cache
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from exoplanet.data import DATA, MISSIONS
from exoplanet.folding import fold_views

RAW = DATA / "raw_lc"
MAX_DAYS = 100.0            # D3: se acumulan segmentos hasta superar este tiempo
MAX_SEGMENTS = {"k2": 1, "tess": 1}   # #32 (2026-11-01): los positivos tienen más sectores/campañas que los negativos y eso
                                      # delata la clase (atajo); en Kepler todas las estrellas tienen los mismos trimestres
MASK_WIDTH = 3.0            # D2: el tránsito enmascarado mide 3 duraciones
MAX_MASK_FRAC = 0.8         # ... pero deja al menos un 20 % de cada órbita sin enmascarar (2026-11-01)
MIN_WINDOW_DAYS = 0.5       # D2: ventana de aplanado >= max(3 duraciones, 0.5 d)
MJD_OFFSET = {"kepler": -54832.5, "k2": -54832.5, "tess": -56999.5}   # tiempo de la misión = MJD + offset (BKJD, BTJD)
SEARCH = {"kepler": dict(author="Kepler", cadence="long"), "k2": dict(author="K2", cadence="long"),
          "tess": dict(author="SPOC", exptime=120)}


def expected_transits(tmin: float, tmax: float, period: float, t0: float) -> int:
    """Número de tránsitos (t0 + n·P) dentro de [tmin, tmax]."""
    return max(0, int(np.floor((tmax - t0) / period) - np.ceil((tmin - t0) / period) + 1))


def select_segments(spans: list[tuple[float, float]], period: float, t0: float, max_days: float = MAX_DAYS,
                    max_segments: int | None = None) -> list[int]:
    """Índices de los segmentos elegidos: los de más tránsitos esperados (a igualdad, los más largos),
    acumulando hasta superar `max_days` o llegar a `max_segments`. `spans` y `t0` en el mismo sistema de tiempo."""
    order = sorted(range(len(spans)), key=lambda i: (-expected_transits(*spans[i], period, t0), -(spans[i][1] - spans[i][0])))
    chosen, total = [], 0.0
    for i in order:
        if total > max_days or len(chosen) == max_segments:
            break
        chosen.append(i)
        total += spans[i][1] - spans[i][0]
    return sorted(chosen)


@lru_cache(maxsize=1)
def kepler_quarters() -> pd.DataFrame:
    """Intervalo de cada trimestre de Kepler. MAST da el mismo intervalo (toda la misión) para todos los trimestres,
    así que se usa la tabla medida en las cabeceras de los archivos (scripts/data/build_kepler_quarters.py)."""
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


def mast_sequence(segment: int, mission: str) -> int:
    """Campaña/sector como lo numera MAST: las mitades de las campañas 10 y 11 de K2 (archivos c101/c102, c111/c112)
    van como 10 y 11. Solo en K2: TESS tiene sectores >= 100 de verdad (antes se convertían en 10; corregido 2026-11-03)."""
    return segment // 10 if mission == "k2" and segment >= 100 else segment


def download_star(star_id: str, mission: str, period: float, t0: float, max_days: float = MAX_DAYS,
                  segment: int | None = None) -> list[Path]:
    """Descarga a data/raw_lc los segmentos elegidos por `select_segments` (1 en K2 y TESS), o solo la campaña/sector
    `segment` si se da (negativos emparejados, #32). Lista vacía si MAST no tiene curvas."""
    res = lk.search_lightcurve(target_name(star_id), **SEARCH[mission])
    if segment is not None:
        res = res[np.asarray(res.table["sequence_number"]) == mast_sequence(segment, mission)]
    if len(res) == 0:
        return []
    chosen = select_segments(product_spans(res.table, mission), period, t0, max_days, MAX_SEGMENTS.get(mission))
    RAW.mkdir(parents=True, exist_ok=True)
    return [Path(res[i].download(download_dir=str(RAW)).filename) for i in chosen]


def mask_width(period: float, duration: float) -> float:
    """Anchura de la máscara: 3 duraciones, pero como mucho MAX_MASK_FRAC del periodo (y nunca menos que el tránsito).
    Sin el tope, con duración > P/3 (binarias de eclipse ancho) la máscara tapa toda la curva y no se puede aplanar."""
    return max(duration, min(MASK_WIDTH * duration, MAX_MASK_FRAC * period))


def prepare(lc: "lk.LightCurve", period: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Un segmento -> (tiempo, flujo) aplanado (tránsito enmascarado) y normalizado a 1 fuera de tránsito."""
    lc = lc.remove_nans().normalize()
    width = mask_width(period, duration)
    mask = lc.create_transit_mask(period=period, transit_time=t0, duration=width)
    step = float(np.median(np.diff(lc.time.value)))
    window = int(max(MASK_WIDTH * duration, MIN_WINDOW_DAYS) / step // 2 * 2 + 1)       # puntos, impar
    flat = lc.flatten(window_length=min(window, (len(lc) // 2) * 2 - 1), mask=mask)
    t, y = flat.time.value, flat.flux.value
    out = ~flat.create_transit_mask(period=period, transit_time=t0, duration=width)
    return t, y / np.median(y[out])


def prepare_centroid(lc: "lk.LightCurve", period: float, t0: float, duration: float) -> tuple[np.ndarray, ...]:
    """Un segmento -> (tiempo, dx, dy): centroide (columna, fila, en píxeles) menos su tendencia lenta (deriva del
    apuntado), estimada con una mediana móvil de la misma ventana que el aplanado del flujo y con el tránsito tapado."""
    t = lc.time.value
    cx, cy = np.asarray(lc.centroid_col.value, float), np.asarray(lc.centroid_row.value, float)
    ok = np.isfinite(t) & np.isfinite(cx) & np.isfinite(cy)
    t, cx, cy = t[ok], cx[ok], cy[ok]
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    out = np.abs(phase) >= 0.5 * mask_width(period, duration)
    window = max(MASK_WIDTH * duration, MIN_WINDOW_DAYS)
    idx = pd.to_datetime(t, unit="D")
    res = []
    for c in (cx, cy):
        trend = pd.Series(np.where(out, c, np.nan), index=idx).rolling(pd.Timedelta(days=window), center=True,
                                                                         min_periods=1).median()
        trend = trend.interpolate(limit_direction="both").to_numpy()
        res.append(c - trend)
    return t, res[0], res[1]


def read_centroid(paths: list[Path], period: float, t0: float, duration: float) -> tuple[np.ndarray, ...]:
    """Centroide sin tendencia de los segmentos (misma calidad que el flujo), concatenado: (tiempo, dx, dy)."""
    parts = [prepare_centroid(lk.read(p, quality_bitmask="default"), period, t0, duration) for p in paths]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def read_prepare(paths: list[Path], period: float, t0: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Lee los segmentos (PDCSAP, calidad por defecto), prepara cada uno por separado y los concatena."""
    parts = [prepare(lk.read(p, quality_bitmask="default"), period, t0, duration) for p in paths]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def run_star(row, views_dir: Path, retries: int = 1) -> dict:
    """Descarga y pliega una estrella; guarda sus vistas en views_dir/<star_id>.npz y devuelve su calidad (qc)."""
    t = time.perf_counter()
    for attempt in range(retries + 1):   # MAST da errores de red transitorios
        paths = []
        try:
            segment = getattr(row, "segment", None)    # solo en los negativos emparejados de K2/TESS (#32)
            segment = None if pd.isna(segment) else int(segment)
            paths = download_star(row.star_id, row.mission, row.period_days, row.epoch_bjd, segment=segment)
            if paths:                    # una descarga cortada deja un FITS truncado que lightkurve no vuelve a bajar
                for p in paths:
                    lk.read(p)
            break
        except Exception as e:
            if attempt == retries:
                raise
            bad = re.search(r"Data product (.+?\.fits)", str(e))   # lightkurve lee al descargar: la ruta viene en el error
            for p in paths + ([Path(bad.group(1))] if bad else []):
                p.unlink(missing_ok=True)
            time.sleep(5)
    t_down = time.perf_counter() - t
    if not paths:
        return {"error": "sin curvas en MAST", "t_download_s": t_down}
    t = time.perf_counter()
    T = row.duration_hrs / 24
    views = fold_views(*read_prepare(paths, row.period_days, row.epoch_bjd, T), row.period_days, row.epoch_bjd, T)
    views_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(views_dir / f"{row.star_id}.npz", local=views.local, glob=views.glob, odd=views.odd, even=views.even)
    return {"n_files": len(paths), "mb_downloaded": sum(p.stat().st_size for p in paths) / 1e6,
            "t_download_s": t_down, "t_fold_s": time.perf_counter() - t, **views.qc}


def summarize(qc: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Por misión: % descargadas y válidas, tamaño y tiempo por estrella, y coste estimado de bajar todo el universo."""
    ok = qc[qc["error"].isna()]
    rows = []
    for m in MISSIONS:
        a, o = qc[qc.mission == m], ok[ok.mission == m]
        if a.empty:
            continue
        n_all = int((universe.mission == m).sum())
        rows.append({"mission": m, "n_tried": len(a), "n_downloaded": len(o), "pct_no_curves_or_error": round(100 * (1 - len(o) / len(a)), 1),
                     "pct_valid_of_downloaded": round(100 * o["valid"].mean(), 1) if len(o) else np.nan,
                     "pct_valid_of_tried": round(100 * o["valid"].sum() / len(a), 1),
                     "pct_odd_even_valid": round(100 * o["odd_even_valid"].mean(), 1) if len(o) else np.nan,
                     "mb_per_star_mean": round(o["mb_downloaded"].mean(), 2), "mb_per_star_median": round(o["mb_downloaded"].median(), 2),
                     "s_per_star_median": round((o["t_download_s"] + o["t_fold_s"]).median(), 1),
                     "n_strong_stars_total": n_all,
                     "est_gb_full": round(n_all * o["mb_downloaded"].mean() / 1000, 1),
                     "est_hours_full": round(n_all * (o["t_download_s"] + o["t_fold_s"]).mean() / 3600, 1)})
    return pd.DataFrame(rows)
