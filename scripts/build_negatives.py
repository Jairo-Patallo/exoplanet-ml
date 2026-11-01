"""Lista de negativos NO_SIGNAL de la muestra A (tarea #32); ver exoplanet.negatives para las decisiones.

Necesita la muestra de positivos ya descargada (results/sample/qc.csv y data/raw_lc): de sus FITS sale la campaña (K2)
o sector (TESS) del único segmento que usa cada positivo. Magnitudes: Kp del archivo de K2 y Tmag del TIC (MAST).
Con --missions solo se rehacen esas misiones y se conservan las filas de las demás.

Uso:  python scripts/build_negatives.py [--missions k2 tess]   (PYTHONPATH=src)
Salida: results/sample/negatives.csv (una fila por estrella, con split y efeméride). La descarga y el plegado se hacen
con  python scripts/build_sample.py --negatives
"""
import argparse
from io import StringIO

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astroquery.mast import Catalogs, Observations

from exoplanet.curves import MAX_SEGMENTS, RAW, mast_sequence, select_segments
from exoplanet.data import DATA, SPLITS, TARGET, load_split
from exoplanet.negatives import match_negatives, random_ephemerides, segment_ephemerides, segment_of
from pilot_fold import strong_stars

OUT = "results/sample/negatives.csv"
SEED = 0
TESS_CANDIDATES = 3000      # por sector se consulta el Tmag de este número de candidatos al azar (el TIC va por lotes)
K2_SEARCH = "https://archive.stsci.edu/k2/data_search/search.php"
MAST_QUERY = {"k2": dict(obs_collection="K2", provenance_name="K2", t_exptime=1800),
              "tess": dict(obs_collection="TESS", provenance_name="SPOC", t_exptime=120)}


def mast_pool(mission: str, segment: int) -> pd.DataFrame:
    """Todas las estrellas con curva de cadencia larga (K2) o de 2 min (TESS) en una campaña/sector."""
    t = Observations.query_criteria(dataproduct_type="timeseries", sequence_number=mast_sequence(segment),
                                   **MAST_QUERY[mission])
    ids = [int(str(n).replace("ktwo", "")) for n in t["target_name"]]
    prefix = "EPIC_" if mission == "k2" else "TIC_"
    return pd.DataFrame({"star_id": [f"{prefix}{i}" for i in ids], "segment": segment,
                         "ra": np.asarray(t["s_ra"], float), "dec": np.asarray(t["s_dec"], float)})


def known_candidates() -> set[str]:
    """Estrellas con TOI o candidato K2 (cualquier disposición) y todas las del dataset: no pueden ser negativos."""
    toi = pd.read_csv(DATA / "labels" / "toi.csv")
    k2 = pd.read_csv(DATA / "labels" / "k2pandc.csv")
    ids = {f"TIC_{t}" for t in toi.tid} | {h.replace(" ", "_") for h in k2.epic_hostname.dropna()}
    for s in SPLITS:
        ids |= set(load_split(s, columns=["star_id"]).star_id)
    return ids


def k2_magnitudes(campaign: int) -> pd.DataFrame:
    """Kp de todas las estrellas de cadencia larga de una campaña de K2 (archivo de K2 de MAST)."""
    params = dict(action="Search", sci_campaign=campaign, ktc_target_type="LC", outputformat="CSV",
                  max_records=100000, selectedColumnsCsv="ktc_k2_id,kp")
    t = pd.read_csv(StringIO(requests.get(K2_SEARCH, params=params, timeout=300).text), skiprows=[1])
    return pd.DataFrame({"star_id": "EPIC_" + t["K2 ID"].astype(str), "mag": t["KEP Mag"]}).drop_duplicates("star_id")


def tess_magnitudes(star_ids: list[str]) -> pd.DataFrame:
    """Tmag del TIC, por lotes."""
    ids = [int(s.removeprefix("TIC_")) for s in star_ids]
    parts = [Catalogs.query_criteria(catalog="Tic", ID=ids[i:i + 500]).to_pandas()[["ID", "Tmag"]] for i in range(0, len(ids), 500)]
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ID", "Tmag"])
    return pd.DataFrame({"star_id": "TIC_" + t.ID.astype(str), "mag": t.Tmag.astype(float)}).drop_duplicates("star_id")


def positive_segments(pos: pd.DataFrame, mission: str) -> pd.Series:
    """Campaña/sector del segmento que usa cada positivo: la misma elección que download_star, con los intervalos de
    las cabeceras de los FITS en caché (el segmento elegido siempre se descargó, con o sin tope de segmentos)."""
    files: dict[str, list] = {}
    for path in RAW.rglob("*.fits"):
        if (s := segment_of(path, mission)) and s[0] in set(pos.star_id):
            files.setdefault(s[0], []).append((s[1], path))
    seg = {}
    for r in pos.itertuples():
        found = sorted(set(files.get(r.star_id, [])))
        if not found:
            continue
        spans = [(h["TSTART"], h["TSTOP"]) for h in (fits.getheader(f, 1) for _, f in found)]
        seg[r.star_id] = found[select_segments(spans, r.period_days, r.epoch_bjd, max_segments=MAX_SEGMENTS[mission])[0]][0]
    return pd.Series(seg)


def matched_negatives(pos: pd.DataFrame, mission: str, exclude: set[str], rng: np.random.Generator) -> pd.DataFrame:
    """Negativos de K2/TESS emparejados por segmento y magnitud (exoplanet.negatives.match_negatives)."""
    pos = pos.assign(segment=pos.star_id.map(positive_segments(pos, mission))).dropna(subset=["segment"])
    pools, pos_mags = [], []
    for seg in sorted(pos.segment.astype(int).unique()):
        pool = mast_pool(mission, seg)
        pool = pool[~pool.star_id.isin(exclude)].drop_duplicates("star_id")
        if mission == "k2":
            mags = k2_magnitudes(seg)
            pos_mags.append(mags[mags.star_id.isin(pos.star_id)])
        else:
            pool = pool.sample(min(len(pool), TESS_CANDIDATES), random_state=int(rng.integers(2**31)))
            mags = tess_magnitudes(list(pool.star_id))
        pools.append(pool.merge(mags, on="star_id", how="left"))
        print(f"  {mission} segmento {seg}: {len(pool)} candidatos", flush=True)
    if mission == "tess":
        pos_mags.append(tess_magnitudes(list(pos.star_id)))
    pos = pos.merge(pd.concat(pos_mags).drop_duplicates("star_id"), on="star_id", how="left")
    pool = pd.concat(pools, ignore_index=True)
    return match_negatives(pos, pool, exclude, rng).merge(pool[["star_id", "ra", "dec"]].drop_duplicates("star_id"), on="star_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", nargs="+", default=["kepler", "k2", "tess"])
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    eph_rng = np.random.default_rng([SEED, 1])     # efemérides con su propio generador: no cambian las parejas
    qc = pd.read_csv("results/sample/qc.csv")
    universe = strong_stars()
    pos = qc[qc.disposition != "NO_SIGNAL"].merge(universe[["star_id", "epoch_bjd"]], on="star_id")
    train_pos = universe.query("split == 'train'")
    exclude = known_candidates()
    parts = [pd.read_csv(OUT).query("mission not in @args.missions")] if args.missions != ap.get_default("missions") else []
    for mission in args.missions:
        g = pos[pos.mission == mission]
        if mission == "kepler":     # NO_SIGNAL del dataset, en su propio split
            n_split = g.split.value_counts().reindex(list(SPLITS), fill_value=0).to_dict()
            stars = pd.concat([load_split(s, "kepler", ["star_id", TARGET, "ra", "dec"]).query(f"{TARGET} == 'NO_SIGNAL'")
                               .sample(n_split[s], random_state=SEED).assign(split=s) for s in SPLITS], ignore_index=True)
            eph = random_ephemerides(train_pos[train_pos.mission == mission], len(stars), eph_rng)
        else:
            stars = matched_negatives(g, mission, exclude, rng)
            print(f"  {mission}: |dmag| mediana {stars.dmag.median():.2f}, p90 {stars.dmag.quantile(0.9):.2f}", flush=True)
            tp = g[g.split == "train"]
            eph = segment_ephemerides(stars, tp.assign(segment=tp.star_id.map(positive_segments(tp, mission))), eph_rng)
        extra = [c for c in ["segment", "mag", "pair_star_id", "dmag"] if c in stars]
        parts.append(pd.concat([stars[["star_id", "split", "ra", "dec"] + extra].reset_index(drop=True), eph.reset_index(drop=True)], axis=1)
                     .assign(mission=mission, disposition="NO_SIGNAL"))
        print(f"{mission}: {len(stars)} negativos para {len(g)} positivos; splits {stars.split.value_counts().to_dict()}", flush=True)
    pd.concat(parts, ignore_index=True).to_csv(OUT, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
