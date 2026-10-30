"""Mide el intervalo real (BKJD) de cada trimestre de Kepler leyendo las cabeceras de los archivos de MAST.

MAST da el mismo intervalo (toda la misión) para todos los trimestres de una estrella, y las marcas de tiempo del nombre
de archivo no son fiables, así que se mide: se busca en el dataset una estrella con los 18 trimestres (Q0-Q17) y se leen
TSTART/TSTOP de cada archivo (~10 MB en total, a data/raw_lc).

Uso:  python scripts/build_kepler_quarters.py   (desde la raíz del repo)
Salida: src/exoplanet/kepler_quarters.csv  (quarter, tstart_bkjd, tstop_bkjd)
"""
import warnings
from pathlib import Path

import lightkurve as lk
import pandas as pd
from astropy.io import fits

from exoplanet.curves import RAW, SEARCH, target_name
from exoplanet.data import load_split

OUT = Path(__file__).resolve().parents[1] / "src" / "exoplanet" / "kepler_quarters.csv"
N_QUARTERS = 18


def main() -> None:
    warnings.filterwarnings("ignore")
    stars = load_split("train", "kepler", columns=["star_id"]).star_id.head(60)
    for star in stars:
        res = lk.search_lightcurve(target_name(star), **SEARCH["kepler"])
        quarters = sorted(int(m.split()[-1]) for m in res.table["mission"])
        if quarters == list(range(N_QUARTERS)):
            print(f"{star}: tiene los {N_QUARTERS} trimestres")
            break
    else:
        raise SystemExit("ninguna de las primeras 60 estrellas de Kepler tiene los 18 trimestres")
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(len(res)):
        path = Path(res[i].download(download_dir=str(RAW)).filename)
        h = fits.getheader(path, 1)
        rows.append({"quarter": int(res.table["mission"][i].split()[-1]), "tstart_bkjd": h["TSTART"], "tstop_bkjd": h["TSTOP"]})
        print(rows[-1])
    pd.DataFrame(rows).sort_values("quarter").round(4).to_csv(OUT, index=False)
    print("guardado en", OUT)


if __name__ == "__main__":
    main()
