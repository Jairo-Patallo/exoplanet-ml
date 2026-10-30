"""Descarga las disposiciones originales de TESS (TOI) y K2 del NASA Exoplanet Archive (TAP), que el dataset de HF no conserva.

Uso:  python scripts/fetch_labels.py   (desde la raíz del repo)
Salida (data/labels/, ignorado por git; son tablas pequeñas):
  toi.csv      -> tid, toi, tfopwg_disp (CP, KP, PC, APC, FP, FA), pl_orbper, ra, dec
  k2pandc.csv  -> epic_hostname, k2_name, disposition (CONFIRMED, CANDIDATE, FALSE POSITIVE, REFUTED), ra, dec
"""
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from exoplanet.data import DATA

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
QUERIES = {
    "toi": "select tid,toi,tfopwg_disp,pl_orbper,ra,dec from toi",
    "k2pandc": "select epic_hostname,k2_name,disposition,pl_orbper,ra,dec from k2pandc",
}


def main() -> None:
    out = DATA / "labels"
    out.mkdir(parents=True, exist_ok=True)
    for name, query in QUERIES.items():
        with urlopen(TAP + quote(query), timeout=120) as r:
            body = r.read()
        (out / f"{name}.csv").write_bytes(body)
        rows = body.count(b"\n") - 1
        print(f"{name}.csv: {len(body) / 1e6:.2f} MB, {rows} filas")


if __name__ == "__main__":
    main()
