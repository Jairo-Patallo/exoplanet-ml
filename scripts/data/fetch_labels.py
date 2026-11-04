"""Descarga las disposiciones originales de TESS (TOI) y K2 del NASA Exoplanet Archive (TAP), que el dataset de HF no conserva.

Uso:  python scripts/data/fetch_labels.py   (desde la raíz del repo)
Salida (data/labels/, ignorado por git; son tablas pequeñas):
  toi.csv      -> tid, toi, tfopwg_disp (CP, KP, PC, APC, FP, FA), pl_orbper, ra, dec
  k2pandc.csv  -> epic_hostname, k2_name, disposition (CONFIRMED, CANDIDATE, FALSE POSITIVE, REFUTED), ra, dec
  koi.csv      -> kepid, kepoi_name, koi_disposition, koi_pdisposition, koi_period y los motivos de falso positivo del
                  Robovetter: koi_fpflag_nt (no parece tránsito), _ss (eclipse estelar), _co (centroide desplazado),
                  _ec (coincidencia de efemérides con otra estrella). Para el análisis de errores (2026-11-04).
"""
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from exoplanet.data import DATA

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
QUERIES = {
    "toi": "select tid,toi,tfopwg_disp,pl_orbper,ra,dec from toi",
    "k2pandc": "select epic_hostname,k2_name,disposition,pl_orbper,ra,dec from k2pandc",
    "koi": "select kepid,kepoi_name,koi_disposition,koi_pdisposition,koi_period,"
           "koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec from cumulative",
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
