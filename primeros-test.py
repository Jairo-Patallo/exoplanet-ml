"""Lavado de datos + baseline de la clase más frecuente.

Uso:  python primeros-test.py
Solo se leen las columnas tabulares (no las curvas de luz), así que es rápido.
"""
import pandas as pd
import pyarrow.parquet as pq
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

SEED = 42
FILES = {"train": "data/train.parquet", "val": "data/val.parquet", "test": "data/test.parquet"}
TARGET = "disposition"
CLASSES = ["PLANET", "FALSE_POSITIVE", "NO_SIGNAL"]

# Variables que se eliminan y por qué
LEAKAGE = ["label_source", "confidence_score", "kepoi_name", "disposition_disputed",
           "fp_flag_not_transit", "fp_flag_stellar", "fp_flag_centroid", "fp_flag_ephemeris"]  # derivan de la etiqueta
IDENTIFIERS = ["star_id", "ra", "dec"]  # identifican la estrella / su posición en el cielo
SHORTCUTS = ["mission", "cadence", "has_transit_params", "n_planets_in_system",  # NO_SIGNAL solo existe en Kepler
             "period_days", "epoch_bjd", "duration_hrs", "depth_ppm", "planet_radius_earth",  # 100 % NaN en NO_SIGNAL
             "flux_global_len", "flux_local_len", "flux_folded_len", "flux_raw_len"]  # constantes
DROP = LEAKAGE + IDENTIFIERS + SHORTCUTS


def load_tabular(path: str) -> pd.DataFrame:
    """Lee solo las columnas que no son arrays binarios (evita cargar ~1.5 GB de curvas)."""
    schema = pq.ParquetFile(path).schema_arrow
    cols = [f.name for f in schema if not f.name.startswith("flux_") or f.name.endswith("_len")]
    cols = [c for c in cols if c != "source_urls"]
    return pd.read_parquet(path, columns=cols)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) sin las columnas que filtran la etiqueta o regalan la clase."""
    assert df[TARGET].notna().all() and set(df[TARGET]) <= set(CLASSES), "etiquetas inesperadas"
    y = df[TARGET]
    X = df.drop(columns=[c for c in DROP + [TARGET] if c in df.columns])
    return X, y


def main() -> None:
    raw = {k: load_tabular(p) for k, p in FILES.items()}

    # Comprobación anti-fuga: una misma estrella no puede estar en dos splits
    ids = {k: set(d["star_id"]) for k, d in raw.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"Estrellas compartidas {a}/{b}: {len(ids[a] & ids[b])}")
    for k, d in raw.items():
        print(f"{k}: {len(d)} filas, {d['star_id'].duplicated().sum()} star_id duplicados")

    data = {k: clean(d) for k, d in raw.items()}
    X_tr, y_tr = data["train"]
    print(f"\nVariables eliminadas ({len(DROP)}): {DROP}")
    print(f"Variables que quedan ({X_tr.shape[1]}): {list(X_tr.columns)}")

    print("\nDistribución de clases (proporción):")
    print(pd.DataFrame({k: y.value_counts(normalize=True) for k, (_, y) in data.items()}).round(3))

    # Baseline: predice siempre la clase más frecuente del TRAIN
    base = DummyClassifier(strategy="most_frequent", random_state=SEED).fit(X_tr, y_tr)
    print(f"\nClase que predice el baseline: {base.classes_[base.class_prior_.argmax()]}")
    for split in ["train", "val", "test"]:
        X, y = data[split]
        pred = base.predict(X)
        print(f"\n== {split} ==  accuracy={accuracy_score(y, pred):.3f}  "
              f"F1 macro={f1_score(y, pred, average='macro', zero_division=0):.3f}")
        if split == "test":
            print(classification_report(y, pred, labels=CLASSES, zero_division=0))
            print("Matriz de confusión (filas=real, columnas=predicha):")
            print(pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES), index=CLASSES, columns=CLASSES))


if __name__ == "__main__":
    main()
