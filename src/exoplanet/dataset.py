"""Dataset de PyTorch con las vistas local y global de la muestra (tarea #15, fase 2).

Misma entrada que la variante B de los modelos de sklearn (exoplanet.views.normalize): vistas normalizadas por estrella
y log10 de la profundidad. Las vistas par/impar quedan fuera (NaN cuando no hay tránsitos suficientes; decisión
2026-11-02). Todo cabe en memoria (~3 000 estrellas x 283 valores), así que se carga una vez como tensores.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from exoplanet.data import CLASSES, MISSIONS
from exoplanet.features import curve_features
from exoplanet.views import SAMPLE_VIEWS, load_centroid, load_views, normalize


class ViewDataset(Dataset):
    """Una muestra = una estrella: {"local": (1, 81), "glob": (1, 201), "log_depth": (1,), "y": clase, "mission": misión}.
    `stars` es una tabla con star_id, mission y disposition (p. ej. exoplanet.views.load_sample filtrada por split)."""

    def __init__(self, stars: pd.DataFrame, centroid: bool = False, odd_even: bool = False, features: bool = False,
                 ephem: bool = False, views_dir=SAMPLE_VIEWS):
        v = load_views(stars.star_id, ("local", "glob", "odd", "even"), views_dir)
        local, glob, log_depth = normalize(v["local"], v["glob"])
        locs, globs = [local], [glob]
        if centroid:              # canal del desplazamiento del centroide, alineado con el flujo (variante 2026-11-03)
            c = load_centroid(stars.star_id)
            locs.append(c["local"]), globs.append(c["glob"])
        if odd_even:              # canales par e impar en la rama global (2026-11-04), normalizados como la global;
            depth = 10.0 ** log_depth   # si no son válidas (pocos tránsitos) se repite la global: sin indicador de "falta"
            for k in ("odd", "even"):
                o = (v[k] - np.median(v[k], axis=1, keepdims=True)) / depth
                globs.append(np.where(np.isnan(o).any(axis=1, keepdims=True), glob, o))
        self.local = torch.as_tensor(np.stack(locs, axis=1), dtype=torch.float32)   # (n, canales, 81)
        self.glob = torch.as_tensor(np.stack(globs, axis=1), dtype=torch.float32)  # (n, canales, 201)
        self.log_depth = torch.as_tensor(log_depth, dtype=torch.float32)
        if features:              # diagnósticos físicos (exoplanet.features) con la ventana ajustada a la duración
            duty = (stars.duration_hrs / 24 / stars.period_days).to_numpy()
            f = curve_features(v["local"], v["odd"], v["even"], v["glob"], duty=duty, mask_no_dip=False).to_numpy(float)
            self.feats = torch.as_tensor(np.sign(f) * np.log1p(np.abs(f)), dtype=torch.float32)   # NaN se imputa en el modelo
        if ephem:                 # efeméride (2026-11-04): log(duración/periodo) y log(periodo); las vistas normalizadas no
            e = np.log10(np.c_[stars.duration_hrs / 24 / stars.period_days, stars.period_days])   # las contienen
            e = torch.as_tensor(e, dtype=torch.float32)
            self.feats = torch.cat([self.feats, e], dim=1) if features else e
        self.y = torch.as_tensor(stars.disposition.map(CLASSES.index).to_numpy(), dtype=torch.long)
        self.mission = torch.as_tensor(stars.mission.map(MISSIONS.index).to_numpy(), dtype=torch.long)
        self.star_id = stars.star_id.to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"local": self.local[i], "glob": self.glob[i], "log_depth": self.log_depth[i], "y": self.y[i],
                "mission": self.mission[i]}

    def class_weights(self) -> torch.Tensor:
        """Pesos inversos a la frecuencia, como class_weight="balanced" de sklearn: n / (k · n_clase)."""
        counts = torch.bincount(self.y, minlength=len(CLASSES)).float()
        return len(self.y) / (len(CLASSES) * counts)
