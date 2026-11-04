"""Dataset de PyTorch con las vistas local y global de la muestra A (tarea #15, fase 2).

Misma entrada que la variante B de los modelos de sklearn (exoplanet.views.normalize): vistas normalizadas por estrella
y log10 de la profundidad. Las vistas par/impar quedan fuera (NaN cuando no hay tránsitos suficientes; decisión
2026-11-02). Todo cabe en memoria (~3 000 estrellas x 283 valores), así que se carga una vez como tensores.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from exoplanet.data import CLASSES, MISSIONS
from exoplanet.views import load_centroid, load_views, normalize


class ViewDataset(Dataset):
    """Una muestra = una estrella: {"local": (1, 81), "glob": (1, 201), "log_depth": (1,), "y": clase, "mission": misión}.
    `stars` es una tabla con star_id, mission y disposition (p. ej. exoplanet.views.load_sample filtrada por split)."""

    def __init__(self, stars: pd.DataFrame, centroid: bool = False):
        v = load_views(stars.star_id)
        local, glob, log_depth = normalize(v["local"], v["glob"])
        channels = [(local, glob)]
        if centroid:              # segundo canal: desplazamiento del centroide alineado con el flujo (variante 2026-11-03)
            c = load_centroid(stars.star_id)
            channels.append((c["local"], c["glob"]))
        self.local = torch.as_tensor(np.stack([l for l, _ in channels], axis=1), dtype=torch.float32)   # (n, canales, 81)
        self.glob = torch.as_tensor(np.stack([g for _, g in channels], axis=1), dtype=torch.float32)
        self.log_depth = torch.as_tensor(log_depth, dtype=torch.float32)
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
