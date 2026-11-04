"""CNN 1D de dos ramas al estilo AstroNet (Shallue & Vanderburg 2018). Entrenamiento en scripts/models/cnn.py.

Rama global: 5 bloques convolucionales (2 conv de núcleo 5 + BatchNorm + ReLU, y max-pool 2); rama local: 2 bloques.
Se aplanan, se unen con log10 de la profundidad (estandarizado con train) y pasan por 2 capas densas de 256 con dropout.
Canales: `channels` en la local (flujo, + centroide) y `glob_channels` en la global (los mismos, + par e impar).
Variante híbrida (2026-11-04, al estilo ExoMiner): `feats` = diagnósticos físicos (exoplanet.features) que entran en la
cabeza; los NaN se rellenan con la mediana de train y se estandarizan con train (buffers guardados con el modelo).
"""
import torch
from torch import nn

from exoplanet.data import CLASSES

FC = 256


def conv_column(n_blocks: int, base: int, c_in: int = 1) -> nn.Sequential:
    layers = []
    for i in range(n_blocks):
        c_out = base * 2 ** i
        layers += [nn.Conv1d(c_in, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(),
                   nn.Conv1d(c_out, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2)]
        c_in = c_out
    return nn.Sequential(*layers, nn.Flatten())


class AstroCNN(nn.Module):
    def __init__(self, base: int, dropout: float, depth_mean: torch.Tensor, depth_std: torch.Tensor, channels: int = 1,
                 glob_channels: int | None = None, feat_stats: tuple[torch.Tensor, ...] | None = None):
        super().__init__()
        self.register_buffer("depth_mean", depth_mean)
        self.register_buffer("depth_std", depth_std)
        self.glob = conv_column(5, base, glob_channels or channels)   # 201 -> 6 posiciones x 16·base canales
        self.local = conv_column(2, base, channels)                   # 81 -> 20 posiciones x 2·base canales
        n_feats = 0
        if feat_stats is not None:          # (mediana, media, desviación) de cada diagnóstico en train
            for name, t in zip(("feat_median", "feat_mean", "feat_std"), feat_stats):
                self.register_buffer(name, t)
            n_feats = len(feat_stats[0])
        n_flat = 6 * 16 * base + 20 * 2 * base + 1 + n_feats
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_flat, FC), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(FC, FC), nn.ReLU(), nn.Dropout(dropout), nn.Linear(FC, len(CLASSES)))

    def forward(self, local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor,
                feats: torch.Tensor | None = None) -> torch.Tensor:
        parts = [self.glob(glob), self.local(local), (log_depth - self.depth_mean) / self.depth_std]
        if feats is not None and hasattr(self, "feat_median"):
            f = torch.where(torch.isnan(feats), self.feat_median, feats)
            parts.append((f - self.feat_mean) / self.feat_std)
        return self.head(torch.cat(parts, dim=1))


def feat_stats(feats: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Mediana (para imputar NaN), media y desviación de cada diagnóstico, calculadas solo con los datos de ajuste."""
    med = torch.nanmedian(feats, dim=0).values
    filled = torch.where(torch.isnan(feats), med, feats)
    return med, filled.mean(0), filled.std(0) + 1e-6
