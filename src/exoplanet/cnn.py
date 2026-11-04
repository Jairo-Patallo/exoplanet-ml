"""CNN 1D de dos ramas al estilo AstroNet (Shallue & Vanderburg 2018). Entrenamiento en scripts/models/cnn.py.

Rama global: 5 bloques convolucionales (2 conv de núcleo 5 + BatchNorm + ReLU, y max-pool 2); rama local: 2 bloques.
Se aplanan, se unen con log10 de la profundidad (estandarizado con train) y pasan por 2 capas densas de 256 con dropout.
`channels` = 1 (flujo) o 2 (flujo + centroide).
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
    def __init__(self, base: int, dropout: float, depth_mean: torch.Tensor, depth_std: torch.Tensor, channels: int = 1):
        super().__init__()
        self.register_buffer("depth_mean", depth_mean)
        self.register_buffer("depth_std", depth_std)
        self.glob = conv_column(5, base, channels)     # 201 -> 6 posiciones x 16·base canales
        self.local = conv_column(2, base, channels)    # 81 -> 20 posiciones x 2·base canales
        n_flat = 6 * 16 * base + 20 * 2 * base + 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_flat, FC), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(FC, FC), nn.ReLU(), nn.Dropout(dropout), nn.Linear(FC, len(CLASSES)))

    def forward(self, local: torch.Tensor, glob: torch.Tensor, log_depth: torch.Tensor) -> torch.Tensor:
        d = (log_depth - self.depth_mean) / self.depth_std
        return self.head(torch.cat([self.glob(glob), self.local(local), d], dim=1))
