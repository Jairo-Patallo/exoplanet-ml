"""Entrenamiento común de las redes de la fase 2 (MLP y CNN), para que todas se entrenen igual.

Entradas como diccionario de tensores {"local": (n, 1, 81), "glob": (n, 1, 201), "log_depth": (n, 1)} (exoplanet.dataset).
Parada temprana con un 15 % de los datos de entrenamiento apartado y estratificado: val no decide cuándo parar.
Entropía cruzada con pesos de clase "balanced", AdamW. Opcionalmente, aumento por reflexión: se da la vuelta en el
tiempo a las vistas local y global (un tránsito es simétrico, así que sigue siendo una curva válida de su clase).
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from exoplanet.data import CLASSES
from exoplanet.dataset import ViewDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 64, 300, 20


def inputs(ds: ViewDataset) -> dict[str, torch.Tensor]:
    return {"local": ds.local, "glob": ds.glob, "log_depth": ds.log_depth}


def take(x: dict[str, torch.Tensor], idx) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in x.items()}


def reflect(x: dict[str, torch.Tensor], gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Da la vuelta a la mitad de las curvas del lote, elegidas al azar (mismo giro para su local y su global)."""
    flip = torch.rand(len(x["local"]), generator=gen) < 0.5
    out = dict(x)
    for k in ("local", "glob"):
        out[k] = torch.where(flip.to(x[k].device)[:, None, None], x[k].flip(-1), x[k])
    return out


def fit(make_model, x: dict[str, torch.Tensor], y: torch.Tensor, weight_decay: float, seed: int,
        augment: bool = False, lr: float = LR, amp: bool = False, patience: int = PATIENCE) -> tuple[nn.Module, int]:
    """Entrena el modelo que devuelve make_model(x_ajuste) con parada temprana; devuelve el de menor pérdida y su época.
    make_model recibe los datos de ajuste para fijar con ellos la estandarización (nunca con val).
    Si el modelo define param_groups(lr), se usan (p. ej. lr menor para un modelo preentrenado). amp: bf16 en GPU."""
    torch.manual_seed(seed)
    idx_fit, idx_stop = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y.numpy(), random_state=seed)
    xf, yf = {k: v.to(DEVICE) for k, v in take(x, idx_fit).items()}, y[idx_fit].to(DEVICE)
    xs, ys = {k: v.to(DEVICE) for k, v in take(x, idx_stop).items()}, y[idx_stop].to(DEVICE)
    weights = len(yf) / (len(CLASSES) * torch.bincount(yf, minlength=len(CLASSES)).float())
    model = make_model(xf).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    groups = model.param_groups(lr) if hasattr(model, "param_groups") else model.parameters()
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    autocast = torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch, waited = np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b in torch.randperm(len(yf), generator=gen).split(BATCH):
            xb = take(xf, b.to(DEVICE))
            if augment:
                xb = reflect(xb, gen)
            opt.zero_grad()
            with autocast:
                loss = loss_fn(model(**xb).float(), yf[b.to(DEVICE)])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad(), autocast:
            logits = torch.cat([model(**take(xs, c)).float() for c in torch.arange(len(ys), device=DEVICE).split(512)])
            stop_loss = loss_fn(logits, ys).item()
        if stop_loss < best:
            best, best_state, best_epoch, waited = stop_loss, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


@torch.no_grad()
def predict(model: nn.Module, x: dict[str, torch.Tensor], amp: bool = False) -> np.ndarray:
    """Probabilidades (n, 3) con columnas en el orden de CLASSES (por lotes, para no llenar la memoria de la GPU)."""
    model.eval()
    out = []
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=amp):
        for c in torch.arange(len(x["local"])).split(512):
            out.append(torch.softmax(model(**{k: v[c].to(DEVICE) for k, v in x.items()}).float(), dim=1).cpu())
    return torch.cat(out).numpy()
