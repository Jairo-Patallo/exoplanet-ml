"""Receta final con las vistas de TESS multisector (fase 6, opción A de Jairo, 2026-11-06).

Mismo entrenamiento que la línea base de val (final_model.py val): configuración elegida, sin profundidad, solo train,
5 semillas. La única diferencia son las vistas: data/sample/views_tess_multi (build_tess_multisector.py), donde las
parejas de TESS con varios sectores usan k sectores cada estrella y el resto conserva su vista de 1 sector.
La comparación con la línea base está en scripts/checks/tess_multi_compare.py.

Uso (WSL, GPU):  PYTHONPATH=src ~/venvs/exoplanet-rocm/bin/python scripts/models/tess_multisector.py
Salida: data/models/multi_seed*.pt y results/models/multi_val_pred.csv (formato de final_val_pred.csv)
"""
import numpy as np
import torch

from exoplanet.data import CLASSES, DATA
from exoplanet.dataset import ViewDataset
from exoplanet.training import fit, inputs, predict
from exoplanet.views import load_sample
from final_model import MODELS, OUT, SEEDS, WEIGHT_DECAY, chosen, make

VIEWS = DATA / "sample" / "views_tess_multi"


def x_multi(stars):
    d = ViewDataset(stars, views_dir=VIEWS)
    return {**inputs(d), "log_depth": torch.zeros_like(d.log_depth)}, d.y


def main() -> None:
    cfg, va = chosen(), load_sample(("val",))
    x, y = x_multi(load_sample(("train",)))
    xv, _ = x_multi(va)
    probas = []
    for s in SEEDS:
        model, epoch = fit(make(cfg), x, y, WEIGHT_DECAY, s, augment=True, amp=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "seed": s, "views": str(VIEWS)}, MODELS / f"multi_seed{s}.pt")
        probas.append(predict(model, xv, amp=True))
        print(f"semilla {s}: época {epoch}", flush=True)
    p = np.mean(probas, axis=0)
    va[["star_id", "mission", "disposition"]].assign(**{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}).round(4).to_csv(
        OUT / "multi_val_pred.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
