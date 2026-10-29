"""Baseline dummy (clase más frecuente) con el protocolo anidado: el nuevo listón a superar.

Por fold externo: la clase más frecuente en train+val de las dos misiones de ajuste, evaluada en la misión excluida entera.
Dentro de dominio: la más frecuente en train+val de una misión, evaluada en su test.
Dos variantes: vetting (PLANET vs FALSE_POSITIVE) y 3 clases con los datos actuales (NO_SIGNAL solo existe en Kepler).

Uso:  python scripts/baseline.py   (desde la raíz del repo)
Salida: results/baseline/dummy.csv
IC 95 % por bootstrap sobre estrellas, simulado con una multinomial (el dummy es constante, así que solo importan los recuentos).
Con scores constantes: PR-AUC de cada clase = su prevalencia y ROC-AUC = 0.5; se comprueba con sklearn en la estimación puntual.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from exoplanet.data import CLASSES, MISSIONS, TARGET
from exoplanet.evaluation import in_domain_split, load_meta, nested_folds

OUT = Path("results/baseline")
SEED = 42
N_BOOT = 1000
VARIANTS = {"vetting (PLANET/FP)": ("PLANET", "FALSE_POSITIVE"), "3 clases (datos actuales)": CLASSES}


def macro_f1_const(counts: np.ndarray, pred: int) -> np.ndarray:
    """Macro-F1 de un predictor constante. counts: (n_boot, n_clases) recuentos verdaderos. Solo la clase predicha tiene F1 > 0;
    el macro promedia las clases presentes en la verdad o en la predicción."""
    n = counts.sum(axis=1)
    prev = counts[:, pred] / n
    f1_pred = np.where(prev > 0, 2 * prev / (1 + prev), 0.0)
    n_labels = (counts > 0).sum(axis=1) + (counts[:, pred] == 0)
    return f1_pred / n_labels


def fp_prevalence(counts: np.ndarray, classes: tuple) -> np.ndarray:
    """PR-AUC del azar para FALSE_POSITIVE = su prevalencia (el macro-PR-AUC de un dummy siempre es 1/k, no informa)."""
    return counts[:, classes.index("FALSE_POSITIVE")] / counts.sum(axis=1)


def evaluate(meta: pd.DataFrame, fit_idx: np.ndarray, eval_idx: np.ndarray, classes: tuple, rng) -> dict:
    keep = lambda idx: idx[meta[TARGET].iloc[idx].isin(classes).to_numpy()]
    fit_idx, eval_idx = keep(fit_idx), keep(eval_idx)
    pred = meta[TARGET].iloc[fit_idx].value_counts().idxmax()
    y = meta[TARGET].iloc[eval_idx].to_numpy()
    counts = np.array([(y == c).sum() for c in classes])
    # estimación puntual con sklearn (comprueba las fórmulas cerradas)
    f1 = f1_score(y, np.full(len(y), pred), average="macro", zero_division=0)
    pr = average_precision_score(y == "FALSE_POSITIVE", np.full(len(y), float(pred == "FALSE_POSITIVE")))
    boot = rng.multinomial(counts.sum(), counts / counts.sum(), size=N_BOOT)
    f1_b, pr_b = macro_f1_const(boot, classes.index(pred)), fp_prevalence(boot, classes)
    assert np.isclose(f1, macro_f1_const(counts[None], classes.index(pred))[0]), "fórmula de F1 distinta de sklearn"
    assert np.isclose(pr, fp_prevalence(counts[None], classes)[0]), "fórmula de PR-AUC distinta de sklearn"
    ci = lambda v, x: f"{x:.3f} [{np.percentile(v, 2.5):.3f},{np.percentile(v, 97.5):.3f}]"
    return {"n_estrellas": len(y), "dummy_predice": pred, "macro_f1": ci(f1_b, f1), "pr_auc_FP": ci(pr_b, pr),
            "roc_auc": "0.500 (por definición)"}


def main() -> None:
    meta = load_meta()
    rng = np.random.default_rng(SEED)
    rows = []
    for variant, classes in VARIANTS.items():
        for f in nested_folds(meta):
            rows.append({"variante": variant, "evaluacion": f"externo: excluida {f.held_out}",
                         **evaluate(meta, f.fit_idx, f.eval_idx, classes, rng)})
        for mission in MISSIONS:
            fit, test = in_domain_split(meta, mission)
            rows.append({"variante": variant, "evaluacion": f"dentro de dominio: {mission} (test)",
                         **evaluate(meta, fit, test, classes, rng)})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "dummy.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
