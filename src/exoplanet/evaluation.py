"""Métricas de vetting comunes a todos los escalones (positivo = FALSE_POSITIVE, como en las reglas físicas del EDA)."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

POSITIVE = "FALSE_POSITIVE"


def vetting_metrics(y_true, y_pred, p_fp) -> dict[str, float]:
    """macro-F1 depende del umbral; PR-AUC y ROC-AUC no (útiles con el cambio de % de FP entre misiones)."""
    y = np.asarray(y_true) == POSITIVE
    return {"n": len(y), "frac_fp": y.mean(), "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "pr_auc": average_precision_score(y, p_fp), "roc_auc": roc_auc_score(y, p_fp)}
