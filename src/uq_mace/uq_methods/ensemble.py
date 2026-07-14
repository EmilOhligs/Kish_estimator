"""Ensemble-based UQ: sigma(R) = std of predictions across N MACE models."""
from __future__ import annotations

import numpy as np


def ensemble_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """predictions: shape (n_models, n_configurations). Returns per-configuration sigma(R)."""
    return np.std(predictions, axis=0, ddof=1)
