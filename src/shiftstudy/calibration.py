"""Temperature fitting and least-ambiguous finite-sample split-conformal sets."""

import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from .metrics import PROBABILITY_FLOOR, checked_probabilities, nll


def temperature_scale(probabilities, temperature):
    p = checked_probabilities(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    # log(p) differs from pre-softmax logits by a row-wise additive constant.
    # Clip zero probabilities explicitly; this also defines the numerical floor.
    return softmax(np.log(np.maximum(p, PROBABILITY_FLOOR)) / temperature, axis=1)


def fit_temperature(probabilities, labels):
    p, y = checked_probabilities(probabilities, labels)
    objective = lambda log_t: nll(temperature_scale(p, np.exp(log_t)), y)
    optimum = minimize_scalar(objective, bounds=(-3., 3.), method="bounded",
                              options={"xatol": 1e-7})
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError("Temperature optimization failed")
    # Include identity and boundaries, so optimization never worsens calibration NLL.
    candidates = (0., -3., 3., float(optimum.x))
    log_t = min(candidates, key=objective)
    return {"temperature": float(np.exp(log_t)), "nll_before": nll(p, y),
            "nll_after": objective(log_t), "at_bound": abs(log_t) >= 3. - 1e-5,
            "log_temperature_bounds": [-3., 3.]}


def conformal_threshold(probabilities, labels, alpha=0.1):
    p, y = checked_probabilities(probabilities, labels)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    scores = 1. - p[np.arange(len(y)), y]
    rank = math.ceil((len(y) + 1) * (1 - alpha))
    # score <= 1 always: 1 produces the full set without non-standard JSON infinity.
    threshold = 1. if rank > len(y) else float(np.partition(scores, rank - 1)[rank - 1])
    return threshold


def conformal_sets(probabilities, threshold):
    p = checked_probabilities(probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Conformal threshold must be in [0, 1]")
    # Keep equality; do not force the argmax into empty sets after calibration.
    return 1. - p <= threshold


def set_metrics(sets, labels):
    sets = np.asarray(sets)
    y = np.asarray(labels)
    if sets.dtype != bool or sets.ndim != 2 or len(sets) == 0:
        raise ValueError("sets must be a nonempty Boolean matrix")
    if y.shape != (len(sets),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Invalid labels for prediction sets")
    if (y < 0).any() or (y >= sets.shape[1]).any():
        raise ValueError("Label outside set vocabulary")
    size = sets.sum(axis=1)
    return {"set_coverage": float(sets[np.arange(len(y)), y].mean()),
            "set_size": float(size.mean()), "empty_set_rate": float((size == 0).mean())}
