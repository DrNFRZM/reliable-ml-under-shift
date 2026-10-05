"""Explicit metric definitions, including expected selective risk within ties."""

import numpy as np
from sklearn.metrics import f1_score

PROBABILITY_FLOOR = 1e-15


def checked_probabilities(probabilities, labels=None):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("Probabilities must be a nonempty N-by-K matrix, K >= 2")
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1., atol=1e-7, rtol=0)):
        raise ValueError("Probabilities must be finite, bounded, and sum to one")
    if labels is None:
        return p
    y = np.asarray(labels)
    if (y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer)
            or (y < 0).any() or (y >= p.shape[1]).any()):
        raise ValueError("Labels must be an integer vector indexing probability columns")
    return p, y


def nll(probabilities, labels):
    p, y = checked_probabilities(probabilities, labels)
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], PROBABILITY_FLOOR)).mean())


def calibration_bins(probabilities, labels, bins=15):
    p, y = checked_probabilities(probabilities, labels)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    count = np.bincount(indices, minlength=bins)
    sums = np.bincount(indices, weights=confidence, minlength=bins)
    hits = np.bincount(indices, weights=correct, minlength=bins)
    mean_conf = np.divide(sums, count, out=np.zeros(bins), where=count > 0)
    accuracy = np.divide(hits, count, out=np.zeros(bins), where=count > 0)
    ece = float(np.sum(count * np.abs(mean_conf - accuracy)) / len(y))
    return {"count": count, "confidence": mean_conf, "accuracy": accuracy,
            "lower": np.arange(bins) / bins, "upper": np.arange(1, bins + 1) / bins,
            "ece": ece}


def risk_coverage(probabilities, labels):
    p, y = checked_probabilities(probabilities, labels)
    confidence = p.max(axis=1)
    errors = (p.argmax(axis=1) != y).astype(float)
    order = np.argsort(-confidence, kind="stable")
    ordered_conf, ordered_errors = confidence[order], errors[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_conf) != 0) + 1, len(y)]
    expected_cumulative = np.empty(len(y))
    previous_errors = 0.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block_errors = ordered_errors[start:end].sum()
        expected_cumulative[start:end] = (previous_errors
            + np.arange(1, end - start + 1) * block_errors / (end - start))
        previous_errors += block_errors
    counts = np.arange(1, len(y) + 1)
    risk = expected_cumulative / counts
    return {"coverage": counts / len(y), "risk": risk,
            "random_risk": np.full(len(y), errors.mean()),
            "oracle_risk": np.cumsum(np.sort(errors)) / counts,
            "aurc": float(risk.mean())}


def prediction_metrics(probabilities, labels):
    p, y = checked_probabilities(probabilities, labels)
    pred = p.argmax(axis=1)
    true = np.eye(p.shape[1])[y]
    curve = risk_coverage(p, y)
    result = {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(p.shape[1]),
                                    average="macro", zero_division=0)),
        "nll": nll(p, y), "brier": float(np.square(p - true).sum(axis=1).mean()),
        "aurc": curve["aurc"],
    }
    for bins in (10, 15, 30):
        result[f"ece_{bins}"] = calibration_bins(p, y, bins)["ece"]
    for fraction in (0.5, 0.8, 1.0):
        result[f"risk_at_{int(100 * fraction)}"] = float(
            curve["risk"][int(np.ceil(len(y) * fraction)) - 1])
    return result


def per_class_metrics(probabilities, labels, sets):
    p, y = checked_probabilities(probabilities, labels)
    pred = p.argmax(axis=1)
    return [{"class_index": c, "support": int((y == c).sum()),
             "recall": float((pred[y == c] == c).mean()) if (y == c).any() else None,
             "set_coverage": float(sets[y == c, c].mean()) if (y == c).any() else None}
            for c in range(p.shape[1])]
