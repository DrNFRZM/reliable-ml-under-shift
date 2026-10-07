"""Check a saved run without using the package's own metric code.

Imports nothing from shiftstudy. It checks the file hashes and split
bookkeeping, then recomputes every table from the saved probabilities and
labels with NumPy, SciPy and scikit-learn and compares against the CSV files.

    python scripts/verify_reference.py results/reference
"""

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

FIXED = ("prior", "logistic", "hist_boost")
SEEDED = ("mlp", "mlp_temperature", "ensemble", "ensemble_temperature")
PREDICTION_SPLITS = ("temperature", "conformal", "eval_in", "eval_out")
ALL_SPLITS = ("train", "validation", *PREDICTION_SPLITS)
FLOAT32_ROUNDING = 6e-8   # probabilities are stored as float32
TOLERANCE = 1e-10         # recomputed float64 metrics against the CSV values
failures = []


def check(name, passed, detail=""):
    if not passed:
        failures.append(name)
    print(("ok    " if passed else "FAIL  ") + name + (f" ({detail})" if detail else ""))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_class(p, y):
    return p[np.arange(len(y)), y]


def ece(p, y, bins):
    confidence, correct = p.max(axis=1), p.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    total, table = 0.0, []
    for b in range(bins):
        inside = index == b
        n = int(inside.sum())
        conf, acc = (confidence[inside].mean(), correct[inside].mean()) if n else (0.0, 0.0)
        total += n * abs(conf - acc)
        table.append((n, conf, acc))
    return total / len(y), table


def selective_risk(p, y):
    """Risk after keeping the k most confident rows, averaged over the order inside ties."""
    errors = (p.argmax(axis=1) != y).astype(float)
    groups = (pd.DataFrame({"confidence": p.max(axis=1), "error": errors})
              .groupby("confidence")["error"].agg(["sum", "size"]).iloc[::-1])
    risk, errors_before, kept_before = [], 0.0, 0
    for block_errors, size in zip(groups["sum"].to_numpy(), groups["size"].to_numpy()):
        k = np.arange(1, size + 1)
        risk.append((errors_before + k * block_errors / size) / (kept_before + k))
        errors_before += block_errors
        kept_before += size
    return np.concatenate(risk), errors


def conformal_quantile(p, y, alpha):
    scores = np.sort(1.0 - true_class(p, y))
    rank = math.ceil((len(y) + 1) * (1 - alpha))
    return 1.0 if rank > len(y) else float(scores[rank - 1])


def scale(p, temperature):
    return np.exp(log_softmax(np.log(np.maximum(p.astype(np.float64), 1e-15)) / temperature, axis=1))


def point_metrics(p, y, threshold, n_classes):
    n, prediction, sets = len(y), p.argmax(axis=1), (1.0 - p) <= threshold
    labels = np.arange(n_classes)
    risk, _ = selective_risk(p, y)
    row = {"accuracy": accuracy_score(y, prediction),
           "macro_f1": f1_score(y, prediction, labels=labels, average="macro", zero_division=0),
           "nll": float(-np.log(np.maximum(true_class(p, y), 1e-15)).mean()),
           "brier": float(((p - np.eye(n_classes)[y]) ** 2).sum(axis=1).mean()),
           "aurc": float(risk.mean())}
    for bins in (10, 15, 30):
        row[f"ece_{bins}"] = ece(p, y, bins)[0]
    for fraction in (0.5, 0.8, 1.0):
        row[f"risk_at_{int(100 * fraction)}"] = float(risk[math.ceil(n * fraction) - 1])
    size = sets.sum(axis=1)
    row.update(set_coverage=float(true_class(sets, y).mean()), set_size=float(size.mean()),
               empty_set_rate=float((size == 0).mean()))
    return row


def compare(name, reported, recomputed, keys):
    if len(reported) != len(recomputed) or set(reported.columns) != set(recomputed.columns):
        check(name, False, "row count or columns differ")
        return
    a = reported.sort_values(keys).reset_index(drop=True)
    b = recomputed.sort_values(keys).reset_index(drop=True)[a.columns]
    same_keys = all((a[k].astype("string").fillna("") == b[k].astype("string").fillna("")).all() for k in keys)
    values = [c for c in a.columns if c not in keys]
    x, y = a[values].to_numpy(float), b[values].to_numpy(float)
    same_blanks = np.array_equal(np.isnan(x), np.isnan(y))
    worst = float(np.nanmax(np.abs(x - y)))
    check(name, same_keys and same_blanks and worst <= TOLERANCE,
          f"{x.size} values, largest difference {worst:.1e}")


def read(run, name):
    frame = pd.read_csv(run / name)
    if "seed" in frame:
        frame["seed"] = frame["seed"].astype("Int64")
    return frame


def verify_files(run, manifest, config, metadata):
    for name, expected in {**manifest["artifact_hashes"], **manifest["prediction_hashes"]}.items():
        check(f"hash of {name}", sha256(run / name) == expected)
    check("prediction files cover the configured seeds",
          sorted(manifest["prediction_hashes"]) == [f"seed_{s}.npz" for s in config["seeds"]])
    check("prepared data used this configuration", metadata["preparation"] == config["preparation"])
    with np.load(run / "selection.npz", allow_pickle=False) as selection:
        seen = set()
        disjoint = True
        for split in ALL_SPLITS:
            n = metadata["splits"][split]["n"]
            rows, keys, y = (selection[f"{split}_{part}"] for part in ("rows", "observation_keys", "y"))
            raw = np.asarray(metadata["classes"])[y]
            counts = {str(c): int((raw == c).sum()) for c in metadata["classes"]}
            check(f"{split}: {n} rows, unique source rows, label counts as in metadata",
                  len(rows) == len(keys) == len(y) == n and len(np.unique(rows)) == n
                  and counts == metadata["splits"][split]["label_counts"])
            split_keys = set(map(tuple, keys))
            disjoint &= len(split_keys) == n and not (split_keys & seen)
            seen |= split_keys
        check("no (time, latitude, longitude) key is shared within or between splits", disjoint)
        latest_training = max(selection[f"{s}_observation_keys"][:, 0].max() for s in ALL_SPLITS[:5])
        if metadata["kind"] == "shifts_weather":
            check("every eval_out observation is later than all in-domain observations",
                  selection["eval_out_observation_keys"][:, 0].min() > latest_training)
        for seed in config["seeds"]:
            with np.load(run / f"seed_{seed}.npz", allow_pickle=False) as saved:
                check(f"seed {seed}: labels and row ids equal selection.npz",
                      all(np.array_equal(saved[f"labels__{s}"], selection[f"{s}_y"])
                          and np.array_equal(saved[f"rows__{s}"], selection[f"{s}_rows"])
                          for s in PREDICTION_SPLITS))


def verify_run(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "data_metadata.json").read_text(encoding="utf-8"))
    training = json.loads((run / "training.json").read_text(encoding="utf-8"))
    verify_files(run, manifest, config, metadata)

    n_classes, alpha = len(metadata["classes"]), config["alpha"]
    labels = np.arange(n_classes)
    metrics, per_class, bins, curves, thresholds = [], [], [], [], []
    worst = {"member": 0.0, "mean": 0.0, "scaled": 0.0, "temperature": 0.0}
    baselines = None
    for seed in config["seeds"]:
        saved = dict(np.load(run / f"seed_{seed}.npz", allow_pickle=False))
        seeds_used = [m["seed"] for m in training["seeds"][str(seed)]["members"]]
        expected = [int(np.random.SeedSequence([seed, i, 1741]).generate_state(1)[0])
                    for i in range(config["ensemble_size"])]
        check(f"seed {seed}: member seeds derive from the outer seed", seeds_used == expected)
        for split in PREDICTION_SPLITS:
            members = saved[f"members__{split}"]
            worst["member"] = max(worst["member"], float(np.abs(members[0] - saved[f"mlp__{split}"]).max()))
            mean = members.astype(np.float64).mean(axis=0)
            worst["mean"] = max(worst["mean"], float(np.abs(mean - saved[f"ensemble__{split}"]).max()))
        for base in ("mlp", "ensemble"):
            recorded = training["seeds"][str(seed)]["temperature_fits"][base]["temperature"]
            for split in PREDICTION_SPLITS:
                difference = np.abs(scale(saved[f"{base}__{split}"], recorded) - saved[f"{base}_temperature__{split}"])
                worst["scaled"] = max(worst["scaled"], float(difference.max()))
            # Refit on the temperature split only, optimising T directly rather than log T.
            p, y = saved[f"{base}__temperature"], saved["labels__temperature"]
            refit = minimize_scalar(lambda t: log_loss(y, scale(p, t), labels=labels),
                                    bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-9})
            worst["temperature"] = max(worst["temperature"], abs(refit.x - recorded))
        copies = {f"{m}__{s}": saved[f"{m}__{s}"] for m in FIXED for s in PREDICTION_SPLITS}
        if baselines is None:
            baselines, methods = copies, FIXED + SEEDED
        else:
            check(f"seed {seed}: baseline predictions are the same single fit as in the first seed file",
                  all(np.array_equal(baselines[k], copies[k]) for k in copies))
            methods = SEEDED
        for method in methods:
            fit = pd.NA if method in FIXED else seed
            q = conformal_quantile(saved[f"{method}__conformal"].astype(np.float64), saved["labels__conformal"], alpha)
            thresholds.append({"seed": fit, "method": method, "threshold": q})
            for domain in ("eval_in", "eval_out"):
                p, y = saved[f"{method}__{domain}"].astype(np.float64), saved[f"labels__{domain}"]
                group = {"seed": fit, "method": method, "domain": domain}
                metrics.append({**group, **point_metrics(p, y, q, n_classes)})
                sets, prediction = (1.0 - p) <= q, p.argmax(axis=1)
                recall = recall_score(y, prediction, labels=labels, average=None, zero_division=0)
                for c in range(n_classes):
                    present = y == c
                    per_class.append({**group, "class_index": c, "support": int(present.sum()),
                                      "recall": float(recall[c]) if present.any() else np.nan,
                                      "set_coverage": float(sets[present, c].mean()) if present.any() else np.nan,
                                      "class_label": metadata["classes"][c]})
                for i, (count, conf, acc) in enumerate(ece(p, y, 15)[1]):
                    bins.append({**group, "bin": i, "count": count, "confidence": conf, "accuracy": acc,
                                 "lower": i / 15, "upper": (i + 1) / 15})
                risk, errors = selective_risk(p, y)
                oracle = np.cumsum(np.sort(errors)) / np.arange(1, len(y) + 1)
                for point, i in enumerate(np.unique(np.ceil(np.linspace(1, len(y), 101)).astype(int) - 1)):
                    curves.append({**group, "point": point, "coverage": (i + 1) / len(y), "risk": risk[i],
                                   "random_risk": errors.mean(), "oracle_risk": oracle[i]})

    check("single MLP is ensemble member 0", worst["member"] == 0)
    check("ensemble is the mean of its three members", worst["mean"] <= FLOAT32_ROUNDING, f"{worst['mean']:.1e}")
    check("scaled probabilities follow from the recorded temperatures",
          worst["scaled"] <= FLOAT32_ROUNDING, f"{worst['scaled']:.1e}")
    check("temperatures refitted on the temperature split agree with the recorded ones",
          worst["temperature"] <= 1e-4, f"{worst['temperature']:.1e}")

    metrics = pd.DataFrame(metrics)
    metrics["seed"] = metrics["seed"].astype("Int64")
    group_keys = ["seed", "method", "domain"]
    compare("metrics.csv", read(run, "metrics.csv"), metrics, group_keys)
    compare("per_class.csv", read(run, "per_class.csv"), pd.DataFrame(per_class), group_keys + ["class_index"])
    compare("calibration_bins.csv", read(run, "calibration_bins.csv"), pd.DataFrame(bins), group_keys + ["bin"])
    compare("conformal_thresholds.csv", read(run, "conformal_thresholds.csv"), pd.DataFrame(thresholds),
            ["seed", "method"])
    reported = read(run, "risk_coverage.csv")
    reported["point"] = reported.groupby(group_keys, dropna=False).cumcount()
    compare("risk_coverage.csv", reported, pd.DataFrame(curves), group_keys + ["point"])

    values = [c for c in metrics.columns if c not in group_keys]
    summary = metrics.groupby(["method", "domain"])[values].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    reported = read(run, "summary.csv")
    compare("summary.csv", reported, summary.reset_index(), ["method", "domain"])
    single = reported[reported.method.isin(FIXED)]
    check("baselines are summarised as one fit with no standard deviation",
          (single["nll_count"] == 1).all() and single["nll_std"].isna().all())
    check("MLP-based predictors are summarised over the outer seeds",
          (reported[reported.method.isin(SEEDED)]["nll_count"] == len(config["seeds"])).all())

    deltas = []
    for target, reference in (("mlp_temperature", "mlp"), ("ensemble", "mlp"),
                              ("ensemble_temperature", "ensemble"), ("hist_boost", "mlp")):
        b = metrics[metrics.method == reference]
        on = ["domain"] if target in FIXED else ["seed", "domain"]
        a = metrics[metrics.method == target].drop(columns=["seed"] if target in FIXED else [])
        both = b.merge(a, on=on, suffixes=("_reference", ""))
        difference = both[["seed", "domain"]].copy()
        for column in values:
            difference[column] = both[column] - both[column + "_reference"]
        difference["comparison"] = f"{target} minus {reference}"
        deltas.append(difference)
    deltas = pd.concat(deltas, ignore_index=True)
    compare("paired_deltas.csv", read(run, "paired_deltas.csv"), deltas, ["comparison", "seed", "domain"])
    paired = deltas.groupby(["comparison", "domain"])[values].agg(["mean", "std"])
    paired.columns = [f"{metric}_{stat}" for metric, stat in paired.columns]
    compare("paired_summary.csv", read(run, "paired_summary.csv"), paired.reset_index(), ["comparison", "domain"])
    return manifest, config, n_classes


def verify_diagnostic(directory, parent_manifest_path, alpha, n_classes):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    check("diagnostic: prediction hash", sha256(directory / "predictions.npz") == manifest["prediction_sha256"])
    check("diagnostic: refers to this primary run", sha256(parent_manifest_path) == manifest["parent_manifest_sha256"])
    check("diagnostic: recorded as not pre-specified", manifest["pre_specified"] is False)
    rows = []
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for variant in ("original_l2_0", "diagnostic_l2_1"):
            q = conformal_quantile(saved[f"{variant}__conformal"].astype(np.float64), saved["labels__conformal"], alpha)
            for domain in ("eval_in", "eval_out"):
                p, y = saved[f"{variant}__{domain}"].astype(np.float64), saved[f"labels__{domain}"]
                rows.append({"variant": variant, "domain": domain, **point_metrics(p, y, q, n_classes)})
    compare("tree-diagnostic/metrics.csv", pd.read_csv(directory / "metrics.csv"), pd.DataFrame(rows),
            ["variant", "domain"])


if __name__ == "__main__":
    run = Path(sys.argv[1])
    manifest, config, n_classes = verify_run(run)
    if (run / "tree-diagnostic").is_dir():
        verify_diagnostic(run / "tree-diagnostic", run / "manifest.json", config["alpha"], n_classes)
    print(f"\n{len(failures)} failed check(s)" if failures else "\nAll checks passed.")
    sys.exit(1 if failures else 0)
