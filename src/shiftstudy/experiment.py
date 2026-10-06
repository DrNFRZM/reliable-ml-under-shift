"""Run the fixed protocol and recompute all evaluation from saved predictions."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .calibration import (conformal_sets, conformal_threshold, fit_temperature,
                          set_metrics, temperature_scale)
from .config import validate_config
from .data import fit_preprocessor, load_prepared, sha256_file
from .metrics import calibration_bins, per_class_metrics, prediction_metrics, risk_coverage
from .models import aligned_probabilities, member_seed, train_baselines, train_mlp

METHODS = ("prior", "logistic", "hist_boost", "mlp", "mlp_temperature",
           "ensemble", "ensemble_temperature")
PREDICTION_SPLITS = ("temperature", "conformal", "eval_in", "eval_out")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_provenance():
    package = Path(__file__).parent
    hashes = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=package,
                                            stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "package_files": hashes, "package_sha256": digest}


def run_experiment(data_path, output, config):
    validate_config(config)
    arrays, metadata = load_prepared(data_path)
    if metadata["preparation"] != config["preparation"]:
        raise ValueError("Experiment preparation config differs from the prepared data")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    source_at_start = source_provenance()
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "data_metadata.json", metadata)
    splits = ("train", "validation", *PREDICTION_SPLITS)
    n_classes = len(metadata["classes"])
    # Compact membership evidence, including training and stopping rows. Features
    # remain in the ignored prepared dataset; no raw weather CSV is redistributed.
    np.savez_compressed(output / "selection.npz", **{
        f"{s}_{suffix}": arrays[f"{s}_{suffix}"]
        for s in splits for suffix in ("rows", "observation_keys", "y")})
    with threadpool_limits(config["threads"]):
        preprocessor = fit_preprocessor(arrays["train_x"])
        x = {s: preprocessor.transform(arrays[f"{s}_x"]) for s in splits}
        y = {s: arrays[f"{s}_y"] for s in splits}
        if any(not np.isfinite(value).all() for value in x.values()):
            raise ValueError("Preprocessing produced non-finite inputs")
        np.savez_compressed(output / "preprocessing.npz",
            median=preprocessor[0].statistics_, mean=preprocessor[1].mean_,
            scale=preprocessor[1].scale_)
        print("Fitting prior, logistic and histogram-boosting baselines...", flush=True)
        baselines, prior, base_diagnostics = train_baselines(
            x["train"], y["train"], config["baselines"], n_classes)
        base_predictions = {"prior": {s: np.broadcast_to(prior, (len(x[s]), n_classes)).astype(np.float32)
                                      for s in PREDICTION_SPLITS}}
        for name, model in baselines.items():
            base_predictions[name] = {s: aligned_probabilities(model, x[s], n_classes).astype(np.float32)
                                      for s in PREDICTION_SPLITS}
        diagnostics = {"baselines": base_diagnostics, "seeds": {}}
        for seed in config["seeds"]:
            members, training = [], []
            for index in range(config["ensemble_size"]):
                print(f"Outer seed {seed}, member {index + 1}/{config['ensemble_size']}...", flush=True)
                model, record = train_mlp(x["train"], y["train"], x["validation"],
                    y["validation"], config["mlp"], member_seed(seed, index), n_classes)
                members.append({s: aligned_probabilities(model, x[s], n_classes).astype(np.float32)
                                for s in PREDICTION_SPLITS})
                training.append(record)
            member_predictions = {s: np.stack([m[s] for m in members]) for s in PREDICTION_SPLITS}
            predictions = dict(base_predictions)
            predictions["mlp"] = {s: member_predictions[s][0] for s in PREDICTION_SPLITS}
            predictions["ensemble"] = {s: member_predictions[s].mean(axis=0, dtype=np.float64).astype(np.float32)
                                       for s in PREDICTION_SPLITS}
            temperature_fits = {}
            for name in ("mlp", "ensemble"):
                fit = fit_temperature(predictions[name]["temperature"], y["temperature"])
                temperature_fits[name] = fit
                predictions[name + "_temperature"] = {
                    s: temperature_scale(predictions[name][s], fit["temperature"]).astype(np.float32)
                    for s in PREDICTION_SPLITS}
            saved = {f"{method}__{s}": p for method, values in predictions.items() for s, p in values.items()}
            saved.update({f"members__{s}": p for s, p in member_predictions.items()})
            saved.update({f"labels__{s}": y[s] for s in PREDICTION_SPLITS})
            saved.update({f"rows__{s}": arrays[f"{s}_rows"] for s in PREDICTION_SPLITS})
            np.savez_compressed(output / f"seed_{seed}.npz", **saved)
            diagnostics["seeds"][str(seed)] = {"members": training, "temperature_fits": temperature_fits}
            write_json(output / "training.json", diagnostics)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "threadpoolctl")}
    lock = Path(__file__).resolve().parents[2] / "requirements-lock.txt"
    if lock.is_file():
        (output / "requirements-lock.txt").write_bytes(lock.read_bytes())
        versions.update({line.split("==")[0]: importlib.metadata.version(line.split("==")[0])
                         for line in lock.read_text().splitlines() if "==" in line and not line.startswith("#")})
    if source_provenance()["package_sha256"] != source_at_start["package_sha256"]:
        raise RuntimeError("Package changed during training; run cannot be marked complete")
    write_json(output / "manifest.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_kind": metadata["kind"], "source": source_at_start,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "threads": config["threads"],
        "runner": {"machine": platform.machine(), "logical_cpus": os.cpu_count(),
                   "github_repository": os.environ.get("GITHUB_REPOSITORY"),
                   "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                   "github_sha": os.environ.get("GITHUB_SHA")},
        "prediction_dtype": "float32; metrics recomputed in float64",
        "baseline_fits": 2, "neural_fits": len(config["seeds"]) * config["ensemble_size"],
        "prediction_hashes": {p.name: sha256_file(p) for p in sorted(output.glob("seed_*.npz"))},
        "artifact_hashes": {name: sha256_file(output / name) for name in
                            ("config.json", "data_metadata.json", "training.json", "preprocessing.npz", "selection.npz")
                            + (("requirements-lock.txt",) if lock.is_file() else ())},
    })
    evaluate_saved(output)
    return output


def evaluate_saved(run_path):
    """Verify hashes and paired averaging, then evaluate without refitting any model."""
    run_path = Path(run_path)
    manifest = json.loads((run_path / "manifest.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        if sha256_file(run_path / name) != expected:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
    config = validate_config(json.loads((run_path / "config.json").read_text()))
    metadata = json.loads((run_path / "data_metadata.json").read_text())
    diagnostics = json.loads((run_path / "training.json").read_text())
    rows, per_class, calibration, curves, thresholds = [], [], [], [], []
    for seed in config["seeds"]:
        name = f"seed_{seed}.npz"
        if sha256_file(run_path / name) != manifest["prediction_hashes"][name]:
            raise ValueError(f"Prediction hash mismatch: {name}")
        with np.load(run_path / name, allow_pickle=False) as saved:
            for split in PREDICTION_SPLITS:
                if not np.array_equal(saved[f"mlp__{split}"], saved[f"members__{split}"][0]):
                    raise ValueError("Single model is not ensemble member zero")
                expected = saved[f"members__{split}"].mean(axis=0, dtype=np.float64).astype(np.float32)
                if not np.array_equal(expected, saved[f"ensemble__{split}"]):
                    raise ValueError("Ensemble differs from saved member mean")
            for method in METHODS:
                if method.endswith("_temperature"):
                    base = method.removesuffix("_temperature")
                    temperature = diagnostics["seeds"][str(seed)]["temperature_fits"][base]["temperature"]
                    for split in PREDICTION_SPLITS:
                        expected = temperature_scale(saved[f"{base}__{split}"], temperature).astype(np.float32)
                        if not np.array_equal(expected, saved[f"{method}__{split}"]):
                            raise ValueError("Scaled probabilities disagree with saved temperature")
                q = conformal_threshold(saved[f"{method}__conformal"],
                                        saved["labels__conformal"], config["alpha"])
                thresholds.append({"seed": seed, "method": method, "threshold": q})
                for domain in ("eval_in", "eval_out"):
                    p, y = saved[f"{method}__{domain}"], saved[f"labels__{domain}"]
                    sets = conformal_sets(p, q)
                    group = {"seed": seed, "method": method, "domain": domain}
                    rows.append({**group, **prediction_metrics(p, y), **set_metrics(sets, y)})
                    for item in per_class_metrics(p, y, sets):
                        item["class_label"] = metadata["classes"][item["class_index"]]
                        per_class.append({**group, **item})
                    b = calibration_bins(p, y, 15)
                    for i in range(15):
                        calibration.append({**group, "bin": i, **{k: b[k][i] for k in
                            ("count", "confidence", "accuracy", "lower", "upper")}})
                    c = risk_coverage(p, y)
                    indices = np.unique(np.ceil(np.linspace(1, len(y), 101)).astype(int) - 1)
                    for i in indices:
                        curves.append({**group, **{k: c[k][i] for k in
                            ("coverage", "risk", "random_risk", "oracle_risk")}})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_path / "metrics.csv", index=False)
    for name, values in (("per_class", per_class), ("calibration_bins", calibration),
                         ("risk_coverage", curves), ("conformal_thresholds", thresholds)):
        pd.DataFrame(values).to_csv(run_path / f"{name}.csv", index=False)
    columns = [c for c in metrics.columns if c not in ("seed", "method", "domain")]
    summary = metrics.groupby(["method", "domain"])[columns].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(run_path / "summary.csv", index=False)
    contrasts = (("mlp_temperature", "mlp"), ("ensemble", "mlp"),
                 ("ensemble_temperature", "ensemble"), ("hist_boost", "mlp"))
    deltas = []
    for target, reference in contrasts:
        a = metrics[metrics.method == target].set_index(["seed", "domain"])[columns]
        b = metrics[metrics.method == reference].set_index(["seed", "domain"])[columns]
        diff = (a - b).reset_index()
        diff["comparison"] = f"{target} minus {reference}"
        deltas.append(diff)
    paired = pd.concat(deltas, ignore_index=True)
    paired.to_csv(run_path / "paired_deltas.csv", index=False)
    paired_summary = paired.groupby(["comparison", "domain"])[columns].agg(["mean", "std"])
    paired_summary.columns = [f"{metric}_{stat}" for metric, stat in paired_summary.columns]
    paired_summary.reset_index().to_csv(run_path / "paired_summary.csv", index=False)
    return metrics
