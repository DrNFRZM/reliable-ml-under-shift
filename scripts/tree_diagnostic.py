"""Post-hoc fixed-L2 diagnostic. This is exploratory, not the primary experiment."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from shiftstudy.calibration import conformal_sets, conformal_threshold, set_metrics
from shiftstudy.data import fit_preprocessor, load_prepared, sha256_file
from shiftstudy.experiment import source_provenance, write_json
from shiftstudy.metrics import nll, prediction_metrics
from shiftstudy.models import aligned_probabilities


def evaluate(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if sha256_file(output / "predictions.npz") != manifest["prediction_sha256"]:
        raise ValueError("Diagnostic prediction hash mismatch")
    rows = []
    with np.load(output / "predictions.npz", allow_pickle=False) as saved:
        for variant in ("original_l2_0", "diagnostic_l2_1"):
            q = conformal_threshold(saved[f"{variant}__conformal"],
                                    saved["labels__conformal"], manifest["alpha"])
            for domain in ("eval_in", "eval_out"):
                p, y = saved[f"{variant}__{domain}"], saved[f"labels__{domain}"]
                rows.append({"variant": variant, "domain": domain,
                             **prediction_metrics(p, y), **set_metrics(conformal_sets(p, q), y)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    return result


def run(data_path, reference, output):
    reference, output = Path(reference), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output}")
    config = json.loads((reference / "config.json").read_text())
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    arrays, metadata = load_prepared(data_path)
    if metadata["data_sha256"] != json.loads((reference / "data_metadata.json").read_text())["data_sha256"]:
        raise ValueError("Diagnostic must use the exact primary-study subset")
    if sha256_file(reference / "seed_0.npz") != reference_manifest["prediction_hashes"]["seed_0.npz"]:
        raise ValueError("Original prediction hash mismatch")
    saved, fits = {}, {}
    splits = ("train", "validation", "conformal", "eval_in", "eval_out")
    hp = config["baselines"]
    with threadpool_limits(config["threads"]):
        transform = fit_preprocessor(arrays["train_x"])
        x = {s: transform.transform(arrays[f"{s}_x"]) for s in splits}
        for regularization in (0., 1.):
            name = "original_l2_0" if regularization == 0 else "diagnostic_l2_1"
            model = HistGradientBoostingClassifier(max_iter=hp["boost_iterations"],
                max_leaf_nodes=hp["boost_leaf_nodes"], learning_rate=hp["boost_learning_rate"],
                l2_regularization=regularization, early_stopping=False, random_state=0)
            started = time.perf_counter()
            model.fit(x["train"], arrays["train_y"])
            fits[name] = {"fit_seconds": time.perf_counter() - started,
                "l2_regularization": regularization,
                "training_nll": nll(aligned_probabilities(model, x["train"], len(metadata["classes"])), arrays["train_y"]),
                "validation_nll": nll(aligned_probabilities(model, x["validation"], len(metadata["classes"])), arrays["validation_y"])}
            for split in ("conformal", "eval_in", "eval_out"):
                saved[f"{name}__{split}"] = aligned_probabilities(model, x[split], len(metadata["classes"])).astype(np.float32)
                if regularization == 0:
                    with np.load(reference / "seed_0.npz", allow_pickle=False) as original:
                        np.testing.assert_array_equal(saved[f"{name}__{split}"], original[f"hist_boost__{split}"])
    saved.update({f"labels__{s}": arrays[f"{s}_y"] for s in ("conformal", "eval_in", "eval_out")})
    output.mkdir(parents=True)
    np.savez_compressed(output / "predictions.npz", **saved)
    write_json(output / "manifest.json", {
        "pre_specified": False,
        "purpose": "Fixed L2=1 diagnostic added after inspecting the primary run; not confirmatory ranking evidence",
        "source": source_provenance(), "script_sha256": sha256_file(__file__),
        "parent_manifest_sha256": sha256_file(reference / "manifest.json"),
        "prepared_data_sha256": metadata["data_sha256"],
        "alpha": config["alpha"], "baseline_configuration": hp, "fits": fits,
        "prediction_sha256": sha256_file(output / "predictions.npz")})
    return evaluate(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("run")
    fit.add_argument("--data", required=True)
    fit.add_argument("--reference", required=True)
    fit.add_argument("--output", required=True)
    check = commands.add_parser("evaluate")
    check.add_argument("--run", required=True)
    args = parser.parse_args()
    result = run(args.data, args.reference, args.output) if args.command == "run" else evaluate(args.run)
    print(result[["variant", "domain", "accuracy", "nll", "set_coverage"]].to_string(index=False))
