"""Generate descriptive tables and scientific plots from measured CSVs only."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .experiment import FIXED_METHODS, METHODS


def generate_report(run_path):
    run_path = Path(run_path)
    summary = pd.read_csv(run_path / "summary.csv")
    pairs = pd.read_csv(run_path / "paired_summary.csv")
    config = json.loads((run_path / "config.json").read_text())
    manifest = json.loads((run_path / "manifest.json").read_text())
    metadata = json.loads((run_path / "data_metadata.json").read_text())
    training = json.loads((run_path / "training.json").read_text())
    real_weather = manifest["dataset_kind"] == "shifts_weather"
    lines = ["# " + ("Shifts Weather subset study" if real_weather else "Synthetic smoke test"), "",
             "All numbers below were computed from this run's saved predictions.", ""]
    if not real_weather:
        lines += ["**This is synthetic test data, not evidence about real distribution shift.**", ""]
    lines += ["## Protocol and provenance", "",
              f"- Code commit: `{manifest['source']['git_commit']}`; dirty working tree: `{manifest['source']['git_dirty']}`.",
              f"- Package hash: `{manifest['source']['package_sha256']}`.",
              f"- Prepared data hash: `{metadata['data_sha256']}`.",
              f"- Completed: {manifest['completed_at_utc']}.",
              f"- Python {manifest['python']}; scikit-learn {manifest['versions']['scikit-learn']}; CPU threads: {config['threads']}.",
              f"- Outer seeds: {config['seeds']}; {config['ensemble_size']} neural fits per seed, {manifest['neural_fits']} total.",
              "- Separate validation, temperature-calibration, and conformal-calibration groups; no shifted-data tuning.",
              "- float32 predictions, float64 evaluation; NLL clips true-class probabilities at 1e-15.", "",
              "## Data support", "",
              "| Split | Observations | Missing feature fraction |",
              "| --- | ---: | ---: |"]
    for name, record in metadata["splits"].items():
        lines.append(f"| {name} | {record['n']} | {record['missing_feature_fraction']:.4f} |")
    lines += ["", "Class codes are upstream numeric labels; no unverified human-readable class names are assigned.", "",
              "| Class | Train | Validation | Temperature | Conformal | Eval in | Eval out |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in metadata["classes"]:
        counts = [metadata["splits"][s]["label_counts"][str(label)] for s in
                  ("train", "validation", "temperature", "conformal", "eval_in", "eval_out")]
        lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    if real_weather:
        audit = metadata["observation_audit"]
        lines += ["", f"Observation-key cleanup removed: `{audit['removed_rows']}`.",
                  "The retained samples have no shared exact time/latitude/longitude keys.",
                  "The candidate sampling was uniform over rows; cleanup does not make the retained sample uniform over unique stations.",
                  f"Non-observational training rows excluded before sampling: {metadata['canonical_sources']['train']['excluded_rows']}."]
    fields = ("accuracy", "macro_f1", "nll", "ece_15", "aurc", "set_coverage", "set_size")
    for domain in ("eval_in", "eval_out"):
        lines += ["", f"## {domain}", "",
                  "MLP-based rows: mean ± sample SD over the outer optimization seeds, conditional on this one data subset.",
                  "Prior, logistic and hist_boost rows: one fit each, so a single value and no SD.", "",
                  "| Predictor | Accuracy | Macro F1 | NLL | ECE 15 | AURC | Set coverage | Set size |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in METHODS:
            row = summary[(summary.method == name) & (summary.domain == domain)].iloc[0]
            values = [f"{row[f'{field}_mean']:.4f}" if name in FIXED_METHODS else
                      f"{row[f'{field}_mean']:.4f} ± {row[f'{field}_std']:.4f}" for field in fields]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "## Paired NLL changes", "",
              "Negative means lower NLL for the first method. These are descriptive seed differences, not significance tests.",
              "hist_boost is a single fit, so the SD of its contrast reflects MLP seed variation only.", "",
              "| Contrast | Domain | NLL change ± sample SD |", "| --- | --- | ---: |"]
    for _, row in pairs.iterrows():
        lines.append(f"| {row.comparison} | {row.domain} | {row.nll_mean:+.4f} ± {row.nll_std:.4f} |")
    members = [m for record in training["seeds"].values() for m in record["members"]]
    fit_seconds = sum(m["fit_seconds"] for m in members) + sum(
        r["fit_seconds"] for r in training["baselines"].values())
    warnings = [(name, w) for name, r in training["baselines"].items() for w in r["warnings"]]
    warnings += [(f"member seed {m['seed']}", w) for m in members for w in m["warnings"]]
    lines += ["", "## Training diagnostics", "",
              f"Total recorded model fit time: {fit_seconds:.1f} seconds on the recorded environment (not a hardware benchmark).",
              f"Selected MLP epochs range from {min(m['selected_epoch'] for m in members)} to {max(m['selected_epoch'] for m in members)}.",
              f"Members that reached the epoch budget: {sum(m['reached_epoch_budget'] for m in members)}/{len(members)}.",
              f"Recorded fitting warnings: {len(warnings)}."]
    lines += [f"- {name}: {message}" for name, message in warnings]
    lines += ["", "## Interpretation limits", "",
              "- Prior, logistic and tree baselines were each fitted once. They carry no seed-to-seed SD, which says nothing about how stable they would be under a different data subset.",
              "- Calibration set coverage targets 0.9 marginally under exchangeability; weather dependence and domain shift prevent an unconditional coverage guarantee here.",
              "- Coverage can conceal failures for rare classes. Some class/domain cells have few or no examples; inspect per_class.csv, which leaves unsupported recall/coverage blank.",
              "- Macro F1 uses the fixed full-training vocabulary, including classes absent from an evaluation sample (zero contribution).",
              "- ECE depends on bins; metrics.csv also includes 10- and 30-bin estimates. NLL is the primary probability-quality comparison.",
              "- AURC is mean selective 0–1 risk over retained counts. It is not the Shifts paper's error-replacement R-AUC; these numbers are not comparable to its leaderboard.",
              "- Rejection curves average order within exact confidence ties. They are retrospective evaluations, not tuned deployment policies.",
              "- The ensemble uses three fits and shares member zero with the single model; this controls architecture and pairing, not compute.",
              "- One subset, fixed hyperparameters, few optimization seeds, no station-held-out validation, no novelty claim, and no broad superiority claim.", "",
              "## Figures", "", "![Calibration for first outer seed](calibration.png)", "",
              "![Selective risk averaged across outer seeds](risk_coverage.png)", "",
              "## Re-evaluation", "",
              "Run `shift-study evaluate --run <this-directory>` to verify prediction hashes, ensemble averaging, temperature transforms, and recompute metrics/figures without training.",
              "The commit above is the code that trained the models and saved the predictions; the tables are what the currently installed evaluator computes from them.",
              "Full metric values are in metrics.csv and summary.csv; paired_deltas.csv preserves each outer-seed difference.",
              "risk_coverage.csv contains 101 display points per curve; AURC uses every retained count, reconstructible from the NPZ probabilities.", ""]
    (run_path / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    plot_results(run_path, config["seeds"][0])
    return run_path / "report.md"


def plot_results(run_path, first_seed):
    calibration = pd.read_csv(run_path / "calibration_bins.csv")
    curves = pd.read_csv(run_path / "risk_coverage.csv")
    shown = ("logistic", "hist_boost", "mlp", "mlp_temperature", "ensemble", "ensemble_temperature")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, domain in zip(axes, ("eval_in", "eval_out")):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="ideal reference")
        for name in shown:
            data = calibration[((calibration.seed == first_seed) | calibration.seed.isna()) & (calibration.domain == domain)
                               & (calibration.method == name) & (calibration['count'] > 0)]
            ax.plot(data.confidence, data.accuracy, marker=".", label=name, linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
               ylabel="Empirical accuracy in bin", title=f"{domain}, outer seed {first_seed}")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.savefig(run_path / "calibration.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, domain in zip(axes, ("eval_in", "eval_out")):
        for name in shown:
            data = curves[(curves.domain == domain) & (curves.method == name)]
            means = data.groupby("coverage")["risk"].mean()
            ax.plot(means.index, means.values, label=name, linewidth=1.4)
        reference = curves[(curves.domain == domain) & (curves.method == "mlp")]
        oracle = reference.groupby("coverage")["oracle_risk"].mean()
        random = reference.groupby("coverage")["random_risk"].mean()
        ax.plot(oracle.index, oracle.values, "k--", label="MLP oracle rejection")
        ax.plot(random.index, random.values, "k:", label="MLP random rejection")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage (fraction retained)",
               ylabel="Selective error risk", title=f"{domain}, mean across outer seeds")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.savefig(run_path / "risk_coverage.png", dpi=160)
    plt.close(fig)
