"""Write the README result tables from results/reference, or check they are current.

    python scripts/readme_tables.py           # rewrite the block in README.md
    python scripts/readme_tables.py --check   # exit 1 if README.md is out of date
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/reference"
START, END = "<!-- tables:start -->", "<!-- tables:end -->"
NAMES = {"prior": "Class prior", "logistic": "Logistic regression",
         "hist_boost": "Histogram boosting", "mlp": "MLP",
         "mlp_temperature": "MLP + temperature", "ensemble": "Ensemble of 3 MLPs",
         "ensemble_temperature": "Ensemble + temperature"}
COLUMNS = (("accuracy", "Accuracy"), ("nll", "NLL"), ("brier", "Brier"), ("ece_15", "ECE"),
           ("aurc", "AURC"), ("set_coverage", "Set coverage"), ("set_size", "Set size"))
DOMAINS = (("eval_in", "In-domain evaluation (`eval_in`)"), ("eval_out", "Shifted evaluation (`eval_out`)"))


def day(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def cell(row, metric, digits=3):
    mean, count = row[f"{metric}_mean"], row[f"{metric}_count"]
    if count == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"


def build():
    metadata = json.loads((RUN / "data_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RUN / "summary.csv").set_index(["method", "domain"])
    deltas = pd.read_csv(RUN / "paired_deltas.csv")
    diagnostic = pd.read_csv(RUN / "tree-diagnostic/metrics.csv").set_index(["variant", "domain"])
    out = ["**Selected data** (after removing duplicated observations)", "",
           "| Split | Used for | Rows | Dates (UTC) | Climate zones |", "| --- | --- | ---: | --- | --- |"]
    uses = {"train": "fitting", "validation": "MLP early stopping", "temperature": "fitting the temperature",
            "conformal": "conformal threshold", "eval_in": "in-domain evaluation", "eval_out": "shifted evaluation"}
    for split, use in uses.items():
        record = metadata["splits"][split]
        out.append(f"| `{split}` | {use} | {record['n']} | {day(record['time_min'])} to {day(record['time_max'])} "
                   f"| {', '.join(sorted(record['climate_counts']))} |")
    for domain, title in DOMAINS:
        out += ["", f"**{title}**", "",
                "| Predictor | Fits | " + " | ".join(label for _, label in COLUMNS) + " |",
                "| --- | ---: | " + " | ".join("---:" for _ in COLUMNS) + " |"]
        for method, name in NAMES.items():
            row = summary.loc[(method, domain)]
            out.append(f"| {name} | {int(row['nll_count'])} | " + " | ".join(cell(row, m) for m, _ in COLUMNS) + " |")
    out += ["", "**Paired differences over the five outer seeds** (first minus second; negative NLL/ECE is better)", "",
            "| Comparison | Domain | NLL | ECE | Accuracy | Seeds with lower NLL |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for comparison in ("mlp_temperature minus mlp", "ensemble minus mlp", "ensemble_temperature minus ensemble"):
        first, second = comparison.split(" minus ")
        for domain, _ in DOMAINS:
            rows = deltas[(deltas.comparison == comparison) & (deltas.domain == domain)]
            out.append(f"| {NAMES[first]} vs {NAMES[second]} | `{domain}` "
                       f"| {rows.nll.mean():+.4f} ± {rows.nll.std():.4f} "
                       f"| {rows.ece_15.mean():+.4f} ± {rows.ece_15.std():.4f} "
                       f"| {rows.accuracy.mean():+.4f} ± {rows.accuracy.std():.4f} "
                       f"| {int((rows.nll < 0).sum())} of {len(rows)} |")
    out += ["", "**Post-hoc histogram-boosting diagnostic** (exploratory, one fit each)", "",
            "| Variant | Domain | Accuracy | NLL | ECE | Set coverage |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for variant, name in (("original_l2_0", "L2 = 0 (primary configuration)"), ("diagnostic_l2_1", "L2 = 1 (added afterwards)")):
        for domain, _ in DOMAINS:
            row = diagnostic.loc[(variant, domain)]
            out.append(f"| {name} | `{domain}` | {row.accuracy:.3f} | {row.nll:.3f} | {row.ece_15:.3f} | {row.set_coverage:.3f} |")
    return "\n".join(out)


def main():
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    before, rest = text.split(START)
    _, after = rest.split(END)
    updated = f"{before}{START}\n{build()}\n{END}{after}"
    if "--check" in sys.argv:
        if updated != text:
            sys.exit("README.md tables differ from results/reference; run scripts/readme_tables.py")
        print("README.md tables match results/reference")
    else:
        readme.write_text(updated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
