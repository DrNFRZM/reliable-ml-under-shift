"""Small command-line interface; synthetic smoke results are explicitly labeled."""

from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json

import numpy as np
import pandas as pd

from .config import load_config
from .data import SPLITS, prepare_weather
from .experiment import evaluate_saved, run_experiment, write_json
from .report import generate_report


def prepare_synthetic(output, preparation):
    """Test fixture with generated covariate shift, not a real-world benchmark."""
    rng = np.random.default_rng(preparation["seed"])
    weights = rng.normal(size=(12, 3))
    offset = 0
    with TemporaryDirectory() as directory:
        for split in SPLITS:
            n = preparation["sizes"][split]
            x = rng.normal(size=(n, 12))
            if split == "eval_out":
                x[:, :3] += 1.5
            y = (x @ weights + rng.normal(scale=.5, size=(n, 3))).argmax(axis=1)
            frame = pd.DataFrame({"fact_time": np.arange(n) + 1500000000 + offset,
                                  "fact_latitude": np.zeros(n), "fact_longitude": np.zeros(n),
                                  "fact_temperature": np.zeros(n), "fact_cwsm_class": y,
                                  "climate": ["synthetic"] * n})
            for i in range(12):
                frame[f"forecast_{i}"] = x[:, i]
            frame.to_csv(Path(directory) / f"{split}.csv", index=False)
            offset += n
        metadata = prepare_weather(directory, output, preparation)
    metadata["kind"] = "synthetic_smoke"
    metadata["upstream_license"] = "Not applicable: generated test fixture"
    metadata["source_name"] = "generated synthetic CSV fixture"
    write_json(Path(output) / "metadata.json", metadata)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Small Shifts Weather calibration study")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Sample complete canonical CSVs from a tar or directory")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--config", required=True)
    run = sub.add_parser("run", help="Fit configured predictors and write verified measurements")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", required=True)
    evaluate = sub.add_parser("evaluate", help="Verify and re-evaluate saved predictions without fitting")
    evaluate.add_argument("--run", required=True)
    smoke = sub.add_parser("smoke", help="Run complete pipeline on synthetic test data only")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_weather(args.source, args.output, load_config(args.config)["preparation"])
        print(f"Prepared data: {args.output}")
    elif args.command == "run":
        run_experiment(args.data, args.output, load_config(args.config))
        print(generate_report(args.output))
    elif args.command == "evaluate":
        evaluate_saved(args.run)
        print(generate_report(args.run))
    else:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
        config = load_config(args.config)
        output.mkdir(parents=True)
        prepare_synthetic(output / "data", config["preparation"])
        run_experiment(output / "data", output / "experiment", config)
        print(generate_report(output / "experiment"))


if __name__ == "__main__":
    main()
