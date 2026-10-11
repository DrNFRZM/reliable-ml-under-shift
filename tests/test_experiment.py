from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd

from shiftstudy.cli import prepare_synthetic
from shiftstudy.config import load_config, validate_config
from shiftstudy.experiment import evaluate_saved, run_experiment
from shiftstudy.report import generate_report

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_config_rejects_typos_invalid_values_and_duplicate_seeds(self):
        cfg = load_config(ROOT / "configs/smoke.json")
        changes = (("threads", 0), ("ensemble_size", True), ("seeds", [1, 1]), ("alpha", 1.))
        for name, value in changes:
            bad = deepcopy(cfg)
            bad[name] = value
            with self.assertRaises(ValueError):
                validate_config(bad)
        bad = deepcopy(cfg)
        bad["mlp"]["epoch_typo"] = 10
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_end_to_end_recomputation_reproducibility_and_corruption_detection(self):
        cfg = load_config(ROOT / "configs/smoke.json")
        with TemporaryDirectory() as temp:
            path = Path(temp)
            metadata = prepare_synthetic(path / "data", cfg["preparation"])
            self.assertEqual(metadata["kind"], "synthetic_smoke")
            run_experiment(path / "data", path / "run_a", cfg)
            with np.load(path / "run_a/selection.npz", allow_pickle=False) as selected:
                with np.load(path / "data/data.npz", allow_pickle=False) as prepared:
                    for split in ("train", "validation", "temperature", "conformal", "eval_in", "eval_out"):
                        for suffix in ("rows", "observation_keys", "y"):
                            np.testing.assert_array_equal(selected[f"{split}_{suffix}"], prepared[f"{split}_{suffix}"])
            csv_names = ("metrics", "summary", "paired_deltas", "per_class", "risk_coverage")
            original = {name: (path / f"run_a/{name}.csv").read_bytes() for name in csv_names}
            evaluate_saved(path / "run_a")
            for name in csv_names:
                self.assertEqual(original[name], (path / f"run_a/{name}.csv").read_bytes())
                self.assertNotIn(b"\r", original[name])
            # Baselines are one fit: one row per domain with no seed, count 1 and no SD.
            metrics = pd.read_csv(path / "run_a/metrics.csv")
            summary = pd.read_csv(path / "run_a/summary.csv").set_index(["method", "domain"])
            for method in ("prior", "logistic", "hist_boost"):
                rows = metrics[metrics.method == method]
                self.assertEqual(len(rows), 2)
                self.assertTrue(rows.seed.isna().all())
                self.assertEqual(summary.loc[(method, "eval_out"), "nll_count"], 1)
                self.assertTrue(np.isnan(summary.loc[(method, "eval_out"), "nll_std"]))
            for method in ("mlp", "ensemble_temperature"):
                self.assertEqual(sorted(metrics[metrics.method == method].seed.unique()), cfg["seeds"])
                self.assertEqual(summary.loc[(method, "eval_in"), "nll_count"], len(cfg["seeds"]))
            deltas = pd.read_csv(path / "run_a/paired_deltas.csv")
            tree = deltas[deltas.comparison == "hist_boost minus mlp"]
            self.assertEqual(len(tree), 2 * len(cfg["seeds"]))
            self.assertFalse(tree.nll.isna().any())
            generate_report(path / "run_a")
            self.assertIn("not evidence", (path / "run_a/report.md").read_text(encoding="utf-8"))
            run_experiment(path / "data", path / "run_b", cfg)
            for seed in cfg["seeds"]:
                with np.load(path / f"run_a/seed_{seed}.npz") as a, np.load(path / f"run_b/seed_{seed}.npz") as b:
                    self.assertEqual(a.files, b.files)
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            with self.assertRaises(FileExistsError):
                run_experiment(path / "data", path / "run_a", cfg)
            original_config = (path / "run_b/config.json").read_text()
            changed = json.loads(original_config)
            changed["alpha"] = .2
            (path / "run_b/config.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                evaluate_saved(path / "run_b")
            with (path / "run_a/seed_0.npz").open("ab") as f:
                f.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_saved(path / "run_a")


if __name__ == "__main__":
    unittest.main()
