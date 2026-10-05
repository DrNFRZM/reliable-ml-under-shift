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
            original = pd.read_csv(path / "run_a/metrics.csv")
            recomputed = evaluate_saved(path / "run_a")
            pd.testing.assert_frame_equal(original, recomputed, check_exact=False, rtol=1e-14)
            generate_report(path / "run_a")
            self.assertIn("not evidence", (path / "run_a/report.md").read_text())
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
