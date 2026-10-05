import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from shiftstudy.cli import prepare_synthetic
from shiftstudy.config import load_config
from shiftstudy.experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tree_diagnostic", ROOT / "scripts/tree_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticTests(unittest.TestCase):
    def test_original_tree_matches_reference_and_diagnostic_is_recomputable(self):
        cfg = load_config(ROOT / "configs/smoke.json")
        with TemporaryDirectory() as directory:
            path = Path(directory)
            prepare_synthetic(path / "data", cfg["preparation"])
            run_experiment(path / "data", path / "reference", cfg)
            result = diagnostic.run(path / "data", path / "reference", path / "diagnostic")
            self.assertEqual(len(result), 4)
            self.assertEqual(result.to_json(), diagnostic.evaluate(path / "diagnostic").to_json())
            with self.assertRaises(FileExistsError):
                diagnostic.run(path / "data", path / "reference", path / "diagnostic")


if __name__ == "__main__":
    unittest.main()
