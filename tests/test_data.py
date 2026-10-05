import hashlib
import io
import unittest

import numpy as np
import pandas as pd

from shiftstudy.data import (META, SPLITS, clean_observations, feature_columns, fit_preprocessor, integer_labels,
                             sample_csv, validate_observations)


def csv_bytes(n=100):
    frame = pd.DataFrame({
        "fact_time": np.arange(n) + 1500000000,
        "fact_latitude": np.ones(n), "fact_longitude": np.ones(n),
        "fact_temperature": np.zeros(n), "fact_cwsm_class": np.arange(n) % 3,
        "climate": ["dry"] * n, "forecast": np.arange(n, dtype=float)})
    return frame.to_csv(index=False).encode()


class DataTests(unittest.TestCase):
    def test_uniform_priorities_chunk_independent_and_not_prefix(self):
        raw = csv_bytes()
        a, ma = sample_csv(io.BytesIO(raw), 15, 2, chunk_rows=7)
        b, mb = sample_csv(io.BytesIO(raw), 15, 2, chunk_rows=60)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(ma, mb)
        self.assertEqual(ma["csv_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(a.source_row.unique()), 15)
        self.assertGreater(a.source_row.max(), 50)

    def test_sampling_matches_independent_full_sort(self):
        raw = csv_bytes()
        a, _ = sample_csv(io.BytesIO(raw), 10, 42, chunk_rows=13)
        expected = np.sort(np.argsort(np.random.default_rng(42).random(100))[:10])
        np.testing.assert_array_equal(a.source_row, expected)

    def test_synthetic_training_row_excluded(self):
        frame = pd.read_csv(io.BytesIO(csv_bytes(10)))
        frame.loc[9, "climate"] = None
        sampled, info = sample_csv(io.BytesIO(frame.to_csv(index=False).encode()), 9, 0, training=True)
        self.assertEqual(info["excluded_rows"], 1)
        self.assertNotIn(9, sampled.source_row.tolist())
        with self.assertRaises(ValueError):
            sample_csv(io.BytesIO(frame.to_csv(index=False).encode()), 9, 0)

    def test_no_target_columns_or_bad_labels(self):
        self.assertEqual(feature_columns([*META, "forecast"]), ["forecast"])
        with self.assertRaises(ValueError):
            feature_columns([*META, "fact_leak"])
        for labels in ([0, 1.2], [0, np.nan]):
            with self.assertRaises(ValueError):
                integer_labels(labels)

    def test_preprocessing_fits_only_train_and_keeps_empty_columns(self):
        train = np.array([[1., np.nan], [3., np.nan]])
        p = fit_preprocessor(train)
        np.testing.assert_allclose(p.transform([[100., np.nan]]), [[98., 0.]])
        np.testing.assert_allclose(p.transform(train), [[-1., 0.], [1., 0.]])

    def test_overlap_rejected(self):
        frame = pd.read_csv(io.BytesIO(csv_bytes(5)))
        with self.assertRaises(ValueError):
            validate_observations({"train": frame, "eval_in": frame})

    def test_cleanup_is_disjoint_and_label_independent(self):
        base = pd.read_csv(io.BytesIO(csv_bytes(20)))
        base["source_row"] = np.arange(len(base))
        frames = {s: base.iloc[i * 4:i * 4 + 6].copy() for i, s in enumerate(SPLITS)}
        cleaned, audit = clean_observations(frames)
        validate_observations(cleaned)
        self.assertEqual(audit["removed_rows"]["dev_in"], 2)
        frames["dev_in"]["fact_cwsm_class"] = 99
        changed, _ = clean_observations(frames)
        for s in SPLITS:
            np.testing.assert_array_equal(cleaned[s].source_row, changed[s].source_row)


if __name__ == "__main__":
    unittest.main()
