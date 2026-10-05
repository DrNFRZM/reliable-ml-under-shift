import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from shiftstudy.models import aligned_probabilities, member_seed, train_mlp


class ModelTests(unittest.TestCase):
    def test_training_deterministic_and_checkpoint_uses_validation_minimum(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(80, 4))
        y = (x[:, 0] > 0).astype(int)
        cfg = {"hidden_units": 8, "l2": .0001, "batch_size": 16,
               "learning_rate": .01, "max_epochs": 6, "min_delta": .0001, "patience": 3}
        with threadpool_limits(1):
            a, da = train_mlp(x[:60], y[:60], x[60:], y[60:], cfg, 12, 2)
            b, db = train_mlp(x[:60], y[:60], x[60:], y[60:], cfg, 12, 2)
        np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))
        self.assertEqual(da["selected_validation_nll"],
                         min(h["validation_nll"] for h in da["history"]))
        self.assertEqual(da["selected_epoch"], db["selected_epoch"])
        np.testing.assert_allclose(aligned_probabilities(a, x, 3)[:, 2], 0.)

    def test_member_seeds_distinct(self):
        seeds = [member_seed(s, m) for s in range(5) for m in range(3)]
        self.assertEqual(len(set(seeds)), 15)


if __name__ == "__main__":
    unittest.main()
