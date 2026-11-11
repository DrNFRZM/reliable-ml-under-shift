# Reproduction check

## 11 November 2026 maintenance verification

The AI-assisted maintenance pass used Python 3.12.16 on Windows with the
recorded direct/transitive numerical dependency lock. The shared audit
environment passed its dependency consistency check.

- All 20 unittest tests passed.
- The independent reference checker passed every hash, split, probability,
  temperature, metric, summary, and diagnostic check. The largest reported
  metric-table difference was 4.4e-16.
- `python scripts/readme_tables.py --check` confirmed both README result blocks.
- A fresh `shift-study smoke --config configs/smoke.json --output <new-directory>`
  completed preparation, fitting, evaluation, and reports on synthetic data.
  The independent reference checker also passed on that fresh smoke output.
- The referenced canonical Actions run 37442161270 was confirmed successful
  at commit `a6228dd203d60a168f8e11275a69f50811a5c26a` through GitHub's API.

No new run on the 7.5 GB Weather archive was performed. The earlier reference
outputs remain unchanged. Documentation now explicitly distinguishes excluded
raw feature arrays from retained source-derived labels and observation keys,
which remain subject to the upstream data terms rather than the code license.

## Earlier verification record

A fresh virtual environment was installed with `requirements-lock.txt` and
`pip install -e .` on Python 3.12.14, Linux, CPU only. `pip check` found no
dependency conflicts. The numerical packages were NumPy 2.3.5, SciPy 1.17.0,
pandas 2.2.3, scikit-learn 1.8.0, and Matplotlib 3.10.8.

| Check | Outcome |
| --- | --- |
| `python -m unittest discover -s tests` | 20 tests passed |
| `python scripts/verify_reference.py results/reference` | All checks passed; largest metric difference was 4.4e-16 |
| `shift-study evaluate --run results/reference` | Recomputed the tracked numerical tables without changing them |
| `python scripts/readme_tables.py --check` | Both README table blocks match the reference results |
| `shift-study smoke --config configs/smoke.json --output runs/verification-smoke` | Completed preparation, fitting, evaluation, and reporting on synthetic data |

The independent checker verified file hashes, split identities and overlap,
ensemble averaging, fitted temperatures, metric summaries, and the post-hoc
tree diagnostic. The smoke run used its own output directory.

This checks the saved Weather predictions and the executable pipeline. It is
not another training run on the 7.5 GB canonical archive, and it does not test
performance on a second data subset. The reference experiment remains the
single run described in [results/README.md](../results/README.md).
