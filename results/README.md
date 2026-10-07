# Reference run

`reference/` holds the output of the one canonical Shifts Weather run this
repository reports. The files are byte-for-byte what the run produced; nothing
was edited by hand.

| | |
| --- | --- |
| Workflow run | [37442161270](https://github.com/DrNFRZM/reliable-ml-under-shift/actions/runs/37442161270) ("Canonical Weather reference"), 6 October 2026 |
| Code commit | `a6228dd203d60a168f8e11275a69f50811a5c26a`, clean working tree |
| Actions artifact | `canonical-reference-a6228dd203d60a168f8e11275a69f50811a5c26a` (id 11402176154) |
| Artifact zip SHA-256 | `d08461569e27b351d97a37dd1d96625046b2b8950e7ee750837a19989f5ceef5` |
| Upstream archive SHA-256 | `a0ebb45789329846415ca31b3c110956f975928bed7f5d0726bfb02044e0edf1` |
| Prepared subset SHA-256 | `de1056b3738b9fe0e698ac9f7baffe0b4857056f1edd82fea43c189a72bc6527` |
| Environment | Python 3.12.14 on a GitHub-hosted Ubuntu runner, packages as in `requirements-lock.txt` |

The Actions artifact itself expires after 14 days, so this directory is the
lasting copy.

## What each file is for

Inputs to re-evaluation (hashed in `manifest.json`):

- `manifest.json`: code commit, package hashes, runner, versions, file hashes.
- `config.json`: the configuration used (same content as `configs/weather.json`).
- `data_metadata.json`: archive and per-CSV hashes, feature list, class counts,
  number of rows removed by the duplicate-observation cleanup.
- `selection.npz`: source row index, (time, latitude, longitude) key and label
  for every selected row in all six splits. No features are stored.
- `preprocessing.npz`: imputation medians and scaling constants fitted on train.
- `training.json`: per-epoch losses, selected epochs, fitted temperatures,
  fit times and warnings.
- `seed_0.npz` ... `seed_4.npz`: float32 class probabilities of every predictor
  and of each ensemble member on the temperature, conformal and two evaluation
  splits, with labels and row indices.
- `requirements-lock.txt`: copy of the lock file used by the run.

Outputs recomputed from the files above by `shift-study evaluate`:

- `metrics.csv`, `summary.csv`, `paired_deltas.csv`, `paired_summary.csv`,
  `per_class.csv`, `calibration_bins.csv`, `risk_coverage.csv`,
  `conformal_thresholds.csv`, `report.md`, `calibration.png`, `risk_coverage.png`.

`tree-diagnostic/` is the post-hoc histogram-boosting comparison (L2 = 0 against
L2 = 1). It was added after the primary results had been seen and is not part
of the pre-specified comparison. Its manifest records `git_dirty: true` because
`results/reference/` already existed as an untracked directory in the runner
when the diagnostic started; the script hash in that manifest matches the
committed `scripts/tree_diagnostic.py`.

The raw dataset and the prepared feature arrays are not in Git (CC BY-NC-SA 4.0
data, about 7.5 GB).
