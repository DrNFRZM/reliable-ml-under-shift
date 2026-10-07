# reliable-ml-under-shift

A small empirical study of how temperature scaling, a three-member MLP
ensemble, confidence-based rejection and split-conformal prediction sets behave
when the test data moves away from the training data. The task is 9-class
precipitation classification on a subset of the Shifts Weather dataset.

There is no new method here. The methods are standard; what the repository adds
is one controlled comparison, with the saved predictions needed to recompute
every number in it.

## Question

Calibration methods are fitted on held-out data that looks like the training
data. Weather forecasts are then used in other places and later months. So:

> On a fixed subset of Shifts Weather, how do in-domain temperature scaling and
> a three-member MLP ensemble change NLL, calibration error, selective risk and
> conformal set coverage on in-domain test data, and what is left of those
> changes on the time- and climate-shifted test data?

NLL is the main metric. The protocol is in
[docs/research_plan.md](docs/research_plan.md). It was fixed before the run
reported here, but it is not a preregistration: an earlier session, whose
outputs were lost, had already run some version of this experiment.

## Data

[Shifts Weather](https://arxiv.org/abs/2107.07455) (Malinin et al., 2021) has
123 forecast-model and measurement features per row and a precipitation class
(`fact_cwsm_class`, 9 codes). Its canonical partition is shifted in two ways at
once: the shifted evaluation set comes from later dates and from climate zones
(snow, polar) that do not occur in training (tropical, dry, mild temperate).

The study uses a uniform random sample of each canonical CSV: 20,000 training
rows, 6,000 `dev_in` rows and 5,000 rows from each evaluation set. Rows sharing
an exact (time, latitude, longitude) key with an earlier row were dropped
without looking at labels (47 rows in total). The `dev_in` sample is split at
random into three parts with separate jobs, so that early stopping, temperature
fitting and the conformal threshold never use the same rows. Neither `dev_out`
nor the evaluation sets are used for any fitting or tuning.

Time, coordinates, climate zone and the observed temperature are not model
inputs. Imputation medians and scaling constants come from the training rows
only.

## Predictors

| Predictor | Notes |
| --- | --- |
| Class prior | training class frequencies, no features |
| Logistic regression | multinomial, L2 with `C = 1` |
| Histogram gradient boosting | 100 iterations, 15 leaves, learning rate 0.1, no L2 |
| MLP | one hidden layer of 64 ReLU units, Adam, at most 80 epochs, early stopping on validation NLL |
| MLP + temperature | one scalar temperature fitted by NLL on the temperature split |
| Ensemble of 3 MLPs | mean of the class probabilities of three MLPs that differ only in their seed |
| Ensemble + temperature | temperature scaling applied to the averaged probabilities |

The single MLP is member 0 of the ensemble, so "ensemble vs MLP" is a paired
comparison. The MLP experiments are repeated for five outer seeds (0 to 4),
15 fits in total. The three baselines are fitted once each and have no seed.
There was no hyperparameter search.

Every predictor also gets split-conformal sets (score `1 - p(true class)`,
`alpha = 0.1`, so the target coverage is 0.9) and a risk-coverage curve that
rejects the least confident predictions first.

## Results

All numbers below come from one run, stored in
[results/reference](results/reference) with its provenance in
[results/README.md](results/README.md). The tables between the markers are
written by `scripts/readme_tables.py` from those files, and CI fails if they
drift.

For MLP-based rows, `a ± b` is the mean and sample standard deviation over the
five outer seeds. That spread describes training randomness on this one data
subset. It is not a confidence interval for other subsets, stations or periods.
Baselines are single fits and have no spread.

ECE uses 15 equal-width confidence bins. AURC is the mean error rate over all
coverage levels (lower is better). Set coverage is the fraction of rows whose
true class is in the conformal set.

<!-- tables:start -->
**Selected data** (after removing duplicated observations)

| Split | Used for | Rows | Dates (UTC) | Climate zones |
| --- | --- | ---: | --- | --- |
| `train` | fitting | 19987 | 2018-09-01 to 2019-04-07 | dry, mild temperate, tropical |
| `validation` | MLP early stopping | 1998 | 2018-09-01 to 2019-04-07 | dry, mild temperate, tropical |
| `temperature` | fitting the temperature | 1997 | 2018-09-01 to 2019-04-07 | dry, mild temperate, tropical |
| `conformal` | conformal threshold | 1997 | 2018-09-01 to 2019-04-07 | dry, mild temperate, tropical |
| `eval_in` | in-domain evaluation | 4982 | 2018-09-01 to 2019-04-07 | dry, mild temperate, tropical |
| `eval_out` | shifted evaluation | 4992 | 2019-05-14 to 2019-07-08 | polar, snow |

**In-domain evaluation (`eval_in`)**

| Predictor | Fits | Accuracy | NLL | Brier | ECE | AURC | Set coverage | Set size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class prior | 1 | 0.374 | 1.355 | 0.703 | 0.004 | 0.626 | 0.924 | 3.000 |
| Logistic regression | 1 | 0.569 | 1.056 | 0.561 | 0.013 | 0.293 | 0.894 | 2.330 |
| Histogram boosting | 1 | 0.554 | 2.211 | 0.624 | 0.099 | 0.418 | 0.890 | 2.642 |
| MLP | 5 | 0.571 ± 0.006 | 1.042 ± 0.006 | 0.555 ± 0.003 | 0.037 ± 0.013 | 0.282 ± 0.002 | 0.893 ± 0.005 | 2.304 ± 0.028 |
| MLP + temperature | 5 | 0.571 ± 0.006 | 1.036 ± 0.005 | 0.553 ± 0.002 | 0.018 ± 0.006 | 0.282 ± 0.002 | 0.894 ± 0.005 | 2.309 ± 0.029 |
| Ensemble of 3 MLPs | 5 | 0.581 ± 0.003 | 1.022 ± 0.002 | 0.545 ± 0.001 | 0.021 ± 0.005 | 0.273 ± 0.002 | 0.893 ± 0.004 | 2.269 ± 0.025 |
| Ensemble + temperature | 5 | 0.581 ± 0.003 | 1.021 ± 0.002 | 0.545 ± 0.001 | 0.016 ± 0.004 | 0.273 ± 0.002 | 0.893 ± 0.004 | 2.268 ± 0.030 |

**Shifted evaluation (`eval_out`)**

| Predictor | Fits | Accuracy | NLL | Brier | ECE | AURC | Set coverage | Set size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Class prior | 1 | 0.298 | 1.418 | 0.732 | 0.080 | 0.702 | 0.903 | 3.000 |
| Logistic regression | 1 | 0.437 | 1.287 | 0.684 | 0.110 | 0.480 | 0.842 | 2.470 |
| Histogram boosting | 1 | 0.436 | 2.449 | 0.738 | 0.158 | 0.548 | 0.883 | 2.863 |
| MLP | 5 | 0.448 ± 0.005 | 1.295 ± 0.015 | 0.686 ± 0.009 | 0.119 ± 0.012 | 0.463 ± 0.010 | 0.848 ± 0.005 | 2.475 ± 0.040 |
| MLP + temperature | 5 | 0.448 ± 0.005 | 1.271 ± 0.008 | 0.677 ± 0.006 | 0.093 ± 0.003 | 0.463 ± 0.010 | 0.848 ± 0.006 | 2.471 ± 0.040 |
| Ensemble of 3 MLPs | 5 | 0.454 ± 0.002 | 1.256 ± 0.010 | 0.672 ± 0.005 | 0.100 ± 0.005 | 0.451 ± 0.006 | 0.849 ± 0.007 | 2.436 ± 0.031 |
| Ensemble + temperature | 5 | 0.454 ± 0.002 | 1.249 ± 0.006 | 0.669 ± 0.003 | 0.090 ± 0.002 | 0.451 ± 0.006 | 0.848 ± 0.007 | 2.432 ± 0.035 |

**Paired differences over the five outer seeds** (first minus second; negative NLL/ECE is better)

| Comparison | Domain | NLL | ECE | Accuracy | Seeds with lower NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP + temperature vs MLP | `eval_in` | -0.0062 ± 0.0036 | -0.0196 ± 0.0129 | +0.0000 ± 0.0000 | 5 of 5 |
| MLP + temperature vs MLP | `eval_out` | -0.0242 ± 0.0101 | -0.0262 ± 0.0095 | +0.0000 ± 0.0000 | 5 of 5 |
| Ensemble of 3 MLPs vs MLP | `eval_in` | -0.0203 ± 0.0041 | -0.0163 ± 0.0123 | +0.0094 ± 0.0063 | 5 of 5 |
| Ensemble of 3 MLPs vs MLP | `eval_out` | -0.0392 ± 0.0063 | -0.0196 ± 0.0078 | +0.0063 ± 0.0025 | 5 of 5 |
| Ensemble + temperature vs Ensemble of 3 MLPs | `eval_in` | -0.0013 ± 0.0008 | -0.0049 ± 0.0036 | +0.0000 ± 0.0000 | 5 of 5 |
| Ensemble + temperature vs Ensemble of 3 MLPs | `eval_out` | -0.0075 ± 0.0039 | -0.0097 ± 0.0045 | +0.0000 ± 0.0000 | 5 of 5 |

**Post-hoc histogram-boosting diagnostic** (exploratory, one fit each)

| Variant | Domain | Accuracy | NLL | ECE | Set coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| L2 = 0 (primary configuration) | `eval_in` | 0.554 | 2.211 | 0.099 | 0.890 |
| L2 = 0 (primary configuration) | `eval_out` | 0.436 | 2.449 | 0.158 | 0.883 |
| L2 = 1 (added afterwards) | `eval_in` | 0.580 | 1.023 | 0.031 | 0.884 |
| L2 = 1 (added afterwards) | `eval_out` | 0.463 | 1.232 | 0.085 | 0.854 |
<!-- tables:end -->

![Reliability diagrams for outer seed 0](results/reference/calibration.png)

![Risk-coverage curves averaged over outer seeds](results/reference/risk_coverage.png)

### What the run shows

- **Everything is worse under shift.** For every trained predictor accuracy
  drops by 12 to 13 points and NLL rises by 0.23 to 0.25. The ordering of the
  MLP-based predictors by NLL is the same in both domains.
- **Averaging three MLPs lowered NLL in all five seeds, in both domains.** The
  reduction is larger on the shifted set. It costs three times the training,
  and the comparison does not control for that.
- **Temperature scaling fitted in-domain still helped under shift, but did not
  repair calibration.** It lowered NLL and ECE for the single MLP in both
  domains. The fitted temperatures are all slightly above 1 (1.06 to 1.18 for
  the MLP), so the MLPs were mildly over-confident. After scaling, shifted ECE
  is still several times the in-domain value.
- **Conformal coverage held in-domain and fell under shift.** In-domain
  coverage is 0.89 to 0.90 for the trained predictors, close to the 0.9 target
  given a calibration split of about 2,000 rows. On the shifted set it is 0.84
  to 0.85 for logistic regression and the MLP-based predictors, with larger
  sets. This is expected: the guarantee needs calibration and
  test rows to be exchangeable, and they are not here. Coverage is also very
  uneven across classes; rare classes are covered far less often
  (`per_class.csv`).
- **Rejecting low-confidence predictions helps less under shift.** At 50%
  coverage the ensemble's error rate is about 0.29 in-domain and 0.46 shifted
  (`metrics.csv`, `risk_at_50`).
- **The boosting baseline as configured is badly calibrated.** Its NLL is worse
  than the class prior's although its accuracy is comparable to the other
  models. About half of its NLL comes from a few percent of rows where it gives
  the true class a probability below 1e-6; the NLL value therefore depends on
  the 1e-15 clipping floor. This result is kept as it is.

After seeing that last result, one more boosting model with `L2 = 1` was fitted
on the same data to check whether the missing regularisation was the cause. It
is the last table above. With `L2 = 1` the NLL is in the same range as the
MLPs, and on the shifted set slightly below them. This comparison was not planned in advance, used a single alternative
value, and was prompted by test-set results, so it explains the baseline's
behaviour but is not evidence for ranking boosting against the MLPs.

## Limitations

- One sampled subset of about 20,000 training rows out of 3.1 million, one
  architecture, no hyperparameter search. Nothing here is comparable to the
  Shifts leaderboard, which also uses a different rejection metric.
- Five optimisation seeds. Differences are described, not tested for
  significance, and the data subset is the same in every seed.
- Rows from the same station and nearby times are related. Only exact
  duplicates of (time, latitude, longitude) were removed; there is no
  station-held-out split.
- Several classes are very rare (3 training rows for one of them, none in
  `eval_in`). Macro F1 and per-class numbers for those classes carry almost no
  information.
- The shifted set changes time and climate together, so the two effects cannot
  be separated.
- The ensemble uses three times the compute of the single MLP.
- Temperature scaling works on `log p` of the averaged ensemble probabilities,
  not on per-member logits.
- The reference run happened once, on a GitHub-hosted runner. Re-evaluating
  the saved predictions is byte-reproducible; retraining on other hardware is
  not expected to be bit-identical.

## Reproducing

Python 3.11 or 3.12.

```bash
python -m pip install -r requirements-lock.txt -e .
python -m unittest discover -s tests

# Recompute every table from the saved predictions (seconds, no training):
python scripts/verify_reference.py results/reference   # independent of the package's metric code
shift-study evaluate --run results/reference           # the package's own evaluator

# Synthetic end-to-end run that only tests the plumbing:
shift-study smoke --config configs/smoke.json --output runs/smoke
```

To retrain from the raw data, download the
[canonical archive](https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar)
(about 7.5 GB, CC BY-NC-SA 4.0), then:

```bash
shift-study prepare --source canonical-partitioned-dataset.tar --config configs/weather.json --output data/weather-prepared
shift-study run --data data/weather-prepared --config configs/weather.json --output runs/weather
python scripts/tree_diagnostic.py run --data data/weather-prepared --reference runs/weather --output runs/weather/tree-diagnostic
```

Training took under a minute on a 4-core CI runner. The "Canonical Weather
reference" workflow does the same steps and can be started by hand.

## Repository layout

```
configs/            weather.json (the study) and smoke.json (synthetic test)
src/shiftstudy/     data.py         sampling, duplicate removal, preprocessing
                    models.py       baselines and the MLP training loop
                    calibration.py  temperature scaling, conformal sets
                    metrics.py      NLL, Brier, ECE, risk-coverage
                    experiment.py   training run and re-evaluation from saved predictions
                    report.py       report.md and figures
                    cli.py, config.py
scripts/            verify_reference.py  independent recomputation
                    tree_diagnostic.py   post-hoc boosting comparison
                    readme_tables.py     README tables from results/reference
tests/              unit and end-to-end tests on synthetic data
results/reference/  saved predictions, provenance and tables of the reference run
docs/               protocol, audits, references
```

## Development history

The repository was built with AI coding assistants, as the commit trailers
show, and one working session was interrupted before its results were saved.
[docs/recovery_audit.md](docs/recovery_audit.md) records what was lost, how the
reference run was produced afterwards and how it was checked. Results described
by the interrupted session were never recovered and are not used anywhere.

## Data, references and license

The code is under the MIT license (`LICENSE`). The Shifts Weather data is
licensed separately under CC BY-NC-SA 4.0 and is not redistributed here; the
repository stores only row indices, observation keys, labels and model outputs
for the selected rows. Citations are in [docs/references.md](docs/references.md):

- Malinin et al. (2021), *Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks*.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- Lakshminarayanan et al. (2017), *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*.
- Angelopoulos and Bates (2021), *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*.
