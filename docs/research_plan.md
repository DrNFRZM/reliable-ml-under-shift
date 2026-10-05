# Experimental protocol

Protocol specified before running the experiments. This is a small empirical
study of existing methods, not a new uncertainty method or an official Shifts
leaderboard reproduction. See [audit.md](audit.md) for the initial state.

## Question

On a fixed subset of Shifts Weather precipitation classification, how do
in-domain temperature scaling and a three-member MLP ensemble affect NLL,
calibration, confidence-based selective risk, and conformal set coverage when
evaluation moves to the canonical time/climate-shifted domain?

NLL is the primary probability-quality metric. We report paired changes rather
than claim that unlike metrics measure comparable amounts of degradation.
Ensemble superiority, calibration transfer, and conformal coverage under shift
are empirical questions; adverse and null results are retained.

## Data and leakage controls

- Source: official canonical Shifts Weather archive in [references.md](references.md).
  No dataset redistribution under the code license.
- Target: `fact_cwsm_class`. Features exclude time, coordinates, climate, both
  targets, and any other `fact_` column. Never use observed temperature as input.
- Sample 20,000 training rows, 6,000 `dev_in` rows, and 5,000 rows from each of
  `eval_in` and `eval_out`, without replacement using seeded random priorities
  over **every** row in each complete CSV. Preserve original row indices.
- Remove synthetic training rows with absent climate or non-integer observation
  times; log their number. Fail on malformed labels rather than invent labels.
- Divide sampled `dev_in` rows into three disjoint random groups of 2,000:
  validation for stopping, temperature calibration, and conformal calibration.
  This partition is not label-stratified; calibration is not selected using labels.
  Do not tune on `dev_out` or either evaluation domain.
- Median imputation and standardization fit on training only. All missing
  features use a fixed zero fallback. Use the same transformed features for all
  trained models. Record missingness, label counts, dates, and climate counts.
- Save raw-archive and selected-data hashes, row indices, schema, preparation
  seed, and split membership. Fix the sample across optimization seeds.

Canonical splits contain related observations from stations and dates. They
are the benchmark protocol, not proof of independence or of generalization to
unseen stations. Audit exact observation-key overlap in the selected splits.

## Predictors and controlled comparisons

1. Empirical class prior (no features).
2. L2 multinomial logistic regression, `C=1`.
3. Histogram gradient boosting, 100 iterations, 15 leaf nodes, learning rate 0.1.
4. MLP with one 64-unit ReLU hidden layer, Adam, batch size 256, learning rate
   0.001, and L2 coefficient 0.0001.
5. The identical MLP with a positive scalar temperature fit by NLL on the
   temperature-calibration group.
6. Probability average of three MLPs trained on identical data with independent
   initialization/shuffling seeds. The single MLP is member zero of this ensemble.
7. Temperature scaling of ensemble probabilities, isolating calibration from averaging.

MLPs use at most 80 epochs, keep the lowest validation-NLL checkpoint, and stop
following 10 epochs without improvement of at least 0.0001. The two non-neural
fitted baselines are trained once and reused; repeating them does not create
independent evidence. No architecture or hyperparameter search is performed.

Repeat MLP training for five outer seeds (0–4). Each ensemble has three distinct
member seeds. Record validation losses, selected epochs, temperatures, convergence
warnings, fit time, and number of fits. Ensemble comparisons are paired but do
not control compute. No claim of Bayesian posterior inference is made.

## Evaluation

Evaluate frozen predictors separately on `eval_in` and `eval_out`:

- Accuracy, macro F1 over the fixed class vocabulary, and per-class recall.
- NLL (natural logarithms); multiclass Brier score as the sum over classes;
  top-label ECE with 15 fixed equal-width bins, plus 10/30-bin sensitivity checks.
- Risk–coverage curves ranked by maximum probability, AURC as mean risk over
  retained counts 1…n, and risk at 50%, 80%, and 100% coverage. Average over order
  within equal-confidence ties; include random and oracle rejection references.
  These are retrospective curves, not validated deployment thresholds.
- Split-conformal prediction with score `1 - p(y|x)`, alpha=0.1 and finite-sample
  rank `ceil((n+1)*(1-alpha))`. An out-of-range rank yields the full set. Report
  marginal coverage, mean set size, empty-set rate, and per-class coverage.
  Calibration uses the separate conformal group after temperature fitting.

Ordinary split-conformal coverage assumes exchangeability of calibration and
 test examples conditional on the fitted predictor. Weather dependence already
limits that interpretation in-domain; under time/climate shift we report
empirical coverage and make no distribution-free guarantee.

Report mean and **sample** SD over optimization seeds and paired method deltas.
These describe training variability conditional on one fixed subset; they are
not confidence intervals over weather stations, data draws, or future domains.
Do not count ensemble members as additional experiment replications.

## Deliverables and completion criteria

A minimal installable package; JSON configurations; deterministic preparation;
CLI experiment and re-evaluation; saved probabilities, labels, membership,
training diagnostics and provenance; metric/curve CSVs; generated plots and a
short report; unit/integration tests and a small CI smoke run.

Completion requires actual canonical-data results for the small configured
study, with recomputation from saved probabilities. If data access fails, say
explicitly that only the synthetic check ran. Never fill a results table with
expected, illustrative, or borrowed values.

No GPU is required. Limits include one dataset/subset, a modest training budget,
rare classes, dependent observations, fixed hyperparameters, and no novel method.
MC Dropout, large searches, additional datasets, and a full benchmark run are
outside this version.
