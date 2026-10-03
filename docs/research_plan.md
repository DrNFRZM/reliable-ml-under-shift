# Research Plan

## Project
Reliable ML Under Distribution Shift

## Main Research Question

How well do practical uncertainty estimation and calibration methods remain reliable when a machine-learning model encounters real-world distribution shift?

## Dataset

Shifts Weather Prediction Dataset.

Primary task:
- Multiclass precipitation classification

The project will use the official canonical partitions so that in-distribution and distribution-shift performance can be evaluated separately.

Because the complete dataset contains approximately 10 million examples, experiments will use a reproducible subset suitable for a single consumer GPU.

## Experimental Setup

A single neural-network architecture will be used as the main predictive model so that changes in uncertainty quality can be attributed mainly to the uncertainty method rather than to different architectures.

Methods to compare:

1. Deterministic MLP
2. Temperature-scaled MLP
3. MC Dropout
4. Deep Ensemble
5. Conformal Prediction

A simple non-neural tabular baseline may also be included as a sanity check.

## Evaluation

### Predictive performance
- Accuracy
- Macro F1

### Probabilistic quality
- Negative Log-Likelihood
- Brier Score
- Expected Calibration Error

### Selective prediction
- Risk-Coverage curves
- Accuracy as increasingly uncertain predictions are rejected

### Conformal prediction
- Empirical coverage
- Average prediction-set size

## Central Comparison

Every method will be evaluated separately on:

1. In-distribution data
2. Distribution-shifted data

The main analysis will investigate whether methods that appear well calibrated in-distribution remain reliable after distribution shift.

## Hypotheses

H1:
Distribution shift will degrade calibration more strongly than ordinary predictive accuracy alone suggests.

H2:
Deep ensembles will provide more robust uncertainty estimates under shift than a single deterministic model.

H3:
Post-hoc temperature scaling will improve in-distribution calibration but may not transfer reliably to shifted data.

H4:
Conformal methods will provide useful coverage information, but distribution shift may affect coverage guarantees when exchangeability assumptions no longer hold.

## Scope

This is a focused research project, not a large benchmark.

We will NOT:

- test dozens of datasets;
- implement many architectures;
- build a new uncertainty method;
- perform large-scale hyperparameter searches;
- use unnecessarily large models.

The emphasis is on rigorous experimental design, reproducibility, and analysis.

## Compute Constraint

Target hardware:
NVIDIA RTX 2060

Experiments must therefore remain practical on approximately 6 GB of GPU memory.

## Expected Outputs

- Reproducible research repository
- Config-driven experiments
- Saved numerical results
- Calibration plots
- Risk-coverage plots
- Comparison tables
- Statistical analysis across random seeds
- Short research-style technical report
