# Initial repository audit

Audited on 2026-10-05, before implementation, at
`79ec49fe1cf68068b89e03ec853f1f5ca561fd5c`.

## What was present

All four tracked files were inspected: `README.md`, `docs/research_plan.md`,
`LICENSE`, and `.gitignore`. The repository had two commits, one branch (`main`),
and no issues or pull requests. There was no executable code, environment,
configuration, data preparation, test, CI workflow, or experimental output.
The MIT license correctly named Farzam Nikbakhsh Jorshari and is preserved.

The intended question was whether uncertainty estimation, calibration, and
selective prediction remain useful under real distribution shift. The proposed
task was multiclass precipitation classification on Shifts Weather, using its
canonical in-domain and time/climate-shifted partitions.

## Claims and methodological gaps

The README said it was benchmarking methods under real-world distribution shift.
That described an intention, not an implemented or measured capability. It did
not claim numerical superiority, so there were no fabricated numbers to remove.
The four hypotheses in the plan were untested. Comparing the magnitude of a
calibration change with an accuracy change requires a defined scale and
estimand; the original H1 supplied neither.

Missing decisions included a subset algorithm, separation of model selection
from temperature and conformal calibration, preprocessing rules, class handling,
ensemble budget, seeds, stopping criteria, metric definitions, and aggregation.
Conformal prediction is a wrapper producing sets, not another trained model.
Weather observations have temporal and spatial dependence: IID confidence
intervals and unconditional promises of conformal coverage would be misleading.
An ensemble also costs multiple model fits, so equal architecture does not mean
equal compute. No citation or dataset license was supplied.

## Completion decision

Preserve the task and canonical partitions. Implement one CPU-sized MLP, a
three-member independently initialized ensemble, temperature scaling of both,
and prior/logistic/tree baselines. Apply split-conformal sets to each probability
predictor. MC Dropout is excluded from this version to keep the study small;
there is no claim that this answers every original hypothesis.

The revised protocol is in [research_plan.md](research_plan.md). Results must
come from actual runs and be traceable to predictions, configuration, source
hashes, and software versions. A synthetic smoke test verifies plumbing only.

## Source verification

Shifts-Project/shifts was inspected at
`81b80942ace4f227c4f0c7424906e176cd34c8a0`: its README, weather tutorial, partition
documentation, and partitioner. These confirm the target `fact_cwsm_class`,
exclusion of the first six metadata/target columns, canonical split names,
and the separate CC BY-NC-SA 4.0 dataset license. Its partitioner can append
synthetic average rows to training; preparation must identify and exclude
non-observational training rows rather than silently treat them as measurements.
The archive endpoint returned HTTP 200 and a size of 7,472,220,160 bytes.
