# Recovery audit, 6 October 2026

Snapshot inspected before recovery changes:

- `main`: `79ec49fe1cf68068b89e03ec853f1f5ca561fd5c` (early scaffold).
- `research-completion`: `164880be9cb6d115bae7c2e4b8fc818e978b1de9`, seven commits ahead, no PR.
- Remote tree: package, configurations, protocol, exploratory tree script and 20 tests; no `results/` files.
- [Actions run 37431311161](https://github.com/DrNFRZM/reliable-ml-under-shift/actions/runs/37431311161): both Python versions passed tests and synthetic smoke; reference evaluation failed on missing `results/reference/manifest.json`.
- The 20 tests also passed in the fresh Python 3.12 workspace after installing the package.

The README's claim that no experiment had run conflicted with the protocol's
description of primary results and a post-hoc diagnostic. Neither historical
claim can be verified without saved evidence. Recovery starts from the remote
code and unchanged five-seed configuration, and will generate a new canonical
run. Existing history and AI-development attribution are retained.

## Findings and bounded completion plan

The implementation matches a modest study of calibration and ensemble averaging
under canonical Weather time/climate shift. No new method or benchmark-level
claim is justified. It already excludes observed targets/metadata from inputs,
fits preprocessing on training only, separates stopping/temperature/conformal
groups, checks exact observation overlap, and pairs the single MLP with member
zero. Tests cover sampling, leakage controls, metrics, ties and corruption.

Remaining gaps:

1. Run the actual canonical archive and preserve predictions, labels, all selected
   row IDs/split keys, configuration, training diagnostics and source/data hashes.
   Current run output saves evaluation/calibration IDs but omits training and
   stopping IDs; add a compact selection artifact before the recovered fit.
2. Independently recompute every reported metric, ensemble average, calibration
   transform and paired contrast. Summaries currently repeat deterministic
   baselines across seeds; expose one fitted baseline rather than five replicates.
3. Commit compact reference evidence; CI must validate those committed files and
   both Python versions. The existing evaluator's missing file is a real failure,
   not evidence that training or the unit tests failed.
4. Replace the placeholder README only after verification, retain adverse/null
   results and label the inherited fixed-L2 diagnostic exploratory.

Interpretation limits already relevant before fitting: one sampled dataset,
fixed architecture/budget, related station/time observations, rare or absent
classes, threefold ensemble compute and only five optimization seeds. Exact-key
cleanup is label-independent but changes the candidate-row sampling distribution.
Optimization SD is not sampling uncertainty. Conformal exchangeability is not
established even in-domain and is especially inappropriate to assume under shift.
Retrospective risk-coverage curves do not specify a deployable rejection policy.

The canonical host timed out behind this workspace's network proxy. A bounded
GitHub Actions recovery job can download the official archive, execute the frozen
protocol and preserve compact artifacts for local independent verification. Raw
data will remain outside Git. An unsuccessful download/run must not be presented
as a completed canonical experiment.
