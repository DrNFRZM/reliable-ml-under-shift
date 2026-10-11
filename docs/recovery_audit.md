# Recovery audit

The first part was written on 6 October 2026, before the reference run, and is
kept as written. The last section records what happened afterwards.

## Snapshot inspected before recovery changes (6 October 2026)

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

## Outcome (7 October 2026)

The recovery job ran once, as [Actions run 37442161270](https://github.com/DrNFRZM/reliable-ml-under-shift/actions/runs/37442161270)
at commit `a6228dd`, and finished without errors. Its artifact was downloaded,
checked and committed under `results/reference/`; see
[../results/README.md](../results/README.md) for hashes and file roles.

Status of the four gaps listed above:

1. Done. The run stores predictions, labels, row indices and observation keys
   for all six splits, the configuration, training diagnostics and data/source
   hashes. The manifest names commit `a6228dd` with a clean working tree, and
   the package hashes in it equal the files at that commit.
2. Done. `scripts/verify_reference.py` recomputes every table from the saved
   probabilities without importing the package; the largest difference from
   the stored values is below 1e-15. It also checks that the single MLP is
   ensemble member 0, that the ensemble is the member mean, and that the
   stored temperatures are reproduced by an independent fit. The baselines
   were being counted once per outer seed (count 5, SD 0); they are now
   evaluated and reported once.
3. Done. CI runs the verification script and the package evaluator on the
   committed files and fails if any tracked file changes or an untracked file
   appears.
4. Done. The README tables are generated from the committed files. The
   boosting diagnostic is labelled post-hoc everywhere it appears.

Things that remain true and are not fixed:

- The results an earlier session described were never recovered. Nothing in
  this repository depends on them, and whether the configuration was chosen
  before or after that session saw results cannot be established.
- The reference run was executed once. The saved predictions are re-evaluated
  on every CI run, but the training itself has not been repeated on a second
  machine.
- The diagnostic's manifest says `git_dirty: true`. The cause was the untracked
  `results/reference/` directory written by the previous workflow step, not a
  source change (the recorded script hash matches the committed script). The
  manifest is left as produced.
