# Reproducing the compute-matched NODE study

This is the focused `PTCO-COMPUTE-0.1` comparison, specified in
[the design](../docs/superpowers/specs/2026-10-09-compute-matched-node.md).
The initial design has eleven training-seed clusters, ten independent paired
tasks per cluster, and six model/query configurations. It reuses the frozen
AffineFlow and RK4 checkpoints and trains the two-evaluation NODE through its
own solver. Earlier experiments are separate evidence and remain preserved.

All four looks are complete and archived. The final look joins 38 training-seed clusters and twenty paired tasks each: 4,560 primary episodes, plus 912 replay/refinement comparisons. AffineFlow at 40 ms has a supported cost advantage over every tested NODE configuration after the declared correction and influence checks. Its gain over the strongest coarse NODE is 0.71%; the 5 ms AffineFlow configuration does not beat every comparator. The fourth block was registered before its new outcomes and preserves all earlier measurements and search budgets. Software verification and final review are complete; see the [final review and execution record](compute_study_release_review_2026-10-10.md).

## Inputs and artifact location

Run commands from the repository root with native CPython 3.11 and the pinned
CPU environment. The environment can be restored from the existing wheelhouse
and `locks/requirements.lock` using the documented offline, hash-checked pip
command. Do not install different versions to bypass a failed reproduction.

The heavy new artifacts live at the owner-authorized local location:

```
/redacted-home/physical-time-consistency-artifacts/study/compute-matched-node
```

`config/compute_study_freeze.json` names that root and the exact producer,
training-data, original-checkpoint and environment inputs. Each new receipt
names a content-addressed job directory under `jobs/`, its manifest hash and
the complete-file marker. The manifest hashes every saved file. The original
data and checkpoints referenced by the freeze must also be available;
cloning source alone does not recreate the measured evidence. On a different
computer, restore this store and the original indexed evidence at the recorded
paths before verification. Changing a frozen path or source creates a different
job identity and requires an explicit compatibility record.

The small repository receipts are:

- `compute_study_fits/seed*.json`: eleven selected quadrature fits.
- `compute_study_references.json` and `compute_study_measurement.json`: fresh
  cold/warm probe endpoints, refinement envelopes and isolated inference timings.
- `config/compute_study_allocation.json` and
  `compute_study_allocation_check.json`: prospective timing-only budgets and
  the independent check with random-bank preparation outside the timer.
- `compute_study_primary.json`: complete control matrix; each job saves the
  physical state trajectory, observations, commands and every candidate bank.
- `compute_study_replay.json`, `compute_study_finer.json` and
  `compute_study_replicas/`: fresh-process replay, finer physics and first/last
  new-comparator retraining.
- `compute_study_look1.json` and `provenance/compute_study_look_001_result.json`:
  immutable first statistical look and its raw-linked decision.
- `compute_study_precision_002_*` receipts: the independently registered
  five-seed continuation beneath `precision-002/` in the same artifact store.
  Its source and request are frozen separately; all original receipts remain.

Use the named receipts, rather than whichever directory has the newest
modification time. `partial/` is resumable work, never completed evidence.

## Verify and regenerate the presentation

These commands perform hash validation and deterministic reaggregation; they
do not train models or simulate episodes:

```bash
.venv/bin/python -m poc.compute_study status
.venv/bin/python -m poc.compute_study_seed_report verify
.venv/bin/python -m poc.compute_study_seed_report demo
```

Verification rejects incomplete coverage, changed scientific inputs, altered
raw files, unclassified failures and inconsistent costs. It independently
recomputes every complete episode cost from the saved physical truth and
commands, checks replay/refinement and retraining receipts, and reproduces the
paired seed-level comparisons. Rereading unchanged evidence does not spend a
new statistical look. Changed evidence cannot be substituted into an already spent look.

Open `reports/compute_study_demo.html` locally in a browser. It contains all
saved tasks, has no server or network dependency, and lets the reader choose
the training seed and task. Its motion is physical simulation time; displayed
CPU planning time is separate. The companion machine result retains all eight
prespecified controller comparisons, cold/warm prediction results, inference
timings, nominal fitting quality and resource measurements.

## Explicitly rerun experiments

These are expensive commands. Confirm the current local allocation, store
availability and job locks before launching them. Do not start a duplicate of
an already active run. Completed compatible jobs are reused by content hash;
optimizer-update checkpoints resume interrupted fits.

```bash
.venv/bin/python -m poc.compute_study train
.venv/bin/python -m poc.compute_study measure
.venv/bin/python -m poc.compute_study controls
.venv/bin/python -m poc.compute_study reproduce
.venv/bin/python -m poc.compute_study_precision train
.venv/bin/python -m poc.compute_study_precision measure
.venv/bin/python -m poc.compute_study_precision controls
.venv/bin/python -m poc.compute_study_precision reproduce
.venv/bin/python -m poc.compute_study_episodes controls
.venv/bin/python -m poc.compute_study_episodes reproduce
.venv/bin/python -m poc.compute_study_seed_precision train
.venv/bin/python -m poc.compute_study_seed_precision measure
.venv/bin/python -m poc.compute_study_seed_precision controls
.venv/bin/python -m poc.compute_study_seed_precision reproduce
.venv/bin/python -m poc.compute_study_seed_report aggregate
.venv/bin/python -m poc.compute_study_seed_report demo
```

`reproduce` has distinct job identities for replay, finer-physics trajectories
and fresh first/last quadrature fits. Those replications run when their own
receipts are absent, then remain cached like other complete evidence. It does
not overwrite the primary runs or select the closer of multiple attempts.
Keep the frozen timing allocation: recalibrating after task outcomes would
answer a different comparison.

The first-look report implementation is preserved in Git commit `05c97f2`,
the second-look implementation in `b84b2ec`, and the third-look implementation in `bd836c0`;
its machine result and spending record remain immutable. The additive reporter
joins complete compatible old and new seed cells and task blocks for look 3. It does not replace
the first look or count training replications as new independent seeds. The
current source rejects substituting changed analysis into a previously spent
look. Verification of the final look requires both complete continuation receipts.
The task-only block lives beneath `episode-003/`; its request is
`config/compute_study_episode_003.json`. It adds tasks 10–19 for all sixteen
seeds, including 96 new replays and 96 finer-physics checks on task 10.
These additional tasks improve episode precision; they are not new training seeds.

For a complete from-scratch reproduction, preserve the original store, restore
the recorded input corpus and original selected checkpoints, and execute into
a separately registered compatible artifact store. Do not delete evidence or
mutate this study's freeze merely to force its cached jobs to rerun.

The full software checks are separate from the experiments:

```bash
.venv/bin/python -m pytest -q
```

Run them after timing measurements finish, so testing does not compete with
the single-threaded scientific jobs for CPU time. A green software suite is
not an empirical controller result.

The fourth producer is `poc.compute_study_seed_precision`; its frozen request and source bindings are in `config/compute_study_precision_004{,_freeze}.json`. Its mutable receipts live under the explicit `precision-004/receipts` Data-volume directory named in `compute_study_precision_004_store.json`. The recovered disk interruption is documented in `provenance/compute_study_disk_interruption_003.json`; a completed job was reused after progress saving failed, rather than repeated.

The completed fourth pipeline's additive reporter validates all
4,560 primary episodes, 912 replay/refinement comparisons, the previous
training replications and the final new-seed comparator replication:

```bash
.venv/bin/python -m poc.compute_study_seed_report aggregate
.venv/bin/python -m poc.compute_study_seed_report verify
.venv/bin/python -m poc.compute_study_seed_report demo
```

It preserves all three earlier immutable results. Refreshing an operational
progress timestamp does not create a new scientific look; changed raw receipts
or report semantics do. It rejects partial cohorts before statistical analysis.
Its `demo` reads saved trajectories and creates the same local presentation
with every task and all 38 training seeds. The current report and demo show the
fourth look. The first three immutable results remain historical evidence;
regenerating an older look's presentation would replace the current presentation,
so use a separate copy if inspecting it.
