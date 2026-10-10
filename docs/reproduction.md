# Inspecting and reproducing the study

| Activity | Available from source checkout? | What it establishes |
|---|---|---|
| Read reports and play the saved demo | Yes, without Python | Published presentation and saved trajectories |
| Run model/controller unit tests | Yes, with pinned dependencies | Software behavior against synthetic and numerical fixtures |
| Recompute scientific results or rerun original experiments | Requires the separately indexed original artifact store | Raw provenance, complete workloads and experiment reproduction |

A public source snapshot is not the full raw experiment deposit. Candidate
banks, datasets, selected checkpoints, original frozen path bindings and exact
wheels remain external. No downloadable DOI or public raw-store URL is claimed.
The owner retains those artifacts. This limitation must be resolved before
describing a public clone as a self-contained replication package.

## Environment and focused checks

Use native CPython 3.11 and CPU execution. Exact top-level pins are in
[requirements.in](../requirements.in). On macOS ARM64:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.in
python -m pip check
python -m pytest -q tests/test_affine_flow.py tests/test_quadrature_node.py tests/test_stateless_mpc.py
```

For Linux, install `torch==2.8.0` from the official CPU index before installing
the other pins. Cross-platform bitwise reproduction has not been established.

## Prediction interface

```python
from poc.affine_flow import AffineFlow
from poc.models.common import Scales

model = AffineFlow("lagged", scales)  # training-only scales; untrained weights
result = model.predict(state, command, h_s)
next_state = result.next_state
```

Torch inputs: state `[batch,6]` ordered `q1,q2,v1,v2,w1,w2`, command `[batch,2]`
in N·m, and time in seconds as a scalar or `[batch,1]`. Direct torque uses state
dimension four and identifier `"direct"`. Angles are unwrapped radians.
`Scales` contains positive two-component position, velocity and velocity-input
tensors. Measured predictions require the indexed trained state dictionary
and matching scales. [Implementation](../poc/affine_flow.py).

## Original artifact verification

The [historical runbook](../reports/compute_study_reproduce.md) describes job
identities, checkpoints, independent seed blocks, refinements, cost arithmetic
and statistical looks. With the complete original store and bindings restored:

```bash
python -m poc.compute_study status
python -m poc.compute_study_seed_report verify
python -m poc.compute_study_seed_report demo
```

These commands validate/reaggregate existing evidence without retraining.
Expensive producer commands are listed separately in that runbook. New runs
need the operator's own allocation and compatible inputs; the original owner's
authorization does not authorize execution on someone else's machine.

Public packaging omits executor logs and may redact local paths in archival
metadata. These are **derived public copies**, with source and exported hashes
in `PUBLICATION_MANIFEST.json`, rather than byte-identical frozen records.
Raw-evidence verification must use the unmodified originals at their recorded
code revisions; silent changes to frozen paths are rejected by the scientific
validators. The public copy of `poc/compute_study.py` has one disclosed portability
adaptation: its default external directory uses `Path.home()` instead of the
owner's literal home path. The exporter guards the original and derived full-file
hashes. All model and controller code is byte-identical, and no selected tensors,
measurements or original decisions are changed. This public source copy does not
substitute for the archived code and bindings in a frozen-evidence audit.

## Recorded original verification

The final study covers 4,560 control episodes, 912 fresh-process replay and
finer-physics comparisons, independent cost arithmetic and exact selected
tensors for first/last quadrature training replications. Its original final
software suite had 1,053 passing tests. These are recorded execution facts,
not claims that a new clone has already rerun the experiments.
