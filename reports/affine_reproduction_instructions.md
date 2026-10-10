# Reproduce the targeted affine-flow comparison

PTCO-AFF-0.2. The targeted study is complete: all measurements, controller/finer-physics checks, internal reproduction, full tests and fresh final review are finished. The default commands below verify and present archived results; they do not train or simulate.

The scientific sources and input identities are pinned in `config/affine_fresh_freeze.json`, `config/affine_confirmation_freeze.json` and the additive `config/affine_precision_freeze_001.json`. Selected weights and raw arrays are local artifacts referenced by their complete manifests and hashes. A source-only Git checkout does not contain the excluded raw evidence or wheel binaries; transfer those indexed artifacts too when reproducing on another workstation.

Use the authorized native CPython3.11 CPU environment. The existing resolved wheels/lock are in `wheelhouse/` and `locks/requirements.lock`. To restore an environment offline, create a project virtual environment and install with:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse --require-hashes -r locks/requirements.lock
```

These commands install only the archived packages and do not upload data. The recorded same-platform reproducibility claim does not imply bitwise cross-platform agreement.

The completed prediction and controller evidence is now analyzed at look3. Look1 and look2 remain immutable earlier records. Never apply look2 to the changed complete controller evidence. Reaggregating identical inputs at look3 reuses its spending receipt rather than creating a new statistical look.

These default commands verify saved-weight prediction replay, already measured control replay, independent training replication and the existing control result. They do not launch training or controller episodes:

```bash
.venv/bin/python -m poc.affine_precision reproduce
.venv/bin/python -m poc.affine_precision aggregate --look 3
PYTHONPATH=. .venv/bin/python reports/provenance/independent_affine_audit.py
.venv/bin/python -m poc.affine_precision demo
```

The demo reads actual complete saved trajectories and physical costs. Its motion playback follows physical simulation time; measured controller CPU latency is shown separately.

To resume missing experiments after an interruption, first check `reports/project_completion_status.json`, the execution log and process/job locks. Then use only the incomplete stage:

```bash
.venv/bin/python -m poc.affine_precision controls
.venv/bin/python -m poc.affine_precision replay
.venv/bin/python -m poc.affine_precision finer
.venv/bin/python -m poc.affine_fresh retrain --seed 2026100800
.venv/bin/python -m poc.affine_fresh retrain --seed 2026100810
```

These are expensive, explicitly opt-in commands. Each reuses compatible completed content-addressed jobs and resumes missing work; it does not delete or overwrite the original cohort. A still-running matching job must not be duplicated. Independent training replicas use their own fit identity and preserve the original selected weights. Changing the model, data, solver, selection policy or scientific metric is a new experiment rather than a resumption.

The raw-linked result is `reports/affine_refinement_findings.md`; the immutable first and second looks remain under `reports/provenance/` and `evidence/affine-fresh-looks/`. Current independent inputs, measurement pointers, precision request and training resources are linked there. The complete raw controller/reproduction pointers are in the final report and `affine_practical_metrics.json`. No command here requires the original proposal or this conversation.

The user authorized the persistent main-volume training-workspace location `/redacted-home/physical-time-consistency-artifacts/study/affine-fresh`. `study/affine-fresh` is a local symlink to that directory; sealed evidence remains inside the repository. Its complete copy/hash verification is in `storage_fresh_workspace_relocation.json` and the accompanying file index. A restricted agent must run training with authorized write access to that destination. Ordinary local-shell training uses normal file permissions. Verification/reaggregation does not require the intermediate training workspace. When transferring the workspace to another host, retain the recorded logical contents and recreate the link to your authorized persistent artifact location; do not copy a dangling machine-specific link.

The installed `.venv` is also linked to the authorized main-volume location `/redacted-home/physical-time-consistency-artifacts/venv` after the Git volume filled during replica export. All31417 readable file hashes, Python prefix, pinned versions and Torch configuration were verified; the source wheels and lock remain inside this repository. See `storage_venv_relocation.json`. A new workstation should restore an ordinary local environment from the lock instead of copying this machine-specific alias. The disk-failed export attempt is retained; its complete training state was resumed without extra optimizer updates.

The aggregate CLI field `control: True` means that complete control evidence is present; it does not mean control superiority passed. Consult the three explicit `control_comparisons` in `affine_precision_001_result.json`. All three are unresolved for a positive gain and are reported with their bounded uncertainty.
