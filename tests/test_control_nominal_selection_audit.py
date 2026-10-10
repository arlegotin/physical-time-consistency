"""The final nominal audit must cover the exact frozen cohort without reselection."""
from copy import deepcopy

import pytest

from reports.audit_control_nominal_selections import audit_jobs, check_fit_identity


def frozen():
    return dict(state="complete", expected_episodes=1200, selected=[
        dict(actuator=a, model_id=m, seed=s, manifest=f"study/source/evidence/M30/{a}-{m}-{s}/manifest.json",
             manifest_sha256="a" * 64, checkpoint_sha256="b" * 64, canonical_tensor_sha256="c" * 64)
        for a in ("direct", "lagged") for m in ("mlp_dt", "node_generic", "power_correct")
        for s in range(5)])


def test_exact_thirty_checkpoint_cohort_uses_frozen_source_contexts():
    jobs = audit_jobs(frozen())
    assert len(jobs) == 30
    assert {j["source_context"] for j in jobs} == {"study/source"}
    assert len({j["cell_id"] for j in jobs}) == 30


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "development_seed", "unknown_model"])
def test_incomplete_or_substituted_cohort_is_rejected(mutation):
    evidence = frozen()
    if mutation == "missing":
        evidence["selected"].pop()
    elif mutation == "duplicate":
        evidence["selected"][-1] = deepcopy(evidence["selected"][0])
    elif mutation == "development_seed":
        evidence["selected"][0]["seed"] = 9000
    else:
        evidence["selected"][0]["model_id"] = "gru_dt"
    with pytest.raises(ValueError):
        audit_jobs(evidence)


def test_fit_identity_must_be_confirmatory_augmented_shared_loss():
    job = audit_jobs(frozen())[0]
    record = dict(milestone_id="M30", phase="confirmatory", job_inputs=dict(parameters=dict(
        actuator=job["actuator"], model_id=job["model_id"], seed=job["seed"],
        condition="A", loss_mode="shared")))
    check_fit_identity(record, job)
    for mutation in ("phase", "condition", "loss_mode", "seed"):
        changed = deepcopy(record)
        if mutation == "phase":
            changed[mutation] = "exploratory"
        else:
            changed["job_inputs"]["parameters"][mutation] = dict(condition="F", loss_mode="one", seed=9000)[mutation]
        with pytest.raises(ValueError):
            check_fit_identity(changed, job)


def test_uncommitted_child_result_is_preserved_before_retry(tmp_path):
    from poc.io import read_json, sha256_file
    from reports.audit_control_nominal_selections import child_output_ready
    output = tmp_path / "cell.result.json"
    output.write_text('{"status":"pass"}')
    digest = sha256_file(output)
    log = output.with_suffix(".stderr.log")
    log.write_text("Interrupted after result write\n")
    assert child_output_ready(output) is False
    assert not output.exists()
    attempts = list(tmp_path.glob("cell.result.interrupted/*"))
    assert len(attempts) == 1
    assert sha256_file(attempts[0] / output.name) == digest
    assert (attempts[0] / log.name).read_text() == "Interrupted after result write\n"
    assert read_json(attempts[0] / "attempt.json")["status"] == "interrupted"
    assert child_output_ready(output) is False


@pytest.mark.parametrize("mutation", ["unchanged", "edited", "missing"])
def test_committed_child_cache_is_reused_or_rejected_without_replacement(tmp_path, mutation):
    from poc.io import atomic_json, sha256_file
    from reports.audit_control_nominal_selections import child_output_ready
    output = tmp_path / "cell.result.json"
    output.write_text('{"status":"pass"}')
    atomic_json(output.with_suffix(".complete.json"), dict(result_sha256=sha256_file(output)))
    if mutation == "edited":
        output.write_text('{"status":"changed"}')
    elif mutation == "missing":
        output.unlink()
    if mutation == "unchanged":
        assert child_output_ready(output) is True
    else:
        with pytest.raises(ValueError):
            child_output_ready(output)
    assert not list(tmp_path.glob("*.interrupted"))


def test_main_retries_uncommitted_child_after_interruption(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from pathlib import Path
    import subprocess
    import sys
    from poc.io import atomic_json, read_json, sha256_file
    from poc import control_confirmation, io
    from reports import audit_control_nominal_selections as audit
    config = tmp_path / "config"
    config.mkdir()
    atomic_json(config / "control_confirmation_cells.json", frozen())
    atomic_json(config / "control_confirmation_design.json", {})
    monkeypatch.setattr(control_confirmation, "validate_design", lambda value: value)
    monkeypatch.setattr(control_confirmation, "verify_freeze", lambda *_: None)
    monkeypatch.setattr(io, "job_lock", lambda *_: nullcontext())
    monkeypatch.setattr(sys, "argv", ["audit", "--root", str(tmp_path)])
    launches = []

    def child(argv, **_):
        job = read_json(Path(argv[argv.index("--child-job") + 1]))
        output = Path(argv[argv.index("--child-output") + 1])
        launches.append(job["cell_id"])
        atomic_json(output, dict(status="pass", job=job, observed={}, resources={},
            script_sha256=sha256_file(Path(audit.__file__)),
            helper_sha256=sha256_file(Path(audit.__file__).with_name("audit_nominal_selections.py"))))
        if len(launches) == 1:
            raise subprocess.CalledProcessError(-15, argv)
        atomic_json(output.with_suffix(".complete.json"), dict(result_sha256=sha256_file(output)))

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(subprocess.CalledProcessError):
        audit.main()
    audit.main()
    assert len(launches) == 31 and launches[0] == launches[1]
