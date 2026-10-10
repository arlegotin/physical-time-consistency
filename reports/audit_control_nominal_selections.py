"""Recompute every selected control checkpoint's nominal score in its frozen source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def audit_jobs(freeze: dict) -> list[dict]:
    selected = freeze.get("selected", [])
    expected = {(a, m, s) for a in ("direct", "lagged")
                for m in ("mlp_dt", "node_generic", "power_correct") for s in range(5)}
    coordinates = [(r.get("actuator"), r.get("model_id"), r.get("seed")) for r in selected]
    if (freeze.get("state") != "complete" or freeze.get("expected_episodes") != 1200
            or len(selected) != 30 or set(coordinates) != expected
            or any(type(r.get("seed")) is not int for r in selected)):
        raise ValueError("Require the exact frozen thirty-checkpoint control cohort")
    jobs = []
    for row in sorted(selected, key=lambda r: (r["actuator"], r["model_id"], r["seed"])):
        path = Path(row["manifest"])
        if len(path.parts) < 5 or path.parts[-4:-2] != ("evidence", "M30") or path.name != "manifest.json":
            raise ValueError("Selected checkpoint must name its frozen M30 source context")
        jobs.append(dict(row, source_context=str(path.parents[3]),
                         cell_id=f"{row['actuator']}__A__shared__{row['model_id']}__seed{row['seed']}"))
    return jobs


def check_fit_identity(record: dict, job: dict) -> None:
    parameters = record.get("job_inputs", {}).get("parameters", {})
    if (record.get("milestone_id") != "M30" or record.get("phase") != "confirmatory"
            or any(parameters.get(k) != v for k, v in dict(
                actuator=job["actuator"], model_id=job["model_id"], seed=job["seed"],
                condition="A", loss_mode="shared").items())):
        raise ValueError("Nominal audit fit differs from the frozen confirmatory cell")


def child_output_ready(output: Path) -> bool:
    """Reuse committed bytes; retain an interrupted attempt before rerunning it."""
    from poc.io import atomic_json, read_json, sha256_file, utc_now
    marker = output.with_suffix(".complete.json")
    if marker.exists():
        if not output.exists() or read_json(marker).get("result_sha256") != sha256_file(output):
            raise ValueError("Missing or changed committed nominal-audit result")
        return True
    pending = [p for p in (output, output.with_suffix(".stdout.log"),
               output.with_suffix(".stderr.log"), output.with_suffix(".data_access.jsonl")) if p.exists()]
    if pending:
        attempts = output.with_name(output.stem + ".interrupted")
        attempts.mkdir(exist_ok=True)
        number = 1
        while (attempts / f"{number:04}").exists():
            number += 1
        attempt = attempts / f"{number:04}"
        attempt.mkdir()
        record = dict(status="interrupted", reason="Child lacked its atomic completion marker",
            recorded_utc=utc_now(), files={p.name: sha256_file(p) for p in pending}, preservation_complete=False)
        atomic_json(attempt / "attempt.json", record)
        for path in pending:
            path.rename(attempt / path.name)
        atomic_json(attempt / "attempt.json", dict(record, preservation_complete=True))
    return False


def audit_one(root: Path, job_file: Path, output: Path) -> None:
    """A fresh interpreter imports the original trainer and redirects only audit logs."""
    import importlib.metadata
    import platform
    import resource
    import time

    job = json.loads(job_file.read_text())
    source = (root / job["source_context"]).resolve()
    if output.is_relative_to(source):
        raise ValueError("Audit output must stay outside the frozen training source")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    import poc
    if Path(poc.__file__).resolve().parent.parent != source:
        raise ValueError("Audit imported a different training source")
    import torch
    from poc import dataset
    from poc.contracts import verify_run
    from poc.execution import cpu_defaults
    from poc.io import atomic_json, read_json, safe_path, sha256_file, utc_now
    from poc.models.common import Scales
    from poc.models.factory import build_model
    from poc.train import canonical_tensor_hash, validate_nominal
    from audit_nominal_selections import assert_scores, load_validation_for_audit

    start, wall, cpu = utc_now(), time.perf_counter(), time.process_time()
    cpu_defaults(job["seed"])
    manifest_path = safe_path(root, job["manifest"])
    if sha256_file(manifest_path) != job["manifest_sha256"]:
        raise ValueError("Frozen selected manifest changed")
    record = read_json(manifest_path)
    verify_run(source, record)
    check_fit_identity(record, job)
    for name, version in read_json(source / "locks/environment.json")["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Installed nominal-audit environment differs: " + name)
    parameters = record["job_inputs"]["parameters"]
    for name, expected in parameters["data_hashes"].items():
        if sha256_file(safe_path(source, name)) != expected:
            raise ValueError("Changed frozen nominal-training input: " + name)
    validation = safe_path(source, parameters["validation_manifest"])
    if read_json(validation)["role"] != "validation":
        raise ValueError("Selected-score audit permits nominal validation labels only")
    arrays = load_validation_for_audit(dataset, validation, output.with_suffix(".data_access.jsonl"))
    folder = manifest_path.parent
    normalization = read_json(folder / "normalization.json")
    scales = Scales(*(torch.tensor(normalization[k], dtype=torch.float32)
                      for k in ("scale_q", "scale_v", "velocity_input_rms")))
    model = build_model(job["model_id"], job["actuator"], scales, job["seed"])
    weights_path = folder / "weights.pt"
    if sha256_file(weights_path) != job["checkpoint_sha256"]:
        raise ValueError("Frozen selected checkpoint changed")
    weights = torch.load(weights_path, weights_only=True, map_location="cpu")
    before = canonical_tensor_hash(weights)
    if before != job["canonical_tensor_sha256"] or before != record["model_hashes"]["canonical_tensor_sha256"]:
        raise ValueError("Frozen canonical selected tensors changed")
    model.load_state_dict(weights, strict=True)
    selected = read_json(folder / "selection.json")
    observed = validate_nominal(model, arrays, scales, "shared")
    assert_scores(observed, selected)
    assert_scores(observed, read_json(folder / "summary.json")["validation"])
    if canonical_tensor_hash(model.state_dict()) != before:
        raise ValueError("Nominal audit mutated selected parameters")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(output, dict(schema_version=1, status="pass", job=job,
        script_sha256=sha256_file(Path(__file__)), helper_sha256=sha256_file(Path(__file__).with_name("audit_nominal_selections.py")),
        observed=observed, selected=selected, validation_manifest_sha256=sha256_file(validation),
        validation_array_sha256=sha256_file(validation.parent / "trajectories.npz"),
        source_dependency_hashes=record["job_inputs"]["dependencies"],
        started_utc=start, completed_utc=utc_now(),
        resources=dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
            peak_rss_bytes=int(peak * (1 if platform.system() == "Darwin" else 1024))),
        checkpoint_changed=False, training=False, reselection=False, fine_time_labels_opened=False))
    atomic_json(output.with_suffix(".complete.json"), dict(result_sha256=sha256_file(output)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--child-job", type=Path)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.child_job is not None:
        if args.child_output is None:
            raise ValueError("Explicit child output required")
        audit_one(root, args.child_job.resolve(), args.child_output.resolve())
        return
    sys.path.insert(0, str(root))
    from poc.control_confirmation import validate_design, verify_freeze
    from poc.io import atomic_json, job_key, job_lock, read_json, sha256_file, utc_now
    frozen_path = root / "config/control_confirmation_cells.json"
    freeze = read_json(frozen_path)
    jobs = audit_jobs(freeze)
    verify_freeze(root, validate_design(read_json(root / "config/control_confirmation_design.json")), freeze)
    inputs = dict(registry_sha256=sha256_file(frozen_path), script_sha256=sha256_file(Path(__file__)),
                  helper_sha256=sha256_file(Path(__file__).with_name("audit_nominal_selections.py")))
    if args.dry_run:
        print(json.dumps(dict(execute=False, expected_cells=len(jobs), labels="nominal validation only", **inputs)))
        return
    prior = root / "reports/control_nominal_selection_audit.json"
    target = root / "reports/control_nominal_selection_audit_v2.json"
    folder = root / "reports/control_nominal_selection_audit" / job_key(inputs)
    folder.mkdir(parents=True, exist_ok=True)
    with job_lock(job_key(dict(nominal_selection_audit=inputs)), root):
        results = []
        for job in jobs:
            request = folder / (job["cell_id"] + ".request.json")
            output = folder / (job["cell_id"] + ".result.json")
            if request.exists() and read_json(request) != job:
                raise ValueError("Changed nominal-audit request")
            if not request.exists():
                atomic_json(request, job)
            if not child_output_ready(output):
                with output.with_suffix(".stdout.log").open("a") as stdout, output.with_suffix(".stderr.log").open("a") as stderr:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                        "--child-job", str(request), "--child-output", str(output)], check=True,
                        stdout=stdout, stderr=stderr, cwd=root)
            result = read_json(output)
            marker = output.with_suffix(".complete.json")
            if (not marker.exists() or read_json(marker).get("result_sha256") != sha256_file(output)
                    or result.get("status") != "pass" or result.get("job") != job
                    or any(result.get(k) != inputs[k] for k in ("script_sha256", "helper_sha256"))):
                raise ValueError("Incompatible cached nominal audit")
            results.append(dict(cell_id=job["cell_id"], path=str(output.relative_to(root)),
                                sha256=sha256_file(output), observed=result["observed"], resources=result["resources"]))
            print(f"Nominal selected-score audit {len(results)}/30: {job['cell_id']}", flush=True)
        summary = dict(schema_version=1, state="complete", completed_cells=30, expected_cells=30,
            inputs=inputs, cells=results, training=False, reselection=False, statistical_look_reserved=False,
            fine_time_outcomes_opened=False, scope="Selected-weight nominal validation integrity; no utility or replication verdict")
        if prior.exists():
            previous = read_json(prior)
            if {c["cell_id"]: c["observed"] for c in previous["cells"]} != {c["cell_id"]: c["observed"] for c in results}:
                raise ValueError("Repaired audit changed previous measured nominal scores; preserve and diagnose")
            summary["supersedes"] = dict(path=str(prior.relative_to(root)), sha256=sha256_file(prior),
                reason="Repair interrupted-child recovery; measured selected scores unchanged")
        if target.exists():
            if {k: v for k, v in read_json(target).items() if k != "recorded_utc"} != summary:
                raise ValueError("Existing completed audit differs; preserve and diagnose")
        else:
            atomic_json(target, dict(summary, recorded_utc=utc_now()))
        print(target)


if __name__ == "__main__":
    main()
