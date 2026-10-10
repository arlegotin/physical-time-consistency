"""Reconcile the complete 200-episode development workload without statistical looks."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ACTUATORS = ("direct", "lagged")
MODELS = ("mlp_dt", "gru_dt", "node_generic", "power_correct", "node_structured")
STEPS = (40_000_000, 5_000_000)


def aggregate(rows: list[dict]) -> dict:
    expected = {(a, m, h, e) for a in ACTUATORS for m in MODELS for h in STEPS for e in range(10)}
    indexed = {}
    for row in rows:
        partition = row["model_partition_ns"]
        if (not partition or type(partition[0]) is not int or partition[0] not in STEPS
                or partition != [partition[0]] * (40_000_000 // partition[0])
                or row["seed"] != 9000 or row["phase"] != "exploratory"
                or row["condition"] != "clean" or row["complete"] is not True
                or row["actuator_interval_ns"] != 40_000_000 or row["physics_step_ns"] != 250_000
                or type(row["success"]) is not bool or type(row["fallback_count"]) is not int
                or not 0 <= row["fallback_count"] <= 40 or row["power_mode"] is not None
                or not math.isfinite(row["realized_cost"]) or row["realized_cost"] < 0):
            raise ValueError("Invalid development scope, clocks, failure status or episode cost")
        key = (row["actuator"], row["model_id"], partition[0], row["episode_id"])
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate development cell")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Complete 200-cell development workload required")
    means, contrasts = [], []
    for actuator in ACTUATORS:
        for model in MODELS:
            for step in STEPS:
                cells = [indexed[(actuator, model, step, e)] for e in range(10)]
                means.append(dict(actuator=actuator, model_id=model, model_step_ns=step,
                    mean_cost=math.fsum(x["realized_cost"] for x in cells) / 10,
                    successes=sum(x["success"] for x in cells),
                    fallbacks=sum(x["fallback_count"] for x in cells)))
        for step in STEPS:
            method = [indexed[(actuator, "power_correct", step, e)] for e in range(10)]
            for model in (m for m in MODELS if m != "power_correct"):
                baseline = [indexed[(actuator, model, step, e)] for e in range(10)]
                for x, y in zip(baseline, method):
                    if any(x[k] != y[k] for k in ("initial_state", "target", "seed",
                                                "actuator_interval_ns", "physics_step_ns")):
                        raise ValueError("Development task pairing differs")
                b = math.fsum(x["realized_cost"] for x in baseline) / 10
                m = math.fsum(x["realized_cost"] for x in method) / 10
                differences = [x["realized_cost"] - y["realized_cost"] for x, y in zip(baseline, method)]
                contrasts.append(dict(actuator=actuator, model_step_ns=step, baseline_model=model,
                    baseline_role="mechanistic_control" if model == "node_structured" else "conventional",
                    baseline_mean_cost=b, method_mean_cost=m,
                    relative_reduction=None if b == 0 else 1 - m / b,
                    wins=sum(x > 0 for x in differences), losses=sum(x < 0 for x in differences),
                    ties=sum(x == 0 for x in differences), episode_cost_differences=differences))
    return dict(means=means, contrasts=contrasts, statistical_status="descriptive_only", n_training_seeds=1)


def audit(root: Path) -> dict:
    from poc.contracts import verify_run
    from poc.control_utility import audit_raw_episode
    from poc.io import read_json, sha256_file
    from poc.planner_qualification import training_recovery_compatibility
    from poc.remaining_development import verify_registry

    paths = (root / "reports/control_development_direct.json", root / "reports/control_development_remaining.json")
    registry_path = root / "config/control_development_remaining_jobs.json"
    registry = verify_registry(root, read_json(registry_path))
    batches = [read_json(p) for p in paths]
    for batch, count in zip(batches, (60, 140)):
        if batch["state"] != "complete" or batch["completed_episodes"] != count or len(batch["episodes"]) != count:
            raise ValueError("Complete original and remaining development batches required")
    if batches[1]["registry_sha256"] != sha256_file(registry_path):
        raise ValueError("Remaining batch registry differs")
    jobs = {job["cell_id"]: job for job in registry["jobs"]}
    recovery, _ = training_recovery_compatibility(root)
    dependencies, wheels, rows, verified = {}, {}, [], []
    for batch_index, batch in enumerate(batches):
        for entry in batch["episodes"]:
            path = root / entry["manifest"]
            if sha256_file(path) != entry["manifest_sha256"]:
                raise ValueError("Development manifest changed")
            manifest = read_json(path)
            verify_run(root, manifest, check_dependencies=batch_index == 1)
            if batch_index == 1:
                job = jobs.get(entry["cell_id"])
                if job is None or read_json(path.parent / "request.json") != read_json(root / job["path"]):
                    raise ValueError("Completed development job differs from its frozen request")
            for source, destination in ((manifest["job_inputs"].get("dependencies", {}), dependencies),
                                        (manifest["job_inputs"].get("wheel_hashes", {}), wheels)):
                for name, digest in source.items():
                    destination.setdefault(name, set()).add(digest)
            for name, keys in manifest["job_inputs"].get("configuration_keys", {}).items():
                current = read_json(root / name)
                if any(current.get(k) != v for k, v in keys.items()):
                    raise ValueError("Changed relevant development configuration")
            summary = read_json(path.parent / "summary.json")
            if summary != entry["summary"]:
                raise ValueError("Development receipt summary differs")
            raw = audit_raw_episode(path.parent, summary)
            rows.append(summary)
            verified.append(dict(manifest=entry["manifest"], manifest_sha256=entry["manifest_sha256"], **raw))
            if len(rows) % 20 == 0:
                print(f"Raw development audit {len(rows)}/200", flush=True)
    for name, digests in dependencies.items():
        current = sha256_file(root / name)
        for digest in digests:
            if current != digest and (name not in recovery or recovery[name]["prior_sha256"] != digest):
                raise ValueError("Unexplained historical development dependency change: " + name)
    for name, digests in wheels.items():
        if digests != {sha256_file(root / "wheelhouse" / name)}:
            raise ValueError("Changed development wheel")
    return dict(schema_version=1, phase="exploratory", scope="Both plants, five core models, ten paired tasks at one development training seed; no confirmatory inference",
        complete_episode_count=200, seed_ids=[9000], episodes_per_model_partition=10,
        batch_hashes={str(p.relative_to(root)): sha256_file(p) for p in paths},
        registry_sha256=sha256_file(registry_path), verified_episodes=verified,
        statistical_look_reserved=False, scientific_decision="not_evaluated", **aggregate(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from poc.io import atomic_json, read_json, sha256_file, utc_now

    prior = root / "reports/control_development_complete_audit.json"
    target = root / "reports/control_development_complete_audit_v2.json"
    result = audit(root)
    previous = read_json(prior)
    scientific_fields = set(result)
    if {k: previous.get(k) for k in scientific_fields} != result:
        raise ValueError("Strengthened validation changed the prior measured audit; preserve it and diagnose")
    result["supersedes"] = dict(path=str(prior.relative_to(root)), sha256=sha256_file(prior),
        reason="Strengthen partition and failure-status validation; measured results and statistical status unchanged")
    result.update(recorded_utc=utc_now(), script_sha256=sha256_file(Path(__file__)))
    if target.exists():
        existing = read_json(target)
        if {k: v for k, v in existing.items() if k != "recorded_utc"} != {k: v for k, v in result.items() if k != "recorded_utc"}:
            raise ValueError("Existing completed audit differs; preserve it and diagnose")
    else:
        atomic_json(target, result)
    print(target)


if __name__ == "__main__":
    main()
