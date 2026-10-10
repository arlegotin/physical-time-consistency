"""Standard-library validators and explicit PTCO milestone contracts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import TypedDict

from . import PROTOCOL_VERSION
from .io import canonical_json, job_key, read_json, run_directory, safe_path, sha256_file

MILESTONES = ("M00", "M10", "M20", "M30", "MR", "M40", "M50", "M60", "M70", "M80", "M90", "MX", "MF")
GATES = ("G00", "G10", "G20", "G30", "GR", "G40", "G50", "G60", "G70", "G80", "G90", "GX", "GF")
PHASES = ("preflight", "unit", "calibration", "exploratory", "confirmatory", "replication", "report")
OUTCOMES = ("GO", "NO-GO", "INCONCLUSIVE", "INVALID", "BLOCKED")
BOOTSTRAP_CHECKS = ("authorization", "python", "architecture", "virtual_environment", "disk", "packages",
                    "determinism", "nstep", "torch_smoke", "wheel_lock", "offline_install", "validators")
SUCCESSOR = {"G00": "M10", "G10": "M20", "G20": "M30", "G30": "M40", "GR": "M40",
             "G40": "M50", "G50": "M60", "G60": "M70", "G70": "M80", "G80": "M90", "G90": "MF"}
BASE_CONTRACT = {
    "protocol_version": "PTCO-1.0", "device": "cpu", "train_dtype": "float32", "audit_dtype": "float64",
    "h0_s": .04, "tau_s": .08, "physics_step_s": .0005, "observation_interval_s": .04,
    "replanning_interval_s": .04, "history_s": .32,
    "sensitivity_horizons_s": [.0025, .005, .01, .02, .04, .08],
    "model_partitions_s": [.04, .02, .01, .005, .0025], "primary_model_step_s": .005,
    "actuator_intervals_s": [.04, .02, .01], "mpc_horizon_s": .24, "episode_s": 1.6,
    "torque_limit_nm": 2., "initial_training_seeds": [0, 1, 2, 3, 4], "episodes_per_seed": 20,
    "initial_epochs": 100, "effective_batch": 256, "learning_rate": .001, "weight_decay": .000001,
    "gradient_clip_norm": 1., "rollout_loss_weight": .25, "rollout_steps": 4, "bootstrap_resamples": 20000,
}


class JobSpec(TypedDict):
    protocol_version: str
    milestone_id: str
    phase: str
    purpose: str
    semantic_inputs: dict
    expected_cells: list[str]


class RunRecord(TypedDict):
    schema_version: int
    protocol_version: str
    milestone_id: str
    run_id: str
    job_key: str
    job_inputs: dict
    phase: str
    argv: list[str]
    code_revision: str
    code_tree_sha256: str
    dirty: bool
    configuration_sha256: str
    data_hashes: dict
    model_hashes: dict
    environment_sha256: str
    parent_decisions: list
    started_utc: str
    ended_utc: str
    completion_status: str
    expected_cells: list[str]
    produced_files: list[dict]


class DecisionRecord(TypedDict):
    schema_version: int
    protocol_version: str
    milestone_id: str
    gate_id: str
    decision_id: str
    phase: str
    evidence: list[dict]
    code_tree_sha256: str
    configuration_sha256: str
    validity_status: str
    criterion_evaluations: list[dict]
    outcome: str
    terminal_verdict: str | None
    permitted_next_actions: list[dict]
    look_index: int | None
    alpha_spent: float | None
    created_utc: str


class CriterionRecord(TypedDict):
    criterion_id: str
    claim_id: str
    metric: str
    direction: str
    dimensions: dict
    threshold: float
    required_precision: float | None
    evidence_schema: str
    validity_dependencies: list[str]
    statistical: bool
    allowed_next_action: dict


class MXRequest(TypedDict):
    schema_version: int
    request_id: str
    branch: str
    parent_gate: str
    return_milestone: str
    protocol_version: str
    claim_ids: list[str]
    unresolved_question: str
    live_explanations: list[str]
    input_evidence: list[dict]
    new_information: str
    action_parameters: dict
    expected_artifacts: list[str]
    progress_condition: str
    resource_projection: dict
    authorization_reference: dict
    status: str


def _validate_schema(value, schema: dict, location: str = "$", *, recursive=True) -> None:
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                _validate_schema(value, alternative, location)
                return
            except ValueError:
                pass
        raise ValueError(f"{location}: no allowed schema matches")
    if "type" in schema:
        kind = schema["type"]
        if not isinstance(value, types[kind]) or (kind in {"integer", "number"} and isinstance(value, bool)):
            raise ValueError(f"{location}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location}: unsupported value {value!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{location}: empty string")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"{location}: invalid string format")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError(f"{location}: missing {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            raise ValueError(f"{location}: unknown fields {sorted(value.keys() - properties.keys())}")
        for key in value.keys() & properties.keys():
            _validate_schema(value[key], properties[key], f"{location}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{location}: insufficient entries")
        for i, item in enumerate(value):
            _validate_schema(item, schema.get("items", {}), f"{location}[{i}]")


def validate_record(kind: str, value: dict, root: Path) -> None:
    if kind not in {"manifest", "decision"}:
        raise ValueError(f"Unknown record kind: {kind}")
    schema_path = root / "schemas" / f"{kind}.schema.json"
    if not schema_path.exists():
        schema_path = Path(__file__).resolve().parent.parent / "schemas" / f"{kind}.schema.json"
    _validate_schema(value, read_json(schema_path))
    canonical_json(value)  # Reject nonfinite nested values even when schema is unconstrained.
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Incompatible protocol")
    for name in ("started_utc", "ended_utc", "created_utc"):
        if name in value:
            instant = datetime.fromisoformat(value[name].replace("Z", "+00:00"))
            if instant.utcoffset() is None:
                raise ValueError(f"{name} requires an explicit timezone")
    if kind == "manifest":
        if job_key(value["job_inputs"]) != value["job_key"]:
            raise ValueError("Manifest semantic job key mismatch")
        paths = [entry["path"] for entry in value["produced_files"]]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate artifact paths")
        for path in paths:
            safe_path(root, path)
        names = {Path(path).name for path in paths}
        needed = {"stdout.log", "stderr.log", "events.jsonl", "resources.json", "validity.json", "summary.json"}
        if needed - names:
            raise ValueError(f"Missing required run artifacts: {sorted(needed - names)}")
        if len(value["expected_cells"]) != len(set(value["expected_cells"])):
            raise ValueError("Duplicate expected cells")
        if value["started_utc"] > value["ended_utc"]:
            raise ValueError("Run ends before it starts")
    else:
        if dict(zip(GATES, MILESTONES))[value["gate_id"]] != value["milestone_id"]:
            raise ValueError("Gate/milestone mismatch")
        for entry in value["evidence"]:
            safe_path(root, entry["path"])
        if value["outcome"] == "GO" and not value["evidence"]:
            raise ValueError("GO requires evidence")


def verify_run(root: Path, record: RunRecord, *, check_dependencies: bool = True) -> None:
    validate_record("manifest", record, root)
    folder = run_directory(root, record)
    if not (folder / "COMPLETED").is_file():
        raise ValueError(f"Partial run lacks completion marker: {record['run_id']}")
    marker = read_json(folder / "COMPLETED")
    if marker.get("manifest_sha256") != sha256_file(folder / "manifest.json"):
        raise ValueError("Completion marker/manifest hash mismatch")
    if canonical_json(read_json(folder / "manifest.json")) != canonical_json(record):
        raise ValueError("Referenced manifest differs from completed manifest")
    for entry in record["produced_files"]:
        if sha256_file(safe_path(root, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {entry['path']}")
    if not check_dependencies:
        return  # Historical integrity is separate from authorization under current code.
    for filename, expected in record['job_inputs'].get('wheel_hashes', {}).items():
        if sha256_file(safe_path(root, 'wheelhouse/' + filename)) != expected:
            raise ValueError(f'Wheel hash mismatch: {filename}')
    for relative, expected in record["job_inputs"].get("dependencies", {}).items():
        if sha256_file(safe_path(root, relative)) != expected:
            raise ValueError(f"Stale semantic dependency: {relative}")
    for relative, keys in record["job_inputs"].get("configuration_keys", {}).items():
        current = read_json(safe_path(root, relative))
        if any(current.get(key) != expected for key, expected in keys.items()):
            raise ValueError(f"Stale relevant configuration: {relative}")
    for decision in record.get('parent_decisions',[]):
        verify_decision(root,decision)


def verify_decision(root: Path, decision: DecisionRecord) -> None:
    validate_record("decision", decision, root)
    for entry in decision["evidence"]:
        path = safe_path(root, entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Decision evidence changed: {entry['path']}")
        if entry["role"] == "manifest":
            verify_run(root, read_json(path))


def scan_runs(root: Path) -> tuple[list[RunRecord], list[dict]]:
    records, problems = [], []
    for path in sorted((root / "evidence").glob("*/*/manifest.json")):
        try:
            record = read_json(path)
            verify_run(root, record, check_dependencies=False)
            records.append(record)
        except (ValueError, OSError, KeyError) as exc:
            problems.append({"path": str(path.relative_to(root)), "reason": str(exc)})
    active, stale, superseded = [], {}, set()
    for record in records:
        try:
            verify_run(root, record)
            active.append(record)
            superseded.update(record.get('supersedes', []))
            superseded.update(record.get('invalidates', []))
        except (ValueError, OSError, KeyError) as exc:
            stale[record['run_id']] = str(exc)
    for run_id, reason in stale.items():
        if run_id not in superseded:
            problems.append({'path': run_id, 'reason': reason})
    keys = {}
    for record in records:
        if record['completion_status'] != 'complete' or record['run_id'] in superseded:
            continue
        previous = keys.setdefault(record["job_key"], record["run_id"])
        if previous != record["run_id"]:
            problems.append({"path": record["run_id"], "reason": f"Duplicate completed job also has {previous}"})
    return records, problems


def validate_protocol(root: Path) -> dict:
    base = read_json(root / "config/base.json")
    if set(base) != set(BASE_CONTRACT):
        raise ValueError("Unknown or missing base configuration keys")
    if canonical_json(base) != canonical_json(BASE_CONTRACT):
        raise ValueError("Base configuration differs from PTCO-1.0; a protocol revision is required")
    registry = read_json(root / "config/milestones.json")
    if set(registry["milestones"]) != set(MILESTONES):
        raise ValueError("Milestone registry coverage mismatch")
    for milestone, gate in zip(MILESTONES, GATES):
        entry = registry["milestones"][milestone]
        if entry["gate"] != gate or set(entry["predecessors"]) - set(GATES) or set(entry["successors"]) - set(MILESTONES):
            raise ValueError(f"Invalid transition registry: {milestone}")
    criteria = read_json(root / "config/criteria.json")
    if criteria.get("statistical_family_size") != 280:
        raise ValueError("PTCO-1.0 statistical family must remain 280")
    return {"status": "valid", "criteria_status": criteria["status"], "statistical_family_size": 280}
