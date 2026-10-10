"""Fail-closed, immutable gate decisions; no scientific inference from setup."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

from . import PROTOCOL_VERSION
from .contracts import (BOOTSTRAP_CHECKS, GATES, MILESTONES, SUCCESSOR,
                        validate_record, verify_decision, verify_run)
from .io import atomic_json, canonical_json, read_json, run_directory, sha256_file, utc_now


def record_inputs(root: Path, gate_id: str, records: list[dict]) -> None:
    root = root.resolve()
    pointer = root / "decisions/gate_inputs.json"
    value = read_json(pointer) if pointer.exists() else {"protocol_version": PROTOCOL_VERSION, "gates": {}}
    value["gates"][gate_id] = [str((run_directory(root, record) / "manifest.json").relative_to(root)) for record in records]
    atomic_json(pointer, value)


def load_inputs(root: Path, gate_id: str) -> list[dict]:
    pointer = root / "decisions/gate_inputs.json"
    if not pointer.exists():
        return []
    from .io import safe_path
    value = read_json(pointer)
    return [read_json(safe_path(root, path)) for path in value["gates"].get(gate_id, [])]


def _evaluation(check: dict) -> dict:
    status = {"pass": "pass", "fail": "fail", "unresolved": "unresolved",
              "invalid": "invalid", "blocked": "unresolved"}.get(check.get("status"))
    if status is None:
        raise ValueError("Mandatory validity check has no recognized status")
    return {"criterion_id": f"G00.{check['check_id']}", "status": status,
            "estimate": 1.0 if status == "pass" else None, "bounds": None,
            "threshold": None, "operator": "qualification",
            "reason": check.get("reason") or "No reason supplied"}


def decide(gate_id: str, evidence: list[dict], root: Path) -> dict:
    root = root.resolve()
    if gate_id not in GATES:
        raise ValueError(f"Unknown gate: {gate_id}")
    milestone = dict(zip(GATES, MILESTONES))[gate_id]
    outcome, validity_status, reasons, evaluations, refs = "INVALID", "invalid", [], [], []
    code_hashes, config_hashes = [], []
    transition = None
    try:
        if not evidence:
            raise ValueError("No complete evidence supplied")
        seen = {}
        for record in evidence:
            if record["milestone_id"] != milestone:
                raise ValueError("Evidence from the wrong milestone")
            verify_run(root, record)
            folder = run_directory(root, record)
            path = folder / "manifest.json"
            refs.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path),
                         "role": "manifest", "producing_run_id": record["run_id"]})
            code_hashes.append(record["code_tree_sha256"])
            config_hashes.append(record["configuration_sha256"])
            validity = read_json(folder / "validity.json")
            if validity["validity_status"] not in {"valid", "blocked", "unresolved"}:
                raise ValueError("Producer declares invalid evidence")
            for check in validity["checks"]:
                if check["check_id"] in seen and seen[check["check_id"]] != check:
                    raise ValueError(f"Conflicting applicable check: {check['check_id']}")
                seen[check["check_id"]] = check
            if validity.get("blocker"):
                if not validity["blocker"].get("required_prerequisite"):
                    raise ValueError("Blocker lacks an external prerequisite")
        if gate_id == "G00":
            if any(record["phase"] != "preflight" for record in evidence):
                raise ValueError("G00 requires preflight evidence")
            if set(BOOTSTRAP_CHECKS) - seen.keys():
                raise ValueError(f"Missing bootstrap cells: {sorted(set(BOOTSTRAP_CHECKS) - seen.keys())}")
            checks = [seen[name] for name in BOOTSTRAP_CHECKS]
            evaluations = [_evaluation(check) for check in checks]
            if any(check["status"] == "invalid" for check in checks):
                raise ValueError("Implementation/measurement invalidity")
            if any(check["status"] == "blocked" for check in checks):
                if not any(read_json(run_directory(root, r) / "validity.json").get("blocker") for r in evidence):
                    raise ValueError("Blocked check has no concrete prerequisite")
                outcome, validity_status = "BLOCKED", "blocked"
            elif any(check["status"] == "fail" for check in checks):
                if all(check.get("failure_kind") == "backend" for check in checks if check["status"] == "fail"):
                    outcome, validity_status = "NO-GO", "valid"
                else:
                    raise ValueError("Bootstrap failure needs implementation diagnosis")
            elif any(check["status"] == "unresolved" for check in checks):
                outcome, validity_status = "INCONCLUSIVE", "unresolved"
            else:
                outcome, validity_status = "GO", "valid"
        elif gate_id == "GF":
            checks = list(seen.values())
            if not checks or any(check["status"] != "pass" for check in checks):
                raise ValueError("Report consistency is not established")
            evaluations = [{**_evaluation(check), "criterion_id": f"GF.{check['check_id']}"} for check in checks]
            outcome, validity_status = "GO", "valid"
            summaries = [read_json(run_directory(root, record) / 'summary.json') for record in evidence]
            terminal = {summary.get('terminal_verdict') for summary in summaries}
            if len(terminal) != 1:
                raise ValueError('Report receipts disagree on the terminal scientific verdict')
            transition = {'terminal_verdict': terminal.pop()}
        else:
            handler = 'poc.milestones.'+milestone.lower()
            if importlib.util.find_spec(handler) is None:
                raise ValueError(f"Gate {gate_id} evaluator is not implemented; it cannot authorize execution")
            module = importlib.import_module(handler)
            if not hasattr(module, 'evaluate'):
                raise ValueError(f'Gate {gate_id} has no implemented evidence evaluator')
            outcome, validity_status, evaluations = module.evaluate(root, evidence, seen)
            if hasattr(module,'transition'):
                transition = module.transition(outcome,evaluations)
            decision_fields = {'criterion_id','status','estimate','bounds','threshold','operator','reason'}
            evaluations = [{key:value for key,value in row.items() if key in decision_fields} for row in evaluations]
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        reasons.append(str(exc))
        outcome, validity_status = "INVALID", "invalid"
    if reasons:
        evaluations.append({"criterion_id": f"{gate_id}.validity", "status": "invalid", "estimate": None,
                            "bounds": None, "threshold": None, "operator": "qualification",
                            "reason": "; ".join(reasons)})
    package = Path(__file__).resolve().parent
    evaluator_files = [package / 'gates.py', package / 'contracts.py', package / 'io.py']
    handler_file = package / 'milestones' / (milestone.lower()+'.py')
    if handler_file.is_file():
        evaluator_files.append(handler_file)
    evaluator_hash = hashlib.sha256(canonical_json({str(p.relative_to(package)): sha256_file(p) for p in evaluator_files})).hexdigest()
    identity = {"gate_id": gate_id, "evidence": refs, "evaluator": evaluator_hash,
                "outcome": outcome, "criteria": evaluations}
    decision_id = hashlib.sha256(canonical_json(identity)).hexdigest()
    path = root / "decisions" / gate_id / f"{decision_id}.json"
    if path.exists():
        saved = read_json(path)
        verify_decision(root, saved)
        if (saved["decision_id"] != decision_id or saved["outcome"] != outcome or
                saved["criterion_evaluations"] != evaluations or saved["evidence"] != refs):
            raise ValueError("Cached decision no longer matches its content identity")
        atomic_json(root / 'decisions/next_action.json',{
            'protocol_version':PROTOCOL_VERSION,'decision_path':str(path.relative_to(root)),
            'decision_sha256':sha256_file(path),'permitted_next_actions':saved['permitted_next_actions']})
        return saved
    next_actions = []
    if outcome == "GO" and gate_id in SUCCESSOR:
        next_actions = [{"milestone_id": SUCCESSOR[gate_id], "branch": None, "question": "Gate prerequisites passed"}]
    elif outcome == "BLOCKED":
        next_actions = [{"milestone_id": "MF", "branch": None, "question": "Report exact external prerequisite"}]
    elif outcome != "GO":
        next_actions = [{"milestone_id": "MX", "branch": "repair" if outcome == "INVALID" else "diagnosis",
                         "question": reasons[0] if reasons else "Resolve the named failed/unresolved qualification"}]
    decision = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION,
                "milestone_id": milestone, "gate_id": gate_id, "decision_id": decision_id,
                "phase": evidence[0]["phase"] if evidence else "preflight", "evidence": refs,
                "code_tree_sha256": hashlib.sha256(canonical_json(code_hashes or [evaluator_hash])).hexdigest(),
                "configuration_sha256": hashlib.sha256(canonical_json(config_hashes)).hexdigest(),
                "validity_status": validity_status, "criterion_evaluations": evaluations,
                "outcome": outcome, "terminal_verdict": None, "permitted_next_actions": next_actions,
                "look_index": None, "alpha_spent": None, "created_utc": utc_now()}
    if transition is not None and validity_status != 'invalid':
        allowed = {'permitted_next_actions','terminal_verdict','look_index','alpha_spent'}
        if set(transition)-allowed:
            raise ValueError('Milestone transition contains undeclared decision fields')
        decision.update(transition)
    validate_record("decision", decision, root)
    atomic_json(path, decision)
    next_actions = decision['permitted_next_actions']
    atomic_json(root / "decisions/next_action.json", {
        "protocol_version": PROTOCOL_VERSION, "decision_path": str(path.relative_to(root)),
        "decision_sha256": sha256_file(path), "permitted_next_actions": next_actions,
    })
    return decision


def current_decision(root: Path) -> dict | None:
    from .io import safe_path
    pointer = root / "decisions/next_action.json"
    if not pointer.exists():
        return None
    selected = read_json(pointer)
    path = safe_path(root, selected["decision_path"])
    if sha256_file(path) != selected["decision_sha256"]:
        raise ValueError("Decision pointer hash mismatch")
    decision = read_json(path)
    verify_decision(root, decision)
    return decision
