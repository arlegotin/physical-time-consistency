"""Honest stdlib-only reporting, including setup blockers and missing science."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from . import PROTOCOL_VERSION
from .contracts import scan_runs, verify_decision
from .io import atomic_bytes, atomic_json, read_json, utc_now

CLAIM_FIELDS = ["protocol_version", "claim_id", "criterion_id", "cell_id", "status", "threshold",
                "point_estimate", "lower_bound", "upper_bound", "numerical_envelope", "n_seeds",
                "n_episodes", "n_probes", "look_id", "validity", "evidence_paths", "evidence_hashes",
                "gate_id", "reason"]
SECTIONS = ["Verdict and scope", "Claim-by-claim decisions", "Implementation and correctness",
            "Data and selection", "Action-effect evidence", "Causal attribution", "Prediction and control",
            "Robustness and computation", "Statistical integrity", "Replication and audit",
            "Deviations and limitations", "Unresolved questions and resumption", "Reproduction and artifact index"]


def _csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def write_report(root: Path, decisions: list[dict]) -> dict:
    root = root.resolve()
    verified, rejected = [], []
    for decision in decisions:
        try:
            verify_decision(root, decision)
            verified.append(decision)
        except (ValueError, OSError, KeyError) as exc:
            rejected.append({"decision_id": decision.get("decision_id"), "reason": str(exc)})
    runs, problems = scan_runs(root)
    blockers = []
    for run in runs:
        folder = root / "evidence" / run["milestone_id"] / run["run_id"]
        validity = read_json(folder / "validity.json")
        if validity.get("blocker"):
            blockers.append({"milestone_id": run["milestone_id"], "run_id": run["run_id"], **validity["blocker"]})
    status = "BLOCKED" if blockers and any(d["outcome"] == "BLOCKED" for d in verified) else "NONTERMINAL"
    summary = {"protocol_version": PROTOCOL_VERSION, "status": status,
               "terminal_verdict": "BLOCKED" if status == "BLOCKED" else None,
               "scientific_experiments": "not_run", "created_utc": utc_now(),
               "verified_decisions": [d["decision_id"] for d in verified],
               "rejected_decisions": rejected, "evidence_problems": problems, "blockers": blockers}
    claims = [{"protocol_version": PROTOCOL_VERSION, "claim_id": f"C{i:02d}",
               "criterion_id": "", "cell_id": "", "status": "not_tested", "threshold": "",
               "point_estimate": "", "lower_bound": "", "upper_bound": "", "numerical_envelope": "",
               "n_seeds": "", "n_episodes": "", "n_probes": "", "look_id": "", "validity": "not_run",
               "evidence_paths": "[]", "evidence_hashes": "[]", "gate_id": "",
               "reason": "Scientific criterion evidence not obtained; quantities inapplicable."}
              for i in range(1, 7)]
    attempts = []
    for run in runs:
        resources = read_json(root / "evidence" / run["milestone_id"] / run["run_id"] / "resources.json")
        attempts.append({"milestone_id": run["milestone_id"], "run_id": run["run_id"],
                         "job_key": run["job_key"], "status": run["completion_status"],
                         "wall_seconds": resources.get("wall_seconds"), "cpu_seconds": resources.get("cpu_seconds"),
                         "peak_rss_bytes": resources.get("peak_rss_bytes"), "exclusion_reason": ""})
    for problem in problems:
        attempts.append({"milestone_id": "", "run_id": problem["path"], "job_key": "", "status": "invalid",
                         "wall_seconds": "", "cpu_seconds": "", "peak_rss_bytes": "", "exclusion_reason": problem["reason"]})
    folder = root / "reports"
    atomic_bytes(folder / "claims.csv", _csv_bytes(CLAIM_FIELDS, claims))
    atomic_bytes(folder / "run_index.csv", _csv_bytes(
        ["milestone_id", "run_id", "job_key", "status", "wall_seconds", "cpu_seconds", "peak_rss_bytes", "exclusion_reason"], attempts))
    parts = [f"# PTCO-1.0 decision report\n\nStatus: **{status}**.\n\n"]
    for index, title in enumerate(SECTIONS, 1):
        parts.append(f"## {index}. {title}\n\n")
        if index == 1:
            parts.append("No required robot training, MuJoCo dynamics experiment, sensitivity audit or control evaluation has been run. "
                         "This report concerns the available execution envelope; no scientific verdict is supported. "
                         "The intended scope is the fully observed smooth two-link robot, direct/lagged actuation and discrete power head.\n\n")
        elif index == 2:
            parts.append("C01–C06: **not tested**. The complete scientific criteria require later gated evidence. See `claims.csv`.\n\n")
        elif index == 3:
            parts.append(f"Available complete setup/report runs: {len(runs)}. Only checks explicitly recorded in their manifests are execution evidence. "
                         "Planning reviews and source citations are not completed experiment checks.\n\n")
        elif index == 8:
            parts.append("Measured available-run resources are in `run_index.csv` and each run's `resources.json`. "
                         "The owner imposed no artificial RAM, disk or CPU-time limits; actual shortages require owner provisioning.\n\n")
        elif index == 9:
            parts.append("Statistical confirmation not run. The preregistered family has 280 statistical atomic criteria; "
                         "no statistical look or alpha expenditure has been consumed by setup.\n\n")
        elif index == 11:
            parts.append("No scientific protocol change. Resource authorization permits null RAM/disk/time ceilings because the owner explicitly imposed none. "
                         "Real robots, contact, hidden activation, unknown tau and real-time deployment remain outside scope.\n\n")
        elif index == 12:
            if blockers:
                for blocker in blockers:
                    parts.append(f"- {blocker['milestone_id']} / {blocker['run_id']}: {blocker['reason']}. "
                                 f"Required prerequisite: {blocker['required_prerequisite']}.\n")
                parts.append("\nResume the exact blocked milestone after provision; keep its failed attempt.\n\n")
            else:
                parts.append("Execution is nonterminal. Use `python -m poc next --dry-run` for the validated next action.\n\n")
        elif index == 13:
            parts.append("`run_index.csv` lists attempts; `claims.csv` preserves missing claims. "
                         "`reproduce.sh` verifies/reaggregates available artifacts by default and does not launch experiments. "
                         "Missing scientific inputs cannot be reproduced before they exist.\n\n")
        else:
            parts.append("Not run. No measurements or result claims are available for this section.\n\n")
    if rejected or problems:
        parts.append("## Evidence requiring repair\n\n")
        for problem in [*rejected, *problems]:
            parts.append(f"- {problem}\n")
    atomic_bytes(folder / "final_report.md", "".join(parts).encode("utf-8"))
    atomic_json(folder / "state.json", summary)
    script = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--rerun-experiments" ]]; then
  echo 'Scientific reruns are unavailable until their implementations and frozen inputs exist.' >&2
  exit 2
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: reports/reproduce.sh [--rerun-experiments]' >&2
  exit 2
fi
python3.11 -m poc validate --scope protocol
python3.11 -m poc status --verify
python3.11 -m poc run MF
python3.11 -m poc validate --scope release
"""
    atomic_bytes(folder / "reproduce.sh", script.encode("utf-8"))
    return summary
