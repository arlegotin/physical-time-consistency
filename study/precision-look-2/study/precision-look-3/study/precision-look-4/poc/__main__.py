"""Thin, lazy argparse dispatcher for explicitly implemented PTCO contracts."""
from __future__ import annotations

import argparse
import json
import importlib
import importlib.util
import os
from pathlib import Path
import sys

from .contracts import GATES, MILESTONES, scan_runs, validate_protocol
from .gates import current_decision, decide, load_inputs
from .io import read_json


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poc", description="PTCO-1.0 evidence-gated local experiment")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default current directory)")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute an implemented milestone contract")
    run.add_argument("milestone", choices=MILESTONES)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    audit = commands.add_parser("decide", help="Independently re-evaluate exact selected gate evidence")
    audit.add_argument("gate", choices=GATES)
    status = commands.add_parser("status", help="List complete/partial/stale evidence")
    status.add_argument("--verify", action="store_true")
    follow = commands.add_parser("next", help="Show the validated allowed next action")
    follow.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate", help="Validate protocol, evidence or release closure")
    validate.add_argument("--scope", choices=("protocol", "evidence", "release"), required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    try:
        if args.command == "decide":
            decision = decide(args.gate, load_inputs(root, args.gate), root)
            _print(decision)
            return 0 if decision["outcome"] == "GO" else 1
        if args.command in {"status", "next"}:
            records, problems = scan_runs(root)
            decision = current_decision(root)
            permitted = decision["permitted_next_actions"] if decision else [{"milestone_id": "M00"}]
            _print({"runs": [{"run_id": r["run_id"], "milestone_id": r["milestone_id"],
                              "status": r["completion_status"]} for r in records],
                    "problems": problems, "permitted_next_actions": permitted})
            return 1 if problems else 0
        if args.command == "validate":
            result = validate_protocol(root)
            if args.scope in {"evidence", "release"}:
                records, problems = scan_runs(root)
                if problems:
                    raise ValueError(f"Evidence closure errors: {problems}")
                current_decision(root)
                result["complete_runs"] = len(records)
            if args.scope == "release":
                state = read_json(root / "reports/state.json")
                if state["status"] not in {"NONTERMINAL", "BLOCKED"}:
                    raise ValueError("Scientific terminal evaluation is not implemented yet")
                for name in ("final_report.md", "claims.csv", "run_index.csv", "reproduce.sh"):
                    if not (root / "reports" / name).is_file():
                        raise ValueError(f"Missing release artifact: {name}")
                result["report_status"] = state["status"]
            _print(result)
            return 0
        if args.dry_run:
            handler = 'poc.milestones.'+args.milestone.lower()
            implemented = args.milestone in {"M00", "MF"} or importlib.util.find_spec(handler) is not None
            description = {}
            if implemented and args.milestone not in {'M00','MF'}:
                module = importlib.import_module(handler)
                if hasattr(module,'describe'):
                    description = module.describe(root)
                else:
                    description = {'expected_cells':list(getattr(module,'CHECKS',())),
                                   'expected_jobs':[{'job_id':args.milestone+'-deterministic-audit','status':'verify_or_produce'}]}
            _print({"milestone_id": args.milestone, "implemented": implemented,
                    "execute": False, "expected_cells": list(__import__("poc.contracts", fromlist=["BOOTSTRAP_CHECKS"]).BOOTSTRAP_CHECKS)
                    if args.milestone == "M00" else [],
                    "expected_jobs":[{'job_id':args.milestone,'status':'verify_or_produce'}] if implemented else [],
                    "resource_projection": {'status':'not_measured','reason':'No scientific batch is launched by this command'},
                    "prohibiting_conditions": [] if implemented else ["Handler not implemented"],**description})
            return 0 if implemented else 1
        if args.milestone == "MF":
            from .report import write_report
            decision = current_decision(root)
            _print(write_report(root, [decision] if decision else []))
            return 0
        if args.milestone == "M00":
            from .bootstrap import run_m00
            record = run_m00(root, read_json(root / "config/authorization.json"))
            decision = decide("G00", [record], root)
            _print(decision)
            return 0 if decision['outcome'] == 'GO' else 1
        handler = 'poc.milestones.'+args.milestone.lower()
        if importlib.util.find_spec(handler) is None:
            raise RuntimeError(f"Milestone {args.milestone} is not implemented; predecessors cannot be bypassed")
        record = importlib.import_module(handler).run(root, resume=args.resume)
        decision = decide('G'+args.milestone[1:], [record], root)
        _print(decision)
        return 0 if decision['outcome'] == 'GO' else 1
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"PTCO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
