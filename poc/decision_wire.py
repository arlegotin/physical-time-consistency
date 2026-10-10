"""Schema-compatible GR serializer around the unchanged numeric gate evaluator."""
from pathlib import Path
import argparse
import hashlib
import json
import math

from .io import atomic_json, canonical_json, read_json, run_directory, sha256_file, utc_now


def decision_row(row):
    keys = ('criterion_id', 'status', 'estimate', 'bounds', 'threshold', 'operator', 'reason')
    result = {key: row[key] for key in keys}
    bounds = result['bounds']
    if bounds is not None:
        if isinstance(bounds, list):
            if len(bounds) != 2:
                raise ValueError('Statistical bounds require exactly two endpoints')
            bounds = {'lower': bounds[0], 'upper': bounds[1]}
        if (set(bounds) != {'lower', 'upper'} or
                not all(type(v) in (int, float) and math.isfinite(v) for v in bounds.values()) or
                bounds['lower'] > bounds['upper']):
            raise ValueError('Invalid statistical interval')
        result['bounds'] = bounds
    return result


def decide_rejection(root, evidence):
    from .contracts import validate_record, verify_run, verify_decision
    from .gates import record_inputs
    from .milestones.mr import evaluate, transition
    root = Path(root).resolve()
    if len(evidence) != 1 or evidence[0]['milestone_id'] != 'MR':
        raise ValueError('Bound-interface repair supports only the executed MR gate')
    record = evidence[0]
    verify_run(root, record)
    folder = run_directory(root, record)
    validity = read_json(folder / 'validity.json')
    if validity['validity_status'] != 'valid':
        raise ValueError('Invalid producer evidence cannot enter a repaired scientific gate')
    seen = {c['check_id']: c for c in validity['checks']}
    outcome, validity_status, rows = evaluate(root, evidence, seen)
    evaluations = [decision_row(row) for row in rows]
    action = transition(outcome, rows)
    files = ('poc/decision_wire.py', 'poc/milestones/mr.py', 'poc/rejection_subset.py',
             'poc/criterion_gates.py', 'poc/statistics.py', 'poc/contracts.py', 'poc/io.py')
    evaluator = {path: sha256_file(root / path) for path in files}
    manifest = folder / 'manifest.json'
    refs = [{'path': str(manifest.relative_to(root)), 'sha256': sha256_file(manifest),
             'role': 'manifest', 'producing_run_id': record['run_id']}]
    identity = {'gate_id': 'GR', 'evidence': refs, 'evaluator': evaluator,
                'outcome': outcome, 'criteria': evaluations}
    identifier = hashlib.sha256(canonical_json(identity)).hexdigest()
    decision = {'schema_version': 1, 'protocol_version': 'PTCO-1.0', 'milestone_id': 'MR',
        'gate_id': 'GR', 'decision_id': identifier, 'phase': 'confirmatory', 'evidence': refs,
        'code_tree_sha256': hashlib.sha256(canonical_json(evaluator)).hexdigest(),
        'configuration_sha256': record['configuration_sha256'], 'validity_status': validity_status,
        'criterion_evaluations': evaluations, 'outcome': outcome, 'terminal_verdict': None,
        'permitted_next_actions': [], 'look_index': None, 'alpha_spent': None,
        'created_utc': utc_now(), **action}
    validate_record('decision', decision, root)
    path = root / 'decisions/GR' / f'{identifier}.json'
    if path.exists():
        saved = read_json(path)
        verify_decision(root, saved)
        if any(saved[name] != decision[name] for name in decision if name != 'created_utc'):
            raise ValueError('Repaired decision no longer matches its evidence identity')
        decision = saved
    else:
        atomic_json(path, decision)
    record_inputs(root, 'GR', evidence)
    atomic_json(root / 'decisions/next_action.json', {'protocol_version': 'PTCO-1.0',
        'decision_path': str(path.relative_to(root)), 'decision_sha256': sha256_file(path),
        'permitted_next_actions': decision['permitted_next_actions']})
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', choices=['GR'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    from .gates import load_inputs
    decision = decide_rejection(args.root, load_inputs(args.root, args.gate))
    print(json.dumps(decision, indent=2, allow_nan=False))
    return 0 if decision['outcome'] == 'GO' else 1


if __name__ == '__main__':
    raise SystemExit(main())
