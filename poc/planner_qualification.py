"""Freeze the common planner only after complete independent robot calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io import atomic_bytes, atomic_json, read_json, safe_path, sha256_file, utc_now


def qualification(planner, rows, audits):
    """Calculate qualification after the caller verifies all artifact hashes."""
    from .robot_calibration import summarize
    if planner['analytic_qualification']['qualified'] is not True:
        raise ValueError('Analytic search qualification is required')
    expected = {(a, e) for a in ('direct', 'lagged') for e in range(16)}
    identities = [(a['actuator'], a['episode_id']) for a in audits]
    if len(identities) != 32 or set(identities) != expected:
        raise ValueError('All 32 distinct fixed candidate-bank audits are required')
    settings = planner['settings']
    for audit in audits:
        ranking = audit['ranking']
        if (ranking['qualified'] is not True or ranking['resolved_inversions'] != 0
                or ranking['candidate_count'] != settings['candidate_count'] * settings['iterations']):
            raise ValueError('A common-bank ranking check is unqualified or mismatched')
    headroom = summarize(rows)
    if headroom['qualified'] is not True:
        raise ValueError('Independent robot controls have not qualified task headroom')
    return dict(qualified=True, headroom=headroom, bank_count=len(audits),
                learned_model_control='not_run', terminal_verdict=None,
                scope='Planner qualification only; authorizes unchanged learned development')


def verify_planner_transition(prior, current, receipt_entry):
    expected = dict(prior)
    expected.update(status='qualified', robot_reference_qualification='qualified',
                    final_common_budget_frozen=True, recorded_utc=current['recorded_utc'],
                    qualification_evidence=prior['qualification_evidence'] + [receipt_entry])
    if current != expected:
        raise ValueError('Frozen planner changed beyond the evidenced qualification metadata')


def verify_analytic_reports(root, prior):
    from .mpc_calibration import qualification_summary
    files, selected = set(), []
    for entry in prior['qualification_evidence']:
        path = safe_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Analytic calibration report hash changed')
        report = read_json(path)
        files.add(entry['path'])
        if report['completion_status'] != 'complete':
            raise ValueError('Analytic calibration is incomplete')
        for name, digest in report['source_hashes'].items():
            if sha256_file(safe_path(root, name)) != digest:
                raise ValueError('Analytic calibration source changed: ' + name)
            files.add(name)
        measured = qualification_summary(report['rows'])
        if measured != report['qualification']:
            raise ValueError('Analytic calibration summary conflicts with raw rows')
        if report['settings'] == prior['settings'] and measured == prior['analytic_qualification']:
            selected.append(report)
    if len(selected) != 1 or selected[0]['qualification']['qualified'] is not True:
        raise ValueError('Selected analytic budget lacks unique verified qualification')
    return sorted(files)


def verify_inputs(root, batch, *, prior_planner=None, compatible_sources=None):
    from .calibration_batch import validate_registry
    from .contracts import verify_run
    registry = read_json(root / 'config/control_calibration_jobs.json')
    jobs = validate_registry(root, registry)
    if (batch['state'] != 'complete' or batch['completed_episodes'] != 96
            or batch['registry_sha256'] != sha256_file(root / 'config/control_calibration_jobs.json')):
        raise ValueError('A complete compatible calibration batch is required')
    rows, requests, episodes = [], {}, {}

    def load(entry):
        path = safe_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Calibration receipt manifest hash changed')
        record = read_json(path)
        if prior_planner is None:
            verify_run(root, record)
        else:
            # Exact raw completion remains mandatory. Only the declared planner
            # metadata transition is compatible with this historical observation.
            verify_run(root, record, check_dependencies=False)
            inputs = record['job_inputs']
            for relative, digest in inputs.get('dependencies', {}).items():
                actual = prior_planner if relative == 'config/planner.json' else safe_path(root, relative)
                if sha256_file(actual) != digest:
                    compatible = (compatible_sources or {}).get(relative)
                    if (compatible is None or compatible['prior_sha256'] != digest
                            or sha256_file(actual) != compatible['current_sha256']
                            or sha256_file(safe_path(root, compatible['prior_path'])) != digest):
                        raise ValueError(f'Changed calibration dependency: {relative}')
            for relative, keys in inputs.get('configuration_keys', {}).items():
                current = read_json(safe_path(root, relative))
                if any(current.get(k) != v for k, v in keys.items()):
                    raise ValueError('Changed calibration request')
            for filename, digest in inputs.get('wheel_hashes', {}).items():
                if sha256_file(safe_path(root, 'wheelhouse/' + filename)) != digest:
                    raise ValueError('Changed calibration wheel')
            from .contracts import verify_decision
            for decision in record.get('parent_decisions', []):
                verify_decision(root, decision)
        return path

    for entry in batch['episode_manifests']:
        path = load(entry)
        row = read_json(path.parent / 'summary.json')
        identity = (row['actuator'], row['episode_id'], row['model_id'])
        if identity in episodes:
            raise ValueError('Duplicate calibration episode')
        rows.append(row)
        episodes[identity] = entry
        requests[identity] = read_json(path.parent / 'request.json')
    for job in jobs:
        identity = (job['actuator'], job['episode_id'], job['model_id'])
        if identity not in requests or requests[identity] != read_json(safe_path(root, job['request'])):
            raise ValueError('Measured episode differs from its prospective request')
    audits = []
    for entry in batch['bank_audits']:
        path = load(entry)
        audit = read_json(path.parent / 'summary.json')
        source = read_json(path.parent / 'source_episode.json')
        expected = episodes[(audit['actuator'], audit['episode_id'], 'physics_fit')]
        if (source['manifest'] != expected['path'] or source['sha256'] != expected['sha256']
                or source['decision_index'] != 0
                or source['candidate_ids'] != list(range(audit['ranking']['candidate_count']))):
            raise ValueError('Bank audit is not the exact prospective source episode')
        audits.append(audit)
    return rows, audits


def unchanged_source_nodes(prior, current, allowed_functions):
    """Recovery changes cannot exempt a changed numerical or qualification function."""
    import ast
    def retained(source):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                or node.name not in allowed_functions]
    if retained(prior) != retained(current):
        raise ValueError('Unrelated source change cannot use training recovery compatibility')


def training_recovery_compatibility(root):
    root = Path(root)
    revision_path = root / 'config/revisions/training-recovery-repair.json'
    if not revision_path.exists():
        return {}, []
    revision = read_json(revision_path)
    if (revision['schema_version'] != 1 or revision['numerical_algorithms_changed'] is not False
            or revision['source_scope'] != ['poc/train.py', 'poc/planner_qualification.py']):
        raise ValueError('Unexpected training recovery compatibility scope')
    mapping, files = {}, [str(revision_path.relative_to(root))]
    allowed = {'poc/train.py': {'fit', 'save_selection', 'restore_selection'},
               'poc/planner_qualification.py': {'verify_frozen', 'unchanged_source_nodes',
                                              'training_recovery_compatibility'}}
    for entry in revision['compatible_sources']:
        name = entry['path']
        if name not in allowed or name in mapping:
            raise ValueError('Unexpected training recovery source exemption')
        current, prior = safe_path(root, name), safe_path(root, entry['prior_path'])
        if sha256_file(current) != entry['current_sha256'] or sha256_file(prior) != entry['prior_sha256']:
            raise ValueError('Training recovery compatibility source hash changed')
        unchanged_source_nodes(prior.read_text(), current.read_text(), allowed[name])
        mapping[name] = entry
        files.extend((name, entry['prior_path']))
    if set(mapping) != set(allowed):
        raise ValueError('Incomplete training recovery source compatibility')
    return mapping, files


def verify_frozen(root):
    """Read-only authorization; verify all qualification evidence before launch."""
    planner = read_json(root / 'config/planner.json')
    receipt_path = root / 'reports/robot_planner_qualification.json'
    receipt = read_json(receipt_path)
    prior = safe_path(root, receipt['prior_planner']['path'])
    if sha256_file(prior) != receipt['prior_planner']['sha256']:
        raise ValueError('Prior planner configuration hash changed')
    entry = dict(path=str(receipt_path.relative_to(root)), sha256=sha256_file(receipt_path))
    verify_planner_transition(read_json(prior), planner, entry)
    files = set(verify_analytic_reports(root, read_json(prior)))
    files.update(('config/planner.json', entry['path'], receipt['prior_planner']['path'],
                  receipt['batch']['path'], 'config/control_calibration_jobs.json',
                  'config/control_calibration.json', 'reports/robot_calibration_representative.json'))
    batch_path = safe_path(root, receipt['batch']['path'])
    if sha256_file(batch_path) != receipt['batch']['sha256']:
        raise ValueError('Completed calibration batch receipt changed')
    batch = read_json(batch_path)
    compatible = {}
    revision_path = root / 'config/revisions/control-launch-review-repair.json'
    if revision_path.exists():
        revision = read_json(revision_path)
        context_path = safe_path(root, revision['evidence']['path'])
        if sha256_file(context_path) != revision['evidence']['sha256']:
            raise ValueError('Control review compatibility evidence hash changed')
        context = read_json(context_path)
        if (revision['batch_sha256'] != receipt['batch']['sha256']
                or revision['candidate_failure_count'] != 0
                or revision['checked_episode_count'] != 96
                or any(context[k] != revision[k] for k in ('batch_sha256', 'candidate_failure_count', 'checked_episode_count'))
                or revision['numerical_algorithms_changed'] is not False):
            raise ValueError('Control review repair does not apply to this clean calibration')
        compatible = {c['path']: c for c in revision['compatible_sources']}
        if set(compatible) != {'poc/control_jobs.py', 'poc/control_models.py'}:
            raise ValueError('Unexpected calibration source compatibility exemption')
        files.add(str(revision_path.relative_to(root)))
        files.add(revision['evidence']['path'])
        for c in compatible.values():
            if (sha256_file(safe_path(root, c['path'])) != c['current_sha256']
                    or sha256_file(safe_path(root, c['prior_path'])) != c['prior_sha256']):
                raise ValueError('Control review repair source hash changed')
            files.update((c['path'], c['prior_path']))
    recovery, recovery_files = training_recovery_compatibility(root)
    # Calibration did not train a neural model. Its imported train module's
    # numerical functions and qualifier are unchanged; only recovery code changed.
    if recovery:
        compatible['poc/train.py'] = recovery['poc/train.py']
        files.update(recovery_files)
    rows, audits = verify_inputs(root, batch, prior_planner=prior, compatible_sources=compatible)
    calculated = qualification(planner, rows, audits)
    if any(receipt.get(k) != v for k, v in calculated.items()):
        raise ValueError('Recalculated planner qualification conflicts with its receipt')
    for e in batch['episode_manifests'] + batch['bank_audits']:
        record = read_json(safe_path(root, e['path']))
        files.add(e['path'])
        files.add(str((safe_path(root, e['path']).parent / 'COMPLETED').relative_to(root)))
        files.update(p['path'] for p in record['produced_files'])
        files.update(record['job_inputs'].get('dependencies', {}))
        files.update('wheelhouse/' + n for n in record['job_inputs'].get('wheel_hashes', {}))
    return receipt, sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Audit already frozen qualification without rerunning experiments')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    planner_path = root / 'config/planner.json'
    planner = read_json(planner_path)
    if args.verify:
        verify_frozen(root)
        print('Frozen planner qualification and complete historical calibration verified')
        return 0
    if planner['final_common_budget_frozen']:
        raise ValueError('Planner already frozen; verify its qualification receipt instead')
    batch_path = root / 'reports/robot_calibration_batch.json'
    batch = read_json(batch_path)
    rows, audits = verify_inputs(root, batch)
    result = qualification(planner, rows, audits)
    if args.dry_run:
        print(result)
        return 0
    # Preserve the exact prior configuration. Its old run dependencies remain
    # historical; the receipt explicitly establishes unchanged numerical settings.
    prior_path = root / 'config/revisions/planner-before-robot-qualification.json'
    if prior_path.exists() and read_json(prior_path) != planner:
        raise ValueError('Conflicting immutable prior planner configuration')
    atomic_bytes(prior_path, planner_path.read_bytes())
    receipt_path = root / 'reports/robot_planner_qualification.json'
    result.update(schema_version=1, recorded_utc=utc_now(),
        batch=dict(path=str(batch_path.relative_to(root)), sha256=sha256_file(batch_path)),
        prior_planner=dict(path=str(prior_path.relative_to(root)), sha256=sha256_file(prior_path)),
        episode_manifests=batch['episode_manifests'], bank_audits=batch['bank_audits'],
        settings=planner['settings'],
        compatibility='Only qualification flags change; prior completed calibration stays historical with identical planner settings')
    atomic_json(receipt_path, result)
    planner.update(status='qualified', robot_reference_qualification='qualified',
                   final_common_budget_frozen=True, recorded_utc=utc_now())
    planner['qualification_evidence'].append(dict(path=str(receipt_path.relative_to(root)),
                                                 sha256=sha256_file(receipt_path)))
    atomic_json(planner_path, planner)
    print(receipt_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
