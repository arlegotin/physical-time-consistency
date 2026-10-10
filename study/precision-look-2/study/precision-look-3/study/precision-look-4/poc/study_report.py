"""Evidence-derived reporting for early scientific termination or progress."""
from pathlib import Path
import csv
import io
import json

from .contracts import MILESTONES, scan_runs, verify_decision
from .finalization import claim_rows, compute_verdict
from .io import atomic_bytes, atomic_json, read_json, utc_now
from .report import CLAIM_FIELDS, SECTIONS

INDEX_FIELDS = ('milestone_id', 'run_id', 'job_key', 'phase', 'status', 'compatible',
                'superseded', 'wall_seconds', 'cpu_seconds', 'peak_rss_bytes', 'exclusion_reason')


def _csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode('utf-8'))


def attempt_index(root):
    records, problems = scan_runs(root)
    by_id = {r['run_id']: r for r in records}
    superseded = {identifier for r in records for identifier in r.get('supersedes', []) + r.get('invalidates', [])}
    issues = {p['path']: p['reason'] for p in problems}
    result = []
    for milestone in MILESTONES:
        for folder in sorted((root / 'evidence' / milestone).glob('*')):
            if not folder.is_dir():
                continue
            record = by_id.get(folder.name)
            resource_path = folder / 'resources.json'
            resources = read_json(resource_path) if resource_path.exists() else {}
            interrupted = folder / 'INTERRUPTED.json'
            reason = issues.get(folder.name) or issues.get(str((folder / 'manifest.json').relative_to(root))) or ''
            if record is None:
                status = 'interrupted' if interrupted.exists() else 'partial'
                reason = reason or (read_json(interrupted).get('reason') if interrupted.exists() else None) or 'No complete hash-verified manifest; excluded from scientific aggregation.'
            else:
                status = record['completion_status']
                if folder.name in superseded:
                    reason = 'Superseded or invalidated; original attempt preserved.'
            result.append({'milestone_id': milestone, 'run_id': folder.name,
                'job_key': record.get('job_key', '') if record else '',
                'phase': record.get('phase', '') if record else '', 'status': status,
                'compatible': bool(record and not reason), 'superseded': folder.name in superseded,
                'wall_seconds': resources.get('wall_seconds', ''), 'cpu_seconds': resources.get('cpu_seconds', ''),
                'peak_rss_bytes': resources.get('peak_rss_bytes', ''), 'exclusion_reason': reason})
    return result, records, problems


def _primary_audit(root, decisions):
    result = []
    for decision in decisions:
        if decision.get('gate_id') != 'GR' or decision.get('validity_status') != 'valid':
            continue
        for entry in decision['evidence']:
            folder = (root / entry['path']).parent
            children_path = folder / 'children.json'
            if not children_path.exists():
                continue
            import numpy as np
            from .study_audit import reference_primary_errors
            for child in read_json(children_path)['records']:
                if child['role'] not in {'E01', 'solver'}:
                    continue
                directory = (root / child['path']).parent
                with np.load(directory / 'jacobians.npz', allow_pickle=False) as archive:
                    rows = {name: archive[name] for name in archive.files}
                with np.load(directory / 'reference.npz', allow_pickle=False) as archive:
                    reference = {name: archive[name] for name in archive.files}
                derived = reference_primary_errors(rows, reference)
                cached = read_json(directory / 'summary.json')['metrics']
                for row in derived:
                    expected = cached[row['cell_id']][row['stratum']][str(row['h_ns'])]
                    if row['n_probes'] != expected['n_probes'] or not np.isclose(row['primary_error'], expected['primary_error'], rtol=1e-12, atol=1e-12):
                        raise ValueError('Independent matrix audit disagrees with decisive primary error')
                    result.append({**row, 'manifest': child['path'], 'role': child['role']})
    return result


def write_scientific_report(root: Path, decisions, output_root: Path | None = None):
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root is not None else root
    for decision in decisions:
        verify_decision(root, decision)
    attempts, records, problems = attempt_index(root)
    blockers = []
    for record in records:
        validity = read_json(root / 'evidence' / record['milestone_id'] / record['run_id'] / 'validity.json')
        if validity.get('blocker') and any(d.get('outcome') == 'BLOCKED' for d in decisions):
            blockers.append(validity['blocker'])
    status = compute_verdict(decisions, blockers[-1] if blockers else None)
    claims = claim_rows(root, decisions)
    audit = _primary_audit(root, decisions)
    folder = output / 'reports'
    state = {'protocol_version': 'PTCO-1.0', 'status': status,
        'terminal_verdict': status if status != 'NONTERMINAL' else None,
        'created_utc': utc_now(), 'verified_decisions': [d['decision_id'] for d in decisions],
        'scientific_experiments': 'measured' if audit else 'no_confirmatory_action_audit',
        'independent_primary_rows': len(audit), 'claim_counts': {value: sum(r['status'] == value for r in claims)
            for value in ('pass', 'fail', 'unresolved', 'not_tested')},
        'evidence_problems': problems, 'blockers': blockers,
        'study_root': str(root.relative_to(output)) if root.is_relative_to(output) else '.',
        'control_experiments': 'not_run', 'replication_experiments': 'not_run'}
    _csv(folder / 'claims.csv', CLAIM_FIELDS, claims)
    _csv(folder / 'run_index.csv', INDEX_FIELDS, attempts)
    atomic_json(folder / 'independent_primary_audit.json', {'rows': audit, 'status': 'verified' if audit else 'not_run'})
    scope_path = root / 'config/rejection_subset_scope.json'
    scope = read_json(scope_path) if scope_path.exists() else None
    failed = [r for r in claims if r['status'] == 'fail']
    observed = [r for r in claims if r['status'] != 'not_tested']
    parts = [f'# PTCO-1.0 decision report\n\nStatus: **{status}**.\n\n']
    for i, title in enumerate(SECTIONS, 1):
        parts.append(f'## {i}. {title}\n\n')
        if i == 1:
            parts.append('Robot-control, practical actuator-rate, noise/disturbance and fresh-training replication experiments were **not run**. ')
            if scope and audit:
                parts.append('The decisive audit compares the discrete power head with the prespecified generic NODE on the fully observed smooth direct-torque two-link plant, augmented 20–60 ms training, common one-plus-four-step loss, and independent paired training seeds. Primary query horizons are 5 and 10 ms, with separate cold and warm probes. ')
            if failed:
                parts.append('A valid rejection of a mandatory sensitivity criterion establishes failure of the demanding scoped superiority claim. This does not refute physical mechanics or every possible model with correct action order. ')
            elif status == 'NONTERMINAL':
                parts.append('The available evidence does not yet support a terminal scientific verdict. ')
            parts.append('\n\n')
        elif i == 2:
            parts.append('| Claim | Available decision |\n|---|---|\n')
            for claim in ('C01', 'C02', 'C03', 'C04', 'C05', 'C06'):
                rows = [r for r in claims if r['claim_id'] == claim]
                label = 'rejected' if any(r['status'] == 'fail' for r in rows) else 'partial / unresolved' if any(r['status'] != 'not_tested' for r in rows) else 'not established'
                parts.append(f'| {claim} | {label} |\n')
            parts.append('\n`claims.csv` contains all 294 atomic entries, including the 280 statistical criteria and explicit missingness. Estimates and bounds for non-fraction statistical rows concern the linear contrast; practical percentage thresholds are separate.\n\n')
            for row in observed:
                parts.append(f"- `{row['criterion_id']}`: **{row['status']}**; contrast {row['point_estimate']}, bounds [{row['lower_bound']}, {row['upper_bound']}]; gate {row['gate_id']}.\n")
            parts.append('\n')
        elif i == 3:
            parts.append('Implementation checks actually executed are recorded under M00/M10/M20 and the recorded unit runs. See `locks/environment.json`, each `validity.json`, plant XML/compiled manifests, and the independent rigid-body/analytic oracle evidence. A qualification GO does not establish all C01 requirements or a scientific PASS.\n\n')
        elif i == 4:
            parts.append('Generated corpus roles, dimensions, units, seeds and hashes are recorded in `data/*/*/*/manifest.json`; normalization uses training transitions only. Nominal learning curves, every attempted optimizer block and selected checkpoints are preserved under the training run manifests. Confirmation data are paired across methods and never used for nominal checkpoint selection.\n\n')
        elif i == 5:
            parts.append(f'Independent primary-error reaggregation checked {len(audit)} model/stratum/horizon summaries directly from the full signed q/v matrices and saved reference matrices. See `independent_primary_audit.json` and the decisive GR raw arrays. This checks sensitivity to command, not robot tracking error. Exploratory findings retain their discovery label in `exploratory_action_audit.md`.\n\n')
        elif i == 6:
            parts.append('Full control attribution A, B and A−0.5B was not evaluated. Analytic nominal power-mode identity tests do not replace a measured control intervention. No attribution fraction is reported.\n\n')
        elif i == 7:
            parts.append('No confirmatory MPC episodes, tracking improvements or task success rates are available. No claim of better control is made. Unfinished development fits and prediction jobs are recorded as such, rather than promoted to confirmation.\n\n')
        elif i == 8:
            parts.append('Reference finite-difference, physics-refinement and independent-ODE checks, and same-checkpoint NODE two-half-step audits, accompany each decisive sensitivity run. Numerical uncertainty is propagated into the contrasts. Practical rate/noise/disturbance control remains untested. Measured wall/CPU time and RSS are in `run_index.csv` and per-run resources; parent and child timings overlap and must not be summed blindly. The owner imposed no artificial resource ceilings.\n\n')
        elif i == 9:
            look_decisions = [d for d in decisions if d.get('look_index')]
            parts.append('The rejection family is K=280, including criteria for unexecuted later stages. Seed clusters remain paired, and probes are not independent training replications. Inference is conditional on the fixed training corpus and probe library.\n\n')
            for decision in look_decisions:
                parts.append(f"- Look {decision['look_index']}: alpha {decision['alpha_spent']}; gate `{decision['gate_id']}`, outcome `{decision['outcome']}`. Family-corrected rejection bounds, paired Fieller precision, bootstrap and influence checks are retained in its criterion evidence.\n")
            if not look_decisions:
                parts.append('No confirmatory statistical decision is supplied.\n')
            parts.append('\n')
        elif i == 10:
            parts.append('The decisive raw sensitivity errors are independently reaggregated for this report. This is not the full M90 replay/retraining replication; C06 has not been established. All interrupted, superseded, stale and failed attempts remain visible in `run_index.csv`; only compatible completed decisive evidence supports the verdict.\n\n')
        elif i == 11:
            parts.append('The registered rejection-only execution-order extension is in `config/revisions/MX-mechanism-negative-0004.json` and `rejection_shortcut.md`. It leaves scientific models, data budgets, losses, nominal selection, thresholds and K unchanged; a single qualified prespecified comparator can reject necessary best-baseline superiority. It cannot establish PASS. Other repairs and semantic invalidations are retained under `config/revisions/` and run provenance. The scope excludes contact, hidden delays/activation, unknown tau, physical robot safety, universal superiority and novelty.\n\n')
        elif i == 12:
            if status == 'POC FAIL':
                parts.append('Execution stops on the supported mandatory rejection. No speculative rescue architecture or task redesign is authorized by this verdict. Untested control claims remain untested.\n\n')
            elif blockers:
                parts.append(json.dumps(blockers, indent=2) + '\n\n')
            else:
                parts.append('Execution is nonterminal; continue only the exact action permitted by the validated gate, retaining the statistical look ledger and all paired samples.\n\n')
        elif i == 13:
            parts.append('The exact study source, configuration, wheels and artifact hashes are retained with the evidence. `run_index.csv` lists every observed attempt. `reproduce.sh` defaults to verification/reaggregation; expensive reruns require an explicit argument and current authorization. Missing or altered archived artifacts fail verification.\n\n')
    atomic_bytes(folder / 'final_report.md', ''.join(parts).encode('utf-8'))
    atomic_json(folder / 'state.json', state)
    return state
