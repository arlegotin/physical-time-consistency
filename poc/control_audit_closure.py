"""Read-only closure of primary, reference, finer-physics and NODE evidence."""
from __future__ import annotations
from pathlib import Path

from .io import read_json, safe_path, sha256_file
from .control_utility import audit_raw_episode, seed_costs


def reconcile_primary_audit(root, batch, entries, fresh_raw):
    """A saved audit is evidence; rechecking it must not change its byte identity."""
    root=Path(root)
    audit=read_json(root/'reports/control_confirmation_raw_audit.json')
    design=read_json(root/'config/control_confirmation_design.json')
    expected=dict(state='complete',protocol_id=design['protocol_id'],
        registry_sha256=sha256_file(root/'config/control_confirmation_cells.json'),
        batch_sha256=sha256_file(root/'reports/control_confirmation_batch.json'),
        design_sha256=sha256_file(root/'config/control_confirmation_design.json'),
        evaluator_sha256=sha256_file(root/'poc/control_utility.py'))
    if (any(audit.get(k)!=v for k,v in expected.items())
            or batch.get('state')!='complete' or batch.get('expected_episodes')!=1200
            or batch.get('completed_episodes')!=1200 or len(entries)!=1200
            or audit.get('audited_episodes')!=fresh_raw
            or len(fresh_raw)!=1200 or len({r['cell_id'] for r in fresh_raw})!=1200
            or audit.get('seed_costs')!=seed_costs([entry['summary'] for entry in entries],design)):
        raise ValueError('Saved primary raw audit differs from complete verified observations or sources')
    return audit


def _episode_batch(root,registry,receipt,expected_count,tick):
    from .control_confirmation_batch import episode_receipt
    if (receipt.get('state')!='complete' or receipt.get('expected_episodes')!=expected_count
            or receipt.get('completed_episodes')!=expected_count
            or len(receipt.get('episodes',[]))!=expected_count):
        raise ValueError('Complete registered episode batch required')
    jobs={job['cell_id']:job for job in registry['jobs']}
    if len(jobs)!=expected_count:
        raise ValueError('Unexpected registered episode count')
    rows=[];audits=[];seen=set()
    for entry in receipt['episodes']:
        cell=entry['cell_id'];path=safe_path(root,entry['manifest'])
        if cell in seen or cell not in jobs or sha256_file(path)!=entry['manifest_sha256']:
            raise ValueError('Duplicated, unknown or changed completed episode')
        observed=episode_receipt(root,jobs[cell],path)
        if any(entry.get(k)!=v for k,v in observed.items()):
            raise ValueError('Episode receipt differs from its completed observation')
        raw=audit_raw_episode(path.parent,observed['summary'],expected_tick_ns=tick)
        rows.append(observed)
        audits.append(dict(cell_id=cell,manifest=entry['manifest'],manifest_sha256=entry['manifest_sha256'],**raw))
        seen.add(cell)
    if seen!=set(jobs):
        raise ValueError('Incomplete registered episode coverage')
    return rows,audits


def _solver_evidence(root):
    from .confirmation_solver import _verify_inputs,maximum_solver_cost_shift
    from .contracts import verify_run
    registry_path=root/'config/control_confirmation_solver_inputs.json'
    registry=read_json(registry_path);_verify_inputs(root,registry)
    report=read_json(root/'reports/control_confirmation_solver.json')
    if (report.get('state')!='complete' or report.get('expected_episodes')!=100
            or report.get('completed_episodes')!=100 or len(report.get('episodes',[]))!=100
            or report.get('input_registry_sha256')!=sha256_file(registry_path)):
        raise ValueError('Complete NODE solver audit required')
    expected={entry['cell_id'] for entry in registry['episodes']};seen=set();required=[]
    for entry in report['episodes']:
        cell=entry['parent_cell_id'];path=safe_path(root,entry['manifest'])
        if cell in seen or cell not in expected or sha256_file(path)!=entry['manifest_sha256']:
            raise ValueError('Changed or duplicated NODE audit observation')
        manifest=read_json(path);verify_run(root,manifest)
        summary=read_json(path.parent/'summary.json')
        if (entry['summary']!=summary or manifest['milestone_id']!='M70'
                or manifest['phase']!='confirmatory' or summary['cell_id']!=cell
                or summary['n_local_probes']!=40 or summary['n_candidate_banks']!=3):
            raise ValueError('NODE solver audit summary differs from complete raw evidence')
        probes=read_json(path.parent/'local_probes.json')['probes']
        banks=read_json(path.parent/'candidate_banks.json')['banks']
        flags=[b['requires_closed_loop_solver_audit'] for b in banks]
        maximum=maximum_solver_cost_shift(banks)
        if (len(probes)!=40 or len(banks)!=3 or summary['requires_closed_loop_solver_audit']!=any(flags)
                or summary['n_ranking_or_failure_changes']!=sum(flags)
                or summary['maximum_absolute_cost_difference']!=maximum):
            raise ValueError('NODE solver audit omits or misclassifies raw probes/banks')
        if any(flags):required.append(cell)
        seen.add(cell)
    if seen!=expected or report.get('closed_loop_solver_cells_required')!=required:
        raise ValueError('NODE solver coverage/required branch differs from raw observations')
    return report,required


def verify_primary_audit(root):
    """Verify the existing primary audit in place, preserving every saved hash."""
    root=Path(root).resolve()
    primary=read_json(root/'reports/control_confirmation_batch.json')
    if (primary.get('state')!='complete' or primary.get('expected_episodes')!=1200
            or primary.get('completed_episodes')!=1200 or len(primary.get('episodes',[]))!=1200):
        raise ValueError('Complete primary control batch required before utility evidence closure')
    from .control_confirmation import verify_freeze
    design=read_json(root/'config/control_confirmation_design.json')
    freeze=read_json(root/'config/control_confirmation_cells.json');verify_freeze(root,design,freeze)
    if primary['registry_sha256']!=sha256_file(root/'config/control_confirmation_cells.json'):
        raise ValueError('Primary batch differs from its frozen registry')
    entries,raw=_episode_batch(root,freeze,primary,1200,250_000)
    audit=reconcile_primary_audit(root,primary,entries,raw)
    return dict(design=design,entries=entries,audit=audit)


def load_evidence(root,*,scope='subset'):
    """Validate existing observations only; no training, rewriting, look or verdict."""
    root=Path(root).resolve()
    if scope not in ('subset','full'):
        raise ValueError('Unknown finer-plant scope')
    closed=verify_primary_audit(root)
    design,entries,audit=closed['design'],closed['entries'],closed['audit']
    from .control_references import _verify_registry as verify_references
    reference_path=root/'config/control_reference_jobs.json';reference=read_json(reference_path)
    verify_references(root,reference)
    reference_batch=read_json(root/'reports/control_reference_batch.json')
    if reference_batch['registry_sha256']!=sha256_file(reference_path):
        raise ValueError('Reference receipt differs from its frozen registry')
    ref_entries,ref_raw=_episode_batch(root,reference,reference_batch,450,250_000)
    ref_audit=read_json(root/'reports/control_reference_audit.json')
    if (ref_audit.get('state')!='complete' or ref_audit.get('registry_sha256')!=sha256_file(reference_path)
            or ref_audit.get('batch_sha256')!=sha256_file(root/'reports/control_reference_batch.json')
            or ref_audit.get('audited_episodes')!=ref_raw or ref_audit.get('fitted_physics_models_per_plant')!=1):
        raise ValueError('Reference raw audit differs from complete observations')
    from .control_refinement import _verify_registry as verify_refinement
    refinement_path=root/f'config/control_refinement_{scope}.json';refinement=read_json(refinement_path)
    verify_refinement(root,refinement)
    finer_batch_path=root/f'reports/control_refinement_{scope}_batch.json';finer_batch=read_json(finer_batch_path)
    if finer_batch['registry_sha256']!=sha256_file(refinement_path):
        raise ValueError('Finer-plant receipt differs from its frozen registry')
    fine_entries,_=_episode_batch(root,refinement,finer_batch,300 if scope=='subset' else 1200,125_000)
    from .control_numerical import compare_refinement
    comparison=compare_refinement([e['summary'] for e in entries],[e['summary'] for e in fine_entries],design,scope=scope)
    comparison.update(registry_sha256=sha256_file(refinement_path),batch_sha256=sha256_file(finer_batch_path))
    if comparison!=read_json(root/f'reports/control_refinement_{scope}_audit.json'):
        raise ValueError('Finer-plant audit differs from raw paired observations')
    solver,required=_solver_evidence(root)
    from .control_result import solver_effect_diagnostics
    solver_effects=solver_effect_diagnostics(audit['seed_costs'],[
        dict(cell_id=e['parent_cell_id'],actuator=e['summary']['actuator'],
             model_step_ns=e['summary']['model_step_ns'],
             maximum_absolute_cost_difference=e['summary']['maximum_absolute_cost_difference'])
        for e in solver['episodes']])
    required=sorted(set(required)|{r['cell_id'] for r in solver_effects if r['requires_closed_loop_solver_audit']})
    names=['config/control_confirmation_design.json','config/control_confirmation_cells.json',
        'reports/control_confirmation_batch.json','reports/control_confirmation_raw_audit.json',
        'config/control_reference_jobs.json','reports/control_reference_batch.json','reports/control_reference_audit.json',
        str(refinement_path.relative_to(root)),str(finer_batch_path.relative_to(root)),
        f'reports/control_refinement_{scope}_audit.json','config/control_confirmation_solver_inputs.json',
        'reports/control_confirmation_solver.json','poc/control_audit_closure.py','poc/control_result.py']
    return dict(design=design,primary=[e['summary'] for e in entries],refined=[e['summary'] for e in fine_entries],
        references=[e['summary'] for e in ref_entries],seed_costs=audit['seed_costs'],
        solver_cells_required=required,solver_effect_diagnostics=solver_effects,
        scope=scope,input_hashes={n:sha256_file(root/n) for n in names},
        resources=dict(primary=[e['resources'] for e in entries],references=[e['resources'] for e in ref_entries],
                       refinement=[e['resources'] for e in fine_entries],solver=[e['resources'] for e in solver['episodes']]))
