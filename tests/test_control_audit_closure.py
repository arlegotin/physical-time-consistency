"""Rechecking raw evidence must preserve its identity and detect corruption."""
from pathlib import Path

import pytest
from poc.io import atomic_json, read_json, sha256_file
from test_control_utility import registered_design, private_episodes


def saved_audit(root):
    from poc.control_utility import seed_costs
    rows=private_episodes()
    for name in ('config/control_confirmation_design.json','config/control_confirmation_cells.json',
                 'reports/control_confirmation_batch.json','poc/control_utility.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}')
    atomic_json(root/'config/control_confirmation_design.json',registered_design())
    entries=[dict(cell_id=f'cell-{i}',manifest=f'evidence/M70/cell-{i}/manifest.json',
                  manifest_sha256='a'*64,summary=row) for i,row in enumerate(rows)]
    batch=dict(state='complete',expected_episodes=1200,completed_episodes=1200,episodes=entries,
               registry_sha256=sha256_file(root/'config/control_confirmation_cells.json'))
    atomic_json(root/'reports/control_confirmation_batch.json',batch)
    audit=dict(state='complete',protocol_id=registered_design()['protocol_id'],
        registry_sha256=sha256_file(root/'config/control_confirmation_cells.json'),
        batch_sha256=sha256_file(root/'reports/control_confirmation_batch.json'),
        design_sha256=sha256_file(root/'config/control_confirmation_design.json'),
        evaluator_sha256=sha256_file(root/'poc/control_utility.py'),
        seed_costs=seed_costs(rows,registered_design()),audited_episodes=[
            dict(cell_id=e['cell_id'],manifest=e['manifest'],manifest_sha256=e['manifest_sha256'],
                 raw_cost=e['summary']['realized_cost'],success=e['summary']['success'],
                 fallback_count=e['summary']['fallback_count'],n_real_anchors=40,n_observations=40)
            for e in entries])
    atomic_json(root/'reports/control_confirmation_raw_audit.json',audit)
    return batch,entries,audit


def test_revalidation_matches_fresh_raw_results_without_rewriting_audit(tmp_path):
    from poc.control_audit_closure import reconcile_primary_audit
    batch,entries,audit=saved_audit(tmp_path)
    path=tmp_path/'reports/control_confirmation_raw_audit.json';before=sha256_file(path)
    actual=[dict(row) for row in audit['audited_episodes']]
    result=reconcile_primary_audit(tmp_path,batch,entries,actual)
    assert result==audit
    assert sha256_file(path)==before


@pytest.mark.parametrize('mutation',['missing','duplicate','cost','success','fallback','scaler',
                                     'batch_hash','evaluator_hash','registry_hash','summary','partial'])
def test_changed_or_incomplete_raw_audit_cannot_authorize_reporting(tmp_path,mutation):
    from poc.control_audit_closure import reconcile_primary_audit
    batch,entries,audit=saved_audit(tmp_path)
    actual=[dict(row) for row in audit['audited_episodes']]
    if mutation=='missing':audit['audited_episodes'].pop()
    elif mutation=='duplicate':audit['audited_episodes'][-1]=audit['audited_episodes'][0]
    elif mutation=='cost':audit['audited_episodes'][0]['raw_cost']+=.1
    elif mutation=='success':audit['audited_episodes'][0]['success']=not actual[0]['success']
    elif mutation=='fallback':audit['audited_episodes'][0]['fallback_count']+=1
    elif mutation=='scaler':audit['seed_costs']['rows'][0]['cost_by_seed'][0]*=2
    elif mutation.endswith('_hash'):audit[mutation.replace('_hash','_sha256')]='b'*64
    elif mutation=='summary':entries[0]['summary']['realized_cost']+=1
    else:audit['state']='interrupted'
    atomic_json(tmp_path/'reports/control_confirmation_raw_audit.json',audit)
    with pytest.raises(ValueError):reconcile_primary_audit(tmp_path,batch,entries,actual)


def test_incomplete_primary_is_rejected_before_creating_or_reserving_any_output(tmp_path):
    from poc.control_audit_closure import load_evidence
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='running',completed_episodes=1,expected_episodes=1200,episodes=[]))
    with pytest.raises(ValueError,match='Complete primary'):
        load_evidence(tmp_path)
    assert not (tmp_path/'decisions/control_utility_looks').exists()
    assert not (tmp_path/'reports/control_confirmation_result.json').exists()


def solver_fixture(root):
    from poc.io import complete_run,job_key
    entries=[]
    for index in range(100):
        cell=f'node-cell-{index}';folder=root/'evidence/M70'/cell;folder.mkdir(parents=True)
        flag=index==17
        summary=dict(cell_id=cell,n_local_probes=40,n_candidate_banks=3,
                     requires_closed_loop_solver_audit=flag,n_ranking_or_failure_changes=int(flag),
                     maximum_absolute_cost_difference=0.)
        for name,value in [('summary.json',summary),('resources.json',{}),
                ('validity.json',dict(validity_status='valid',checks=[],blocker=None)),
                ('local_probes.json',dict(probes=[{} for _ in range(40)])),
                ('candidate_banks.json',dict(banks=[dict(requires_closed_loop_solver_audit=flag and j==0,
                    differences=[dict(maximum_absolute_cost_difference=0.)]*2) for j in range(3)]))]:
            atomic_json(folder/name,value)
        for name in ('stdout.log','stderr.log','events.jsonl'):(folder/name).write_text('')
        inputs=dict(protocol_version='PTCO-1.0',dependencies={})
        manifest=dict(schema_version=1,protocol_version='PTCO-1.0',milestone_id='M70',run_id=cell,
            job_key=job_key(inputs),job_inputs=inputs,phase='confirmatory',argv=['private_fixture'],
            code_revision='fixture',code_tree_sha256='c'*64,dirty=False,configuration_sha256='d'*64,
            environment_sha256='e'*64,data_hashes={},model_hashes={},parent_decisions=[],
            started_utc='2026-10-02T00:00:00Z',ended_utc='2026-10-02T00:00:01Z',
            completion_status='complete',expected_cells=['control.solver'],
            produced_files=[dict(path=str(p.relative_to(root)),sha256=sha256_file(p)) for p in sorted(folder.iterdir())])
        complete_run(root,manifest)
        path=folder/'manifest.json'
        entries.append(dict(parent_cell_id=cell,manifest=str(path.relative_to(root)),
                            manifest_sha256=sha256_file(path),summary=summary,resources={}))
    registry_path=root/'config/control_confirmation_solver_inputs.json'
    atomic_json(registry_path,dict(state='complete',episodes=[dict(cell_id=e['parent_cell_id']) for e in entries]))
    report=dict(state='complete',expected_episodes=100,completed_episodes=100,episodes=entries,
        input_registry_sha256=sha256_file(registry_path),closed_loop_solver_cells_required=['node-cell-17'])
    atomic_json(root/'reports/control_confirmation_solver.json',report)
    return report


def test_solver_closure_reads_complete_manifests_and_preserves_required_diagnosis(tmp_path,monkeypatch):
    from poc.control_audit_closure import _solver_evidence
    import poc.confirmation_solver as solver
    # Input-registry closure has separate actual-model tests; keep this fixture about the raw output envelope.
    monkeypatch.setattr(solver,'_verify_inputs',lambda *a:None)
    expected=solver_fixture(tmp_path)
    report,cells=_solver_evidence(tmp_path)
    assert report==expected and cells==['node-cell-17']


@pytest.mark.parametrize('mutation',['partial','marker','summary','branch','missing'])
def test_solver_closure_rejects_corrupt_completion_or_hidden_ranking_changes(tmp_path,monkeypatch,mutation):
    from poc.control_audit_closure import _solver_evidence
    import poc.confirmation_solver as solver
    monkeypatch.setattr(solver,'_verify_inputs',lambda *a:None)
    report=solver_fixture(tmp_path)
    if mutation=='partial':report['state']='interrupted'
    elif mutation=='marker':(tmp_path/'evidence/M70/node-cell-0/COMPLETED').unlink()
    elif mutation=='summary':report['episodes'][0]['summary']['n_candidate_banks']=2
    elif mutation=='branch':report['closed_loop_solver_cells_required']=[]
    else:report['episodes'].pop()
    atomic_json(tmp_path/'reports/control_confirmation_solver.json',report)
    with pytest.raises(ValueError):_solver_evidence(tmp_path)
