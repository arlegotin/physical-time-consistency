"""A resumed paired batch keeps complete episodes and every failure status."""
from types import SimpleNamespace

import pytest
from poc.io import atomic_json, complete_run, job_key, read_json, sha256_file


def completed_episode(root, cell, *, fallback=0):
    request = dict(schema_version=1, phase='confirmatory', model_id='power_correct',
        actuator='direct', seed=0, episode_id=cell, condition='clean',
        model_partition_ns=[5_000_000]*8, actuator_interval_ns=40_000_000,
        physics_step_ns=250_000, power_mode=None, model_manifest='evidence/M30/selected/manifest.json',
        manifest_sha256='a'*64, checkpoint_sha256='b'*64,
        planner_settings=dict(candidate_count=128,iterations=4,elite_count=16,initial_std=1.,std_floor=.1),
        resource_projection={'reason':'Private deterministic batch test'})
    path=root/'config/jobs'/f'{cell}.json';atomic_json(path,request)
    folder=root/'evidence/M70'/f'fixture-{cell}';folder.mkdir(parents=True)
    summary=dict(request,complete=True,checkpoint_hash='b'*64,realized_cost=1.+cell,
                 fallback_count=fallback,success=False)
    for name,value in [('request.json',request),('summary.json',summary),
                       ('validity.json',dict(validity_status='valid',checks=[],blocker=None)),
                       ('resources.json',dict(wall_seconds=1.,cpu_seconds=1.,peak_rss_bytes=1))]:
        atomic_json(folder/name,value)
    for name in ('stdout.log','stderr.log','events.jsonl'):
        (folder/name).write_text('')
    inputs=dict(protocol_version='PTCO-1.0',dependencies={str(path.relative_to(root)):sha256_file(path)})
    record=dict(schema_version=1,protocol_version='PTCO-1.0',milestone_id='M70',run_id=folder.name,
        job_key=job_key(inputs),job_inputs=inputs,phase='confirmatory',argv=['private_fixture'],
        code_revision='fixture',code_tree_sha256='c'*64,dirty=False,configuration_sha256='d'*64,
        environment_sha256='e'*64,data_hashes={},model_hashes={},parent_decisions=[],
        started_utc='2026-10-02T00:00:00Z',ended_utc='2026-10-02T00:00:01Z',
        completion_status='complete',expected_cells=['control.episode'],
        produced_files=[dict(path=str(p.relative_to(root)),sha256=sha256_file(p)) for p in sorted(folder.iterdir())])
    complete_run(root,record)
    job=dict(cell_id=f'cell-{cell}',path=str(path.relative_to(root)),sha256=sha256_file(path))
    return job,folder/'manifest.json'


def test_resume_reuses_verified_episode_and_preserves_model_fallback(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    first,p1=completed_episode(tmp_path,0,fallback=1)
    second,p2=completed_episode(tmp_path,1)
    receipt=tmp_path/'reports/control_confirmation_batch.json'
    prior=batch.episode_receipt(tmp_path,first,p1)
    atomic_json(receipt,dict(state='interrupted',registry_sha256='f'*64,episodes=[prior]))
    calls=[]
    def launch(argv,**kwargs):
        calls.append(argv[-1]);return SimpleNamespace(returncode=0,stdout=str(p2)+'\n',stderr='')
    monkeypatch.setattr(batch.subprocess,'run',launch)
    result=batch.run_jobs(tmp_path,[first,second],'f'*64,resume=True)
    assert calls == [second['path']]
    assert result['state']=='complete' and result['completed_episodes']==2
    assert result['episodes'][0]['summary']['fallback_count']==1
    assert result['episodes'][0]['reused'] is True
    assert result['episodes'][1]['reused'] is False


def test_partial_episode_is_not_reused_as_a_completed_observation(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    job,path=completed_episode(tmp_path,0)
    prior=batch.episode_receipt(tmp_path,job,path)
    (path.parent/'COMPLETED').unlink()
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='interrupted',registry_sha256='f'*64,episodes=[prior]))
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:pytest.fail('Corrupt receipt must fail before launch'))
    with pytest.raises(ValueError,match='completion marker'):
        batch.run_jobs(tmp_path,[job],'f'*64,resume=True)


def test_changed_request_fails_before_any_episode_launch(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    job,path=completed_episode(tmp_path,0)
    atomic_json(tmp_path/job['path'],{'changed':True})
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:pytest.fail('Changed request must fail before launch'))
    with pytest.raises(ValueError,match='request hash'):
        batch.run_jobs(tmp_path,[job],'f'*64,resume=True)


def test_interruption_retains_finished_cells_without_partial_costs(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    first,p1=completed_episode(tmp_path,0)
    second,_=completed_episode(tmp_path,1)
    def launch(argv,**kwargs):
        return SimpleNamespace(returncode=0 if argv[-1]==first['path'] else 1,
                               stdout=str(p1)+'\n' if argv[-1]==first['path'] else '',stderr='simulated interruption')
    monkeypatch.setattr(batch.subprocess,'run',launch)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        batch.run_jobs(tmp_path,[first,second],'f'*64,resume=False)
    receipt=read_json(tmp_path/'reports/control_confirmation_batch.json')
    assert receipt['state']=='interrupted' and receipt['completed_episodes']==1
    assert receipt['episodes'][0]['summary']['realized_cost']==1.
    assert 'mean_cost' not in receipt


def test_changed_evaluator_stops_before_launch_and_preserves_interruption(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    job,_=completed_episode(tmp_path,0)
    source=tmp_path/'producer.py';source.write_text('before')
    expected=sha256_file(source);source.write_text('after')
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:pytest.fail('Changed evaluator cannot launch'))
    with pytest.raises(ValueError,match='batch evaluator'):
        batch.run_jobs(tmp_path,[job],'f'*64,resume=False,frozen_sources={'producer.py':expected})
    receipt=read_json(tmp_path/'reports/control_confirmation_batch.json')
    assert receipt['state']=='interrupted' and receipt['completed_episodes']==0


def test_changed_registry_is_never_resumed(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    job,path=completed_episode(tmp_path,0)
    prior=batch.episode_receipt(tmp_path,job,path)
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='interrupted',registry_sha256='f'*64,episodes=[prior]))
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:pytest.fail('Different batch cannot resume'))
    with pytest.raises(ValueError,match='not a resumption'):
        batch.run_jobs(tmp_path,[job],'a'*64,resume=True)


def test_reference_batch_receipts_cannot_overwrite_primary_confirmation(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    from poc.io import sha256_file
    job,manifest=completed_episode(tmp_path,0)
    primary=tmp_path/'reports/control_confirmation_batch.json'
    atomic_json(primary,{'primary_evidence':'preserve byte for byte'})
    original=sha256_file(primary)
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:
                        SimpleNamespace(returncode=0,stdout=str(manifest)+'\n',stderr=''))
    result=batch.run_jobs(tmp_path,[job],'f'*64,resume=True,batch_kind='reference')
    assert result['state']=='complete' and result['completed_episodes']==1
    assert sha256_file(primary)==original
    assert (tmp_path/'reports/control_reference_batch.json').exists()
    assert (tmp_path/'reports/control_reference_attempts.jsonl').exists()


def test_rechecking_a_complete_batch_preserves_receipt_hash_and_statistical_evidence_identity(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    from poc.io import sha256_file
    job,manifest=completed_episode(tmp_path,0)
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:
                        SimpleNamespace(returncode=0,stdout=str(manifest)+'\n',stderr=''))
    first=batch.run_jobs(tmp_path,[job],'f'*64,resume=False)
    path=tmp_path/'reports/control_confirmation_batch.json';original=sha256_file(path)
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:pytest.fail('Completed batch cannot launch again'))
    second=batch.run_jobs(tmp_path,[job],'f'*64,resume=True)
    assert second==first
    assert sha256_file(path)==original


def test_completed_batch_is_reverified_against_current_frozen_sources_before_return(tmp_path,monkeypatch):
    import poc.control_confirmation_batch as batch
    job,manifest=completed_episode(tmp_path,0)
    source=tmp_path/'evaluator.py';source.write_text('original');digest=sha256_file(source)
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**kw:
                        SimpleNamespace(returncode=0,stdout=str(manifest)+'\n',stderr=''))
    batch.run_jobs(tmp_path,[job],'f'*64,resume=False,frozen_sources={'evaluator.py':digest})
    source.write_text('changed')
    with pytest.raises(ValueError,match='batch evaluator'):
        batch.run_jobs(tmp_path,[job],'f'*64,resume=True,frozen_sources={'evaluator.py':digest})
