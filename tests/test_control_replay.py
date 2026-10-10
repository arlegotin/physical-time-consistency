"""Reproduction keeps workload identity and compares physical-unit tolerances."""
import copy

import numpy as np
import pytest
from poc.io import atomic_json,read_json
from test_control_refinement import private_freeze


def test_replay_requests_use_one_task_per_seed_plant_model_and_clock(tmp_path):
    from poc.control_replay import replay_requests
    frozen=private_freeze(tmp_path)
    requests=replay_requests(tmp_path,frozen)
    assert len(requests)==60 and len({r['cell_id'] for r in requests})==60
    parents={job['cell_id']:job for job in frozen['jobs']}
    for row in requests:
        original=read_json(tmp_path/parents[row['parent_cell_id']]['path'])
        assert row['request']=={**original,'phase':'replication'}
        assert row['request']['episode_id']==0
    assert {r['request']['seed'] for r in requests}==set(range(5))
    assert {r['request']['model_id'] for r in requests}=={'mlp_dt','node_generic','power_correct'}


def test_incomplete_freeze_cannot_become_a_small_apparently_complete_replay(tmp_path):
    from poc.control_replay import replay_requests
    frozen=private_freeze(tmp_path);frozen['jobs'].pop()
    with pytest.raises(ValueError,match='Complete'):
        replay_requests(tmp_path,frozen)


@pytest.mark.parametrize('mutation',['hash','duplicate','seed','clock'])
def test_replay_rejects_changed_frozen_workloads(tmp_path,mutation):
    from poc.control_replay import replay_requests
    from poc.io import sha256_file
    frozen=private_freeze(tmp_path)
    if mutation=='duplicate':frozen['jobs'][-1]=frozen['jobs'][0]
    else:
        job=frozen['jobs'][0];path=tmp_path/job['path'];request=read_json(path)
        if mutation=='seed':request['seed']=9000
        elif mutation=='clock':request['model_partition_ns']=[20_000_000]*2
        else:request['checkpoint_sha256']='c'*64
        atomic_json(path,request)
        if mutation!='hash':job['sha256']=sha256_file(path)
    with pytest.raises(ValueError):replay_requests(tmp_path,frozen)


def known_episode():
    from poc.control_episode import episode_conditions
    task=episode_conditions(0,0,'direct','confirmatory')
    times=np.arange(0,1_600_000_001,250_000,dtype=np.int64)
    state=np.tile(task['initial_state'],(len(times),1))
    command=np.tile([.1,-.2],(40,1))
    summary=dict(phase='confirmatory',actuator='direct',model_id='power_correct',seed=0,episode_id=0,
        condition='clean',model_partition_ns=[5_000_000]*8,actuator_interval_ns=40_000_000,
        physics_step_ns=250_000,power_mode=None,checkpoint_hash='a'*64,complete=True,
        initial_state=task['initial_state'].tolist(),target=task['target'].tolist(),
        seed_namespaces={'cem':task['cem_seed_entropy'],'disturbance':task['disturbance_seed_entropy']},
        realized_cost=2.,success=False,fallback_count=0)
    return dict(summary=summary,time_ns=times,state=state,commands=command,
                decision_statuses=['ok']*40)


def test_exact_fresh_replay_matches_all_clocks_states_actions_and_costs():
    from poc.control_replay import compare_episode
    original=known_episode();replay=copy.deepcopy(original);replay['summary']['phase']='replication'
    result=compare_episode(original,replay)
    assert result['status']=='pass'
    assert result['max_q_difference_rad']==0.
    assert result['max_v_difference_rad_s']==0.
    assert result['max_command_difference_nm']==0.
    assert result['relative_cost_difference']==0.
    assert result['fresh_training'] is False


@pytest.mark.parametrize('component,amount',[('q',2e-6),('v',2e-5),('command',2e-6),('cost',.001)])
def test_replay_outside_original_physical_tolerances_requires_diagnosis(component,amount):
    from poc.control_replay import compare_episode
    original=known_episode();replay=copy.deepcopy(original);replay['summary']['phase']='replication'
    if component=='q':replay['state'][1,0]+=amount
    elif component=='v':replay['state'][1,2]+=amount
    elif component=='command':replay['commands'][0,0]+=amount
    else:replay['summary']['realized_cost']+=amount
    result=compare_episode(original,replay)
    assert result['status']=='outside_tolerance' and result['diagnosis_required'] is True


@pytest.mark.parametrize('mutation',['target','checkpoint','clock','phase','missing','nonfinite','fallback'])
def test_changed_replay_identity_or_partial_data_is_invalid_not_a_numerical_mismatch(mutation):
    from poc.control_replay import compare_episode
    original=known_episode();replay=copy.deepcopy(original);replay['summary']['phase']='replication'
    if mutation=='target':replay['summary']['target'][0]+=.001
    elif mutation=='checkpoint':replay['summary']['checkpoint_hash']='b'*64
    elif mutation=='clock':replay['time_ns'][1]+=1
    elif mutation=='phase':replay['summary']['phase']='exploratory'
    elif mutation=='missing':replay['commands']=replay['commands'][:-1]
    elif mutation=='nonfinite':replay['state'][0,0]=np.nan
    else:replay['decision_statuses'][0]='all_candidates_failed'
    with pytest.raises(ValueError):compare_episode(original,replay)


def test_replay_prepare_refuses_pending_primary_before_creating_any_requests(tmp_path):
    from poc.control_replay import prepare
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',dict(state='running'))
    with pytest.raises(ValueError,match='Complete primary'):
        prepare(tmp_path)
    assert not (tmp_path/'config/control_replay_jobs.json').exists()


def test_replay_runner_reuses_complete_request_identity_and_reports_differences(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import poc.control_replay as replay
    from poc.io import sha256_file
    original=known_episode();repeated=copy.deepcopy(original);repeated['summary']['phase']='replication'
    parent=tmp_path/'evidence/M70/parent/manifest.json';parent.parent.mkdir(parents=True);parent.write_text('{}')
    fresh=tmp_path/'evidence/M70/fresh/manifest.json';fresh.parent.mkdir(parents=True);fresh.write_text('{}')
    request=tmp_path/'config/replay-request.json';atomic_json(request,{'fixture':'request envelope separately validated'})
    job=dict(cell_id='replay-cell',parent_cell_id='primary-cell',path=str(request.relative_to(tmp_path)),
        sha256=sha256_file(request),parent_manifest=str(parent.relative_to(tmp_path)),
        parent_manifest_sha256=sha256_file(parent))
    registry=dict(jobs=[job],input_hashes={},resource_projection={'fixture':True})
    monkeypatch.setattr(replay,'_read_episode',lambda root,path,phase,expected_request:
                        original if phase=='confirmatory' else repeated)
    calls=[]
    def launch(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout=str(fresh)+'\n',stderr='')
    monkeypatch.setattr(replay,'_process_start',lambda pid:'private-metadata',raising=False)
    monkeypatch.setattr('poc.io._process_start',lambda pid:'private-metadata')
    monkeypatch.setattr(replay.subprocess,'run',launch)
    first=replay.run_replays(tmp_path,registry,'a'*64,resume=False)
    assert first['state']=='complete' and first['episodes'][0]['comparison']['status']=='pass'
    assert first['fresh_training'] is False and len(calls)==1
    path=tmp_path/'reports/control_replay.json';before=sha256_file(path)
    monkeypatch.setattr(replay.subprocess,'run',lambda *a,**kw:pytest.fail('Complete replay must reverify, not rerun'))
    second=replay.run_replays(tmp_path,registry,'a'*64,resume=True)
    assert first==second and sha256_file(path)==before
    source=tmp_path/'config/replay-source.json';source.write_text('{}')
    registry['input_hashes']['config/replay-source.json']=sha256_file(source)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Changed frozen replay'):
        replay.run_replays(tmp_path,registry,'a'*64,resume=True)
