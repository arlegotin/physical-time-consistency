"""Solver checks preserve the full confirmation workload and saved candidate bank."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.io import read_json


def full_batch():
    from poc.control_episode import episode_conditions
    entries=[]
    for a in ('direct','lagged'):
        for m in ('mlp_dt','node_generic','power_correct'):
            for s in range(5):
                for h in (40_000_000,5_000_000):
                    for e in range(20):
                        task=episode_conditions(s,e,a,'confirmatory')
                        entries.append(dict(manifest=f'episodes/{a}_{m}_{s}_{h}_{e}/manifest.json',
                            manifest_sha256='a'*64,summary=dict(actuator=a,model_id=m,seed=s,
                            episode_id=e,phase='confirmatory',complete=True,condition='clean',
                            physics_step_ns=250_000,actuator_interval_ns=40_000_000,power_mode=None,
                            model_partition_ns=[h]*(40_000_000//h),realized_cost=1.,success=False,
                            fallback_count=0,initial_state=task['initial_state'].tolist(),target=task['target'].tolist(),
                            seed_namespaces={'cem':task['cem_seed_entropy'],'disturbance':task['disturbance_seed_entropy']})))
    return dict(state='complete',expected_episodes=1200,completed_episodes=1200,episodes=entries)


def test_confirmation_solver_uses_all_seeds_plants_and_both_clocks_without_best_task_selection():
    from poc.confirmation_solver import confirmation_subset
    result=confirmation_subset(full_batch(),read_json(Path('config/control_confirmation_design.json')))
    assert len(result)==100
    assert {(r['summary']['actuator'],r['summary']['seed']) for r in result}=={
        (a,s) for a in ('direct','lagged') for s in range(5)}
    assert {r['summary']['episode_id'] for r in result}==set(range(5))
    assert {r['summary']['model_partition_ns'][0] for r in result}=={5_000_000,40_000_000}
    assert all(r['summary']['model_id']=='node_generic' for r in result)


@pytest.mark.parametrize('mutation',['partial','missing','duplicate','task','namespace','phase','clock'])
def test_solver_subset_rejects_missing_or_relabelled_confirmation(mutation):
    from poc.confirmation_solver import confirmation_subset
    batch=full_batch()
    if mutation=='partial':batch['state']='running'
    elif mutation=='missing':batch['episodes'].pop()
    elif mutation=='duplicate':batch['episodes'][-1]=batch['episodes'][0]
    else:
        row=batch['episodes'][0]['summary']
        if mutation=='task':row['target'][0]+=.01
        elif mutation=='namespace':row['seed_namespaces']['cem'][0]=7101
        elif mutation=='phase':row['phase']='exploratory'
        else:row['model_partition_ns']=[20_000_000]*2
    with pytest.raises(ValueError):
        confirmation_subset(batch,read_json(Path('config/control_confirmation_design.json')))


def unit_mass_node():
    from poc.models.ode import NeuralODE
    from poc.models.common import Scales
    model=NeuralODE('direct',Scales(torch.ones(2),torch.ones(2),torch.ones(2))).double()
    model.network=torch.nn.Linear(8,4,dtype=torch.float64)
    with torch.no_grad():
        model.network.weight.zero_();model.network.bias.zero_()
        model.network.weight[2,6]=.08;model.network.weight[3,7]=.08
    return model


def exact_cost():
    times=[.04,.08,.12,.16,.20,.24]
    # Two independent unit masses, zero initial state/target, constant unit commands.
    return math.fsum(.04*(2*(.5*t*t)**2+.05*2*t*t+.001*2) for t in times)+5*2*(.5*.24**2)**2


def test_candidate_solver_check_replays_same_bank_and_preserves_physical_cost_anchors():
    from poc.confirmation_solver import audit_candidate_bank
    model=unit_mass_node();before=copy.deepcopy(model.state_dict())
    knots=np.stack([np.zeros((6,2)),np.ones((6,2)),np.ones((6,2))])
    result=audit_candidate_bank(model,torch.zeros(4,dtype=torch.float64),np.zeros(2),knots,
        np.array([0.,exact_cost(),exact_cost()]),('ok',)*3,('',)*3,0,(5_000_000,)*8)
    assert result['substeps']==[1,2,4]
    assert result['best_candidate_ids']==[0,0,0]
    assert result['requires_closed_loop_solver_audit'] is False
    assert result['costs'][0][0]==0.
    assert result['costs'][2][1]==pytest.approx(exact_cost(),abs=1e-12)
    assert len(result['anchors'])==3 and np.asarray(result['anchors']).shape==(3,3,6,4)
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())
    assert result['work'][0]['vector_field_calls']==192
    assert result['work'][1]['vector_field_calls']==384
    assert result['work'][2]['vector_field_calls']==768


@pytest.mark.parametrize('mutation',['cost','best','status','nan'])
def test_solver_audit_detects_saved_cost_or_selected_candidate_confounds(mutation):
    from poc.confirmation_solver import audit_candidate_bank
    costs=np.array([0.,exact_cost()]);statuses=['ok','ok'];selected=0
    if mutation=='cost':costs[1]+=.001
    elif mutation=='best':selected=1
    elif mutation=='status':statuses[1]='model_failure'
    else:costs[1]=np.nan
    with pytest.raises(ValueError):
        audit_candidate_bank(unit_mass_node(),torch.zeros(4,dtype=torch.float64),np.zeros(2),
            np.stack([np.zeros((6,2)),np.ones((6,2))]),costs,statuses,('',)*2,selected,(40_000_000,))


def test_solver_audit_does_not_drop_failed_candidate_costs_or_rank_infinity_as_best():
    from poc.confirmation_solver import audit_candidate_bank
    model=unit_mass_node()
    # A deliberately failed NODE field is represented by its guarded output, not omitted.
    with torch.no_grad():model.network.weight[2,6]=8000.
    result=audit_candidate_bank(model,torch.zeros(4,dtype=torch.float64),np.zeros(2),
        np.stack([np.zeros((6,2)),np.ones((6,2))]),np.array([0.,np.inf]),
        ('ok','model_failure'),('','imagined_state_guard'),0,(40_000_000,))
    assert result['best_candidate_ids']==[0,0,0]
    assert all(row[1] is None for row in result['costs'])
    assert all(row[1]=='model_failure' for row in result['statuses'])
    assert result['failed_candidate_count']==[1,1,1]


def test_changed_candidate_ranking_requires_closed_loop_solver_diagnosis():
    from poc.confirmation_solver import audit_candidate_bank
    model=unit_mass_node()
    with torch.no_grad():
        model.network.weight[2,4]=-4.;model.network.weight[3,5]=-4.
    # q_dot=v, v_dot=u-100*v. At h=.04, RK4's velocity multiplier is 5.
    q=v=0.;terms=[]
    for _ in range(6):
        q,v=q+.0008-.04*v,5*v-.04
        terms.append(.04*(2*(q-.1)**2+.05*2*v*v+.001*2))
    saved=np.array([.1048,math.fsum(terms)+5*2*(q-.1)**2])
    result=audit_candidate_bank(model,torch.zeros(4,dtype=torch.float64),np.array([.1,.1]),
        np.stack([np.zeros((6,2)),np.ones((6,2))]),saved,('ok','ok'),('',''),0,(40_000_000,))
    assert result['best_candidate_ids']==[0,1,1]
    assert result['requires_closed_loop_solver_audit'] is True
    assert result['scientific_decision']=='not_evaluated'


def test_solver_wrapper_keeps_initial_lag_activation_fixed_and_counts_actual_substeps():
    from poc.confirmation_solver import SolverResolution
    from poc.models.ode import NeuralODE
    from poc.models.common import Scales
    model=NeuralODE('lagged',Scales(torch.ones(2),torch.ones(2),torch.ones(2))).double()
    state=torch.tensor([[.1,-.2,.3,.1,.7,-.4]],dtype=torch.float64)
    u=torch.tensor([[.5,-.8]],dtype=torch.float64)
    before=state.clone()
    prediction=SolverResolution(model,4).predict(state,u,.005)
    assert prediction.vector_field_calls==16
    assert torch.equal(state,before)
    expected=state[:,4:]+(u-state[:,4:])*(-math.expm1(-.005/.08))
    assert torch.allclose(prediction.next_state[:,4:],expected,atol=1e-12,rtol=1e-12)


def test_confirmation_solver_dry_run_cannot_require_or_create_unmeasured_outcomes(tmp_path,capsys):
    from poc.confirmation_solver import main
    from poc.io import atomic_json
    atomic_json(tmp_path/'config/control_confirmation_solver_audit.json',dict(
        schema_version=1,protocol_id='UTILITY-2026-10-02-control-v1',model_id='node_generic',
        phase='confirmatory',actuators=['direct','lagged'],training_seed_ids=list(range(5)),
        episode_ids=list(range(5)),horizons_ns=[5_000_000,40_000_000],substeps=[1,2,4],
        command_namespace=9104,candidate_decision_ids=[0,20,39]))
    assert main(['--root',str(tmp_path),'--dry-run'])==0
    output=capsys.readouterr().out
    assert '4000' in output and '300' in output
    assert not (tmp_path/'reports/control_confirmation_solver.json').exists()


def saved_zero_bank(folder):
    import json
    from poc.io import atomic_json
    folder.mkdir(parents=True,exist_ok=True)
    states=np.zeros((40,4));times=np.arange(40,dtype=np.int64)*40_000_000
    np.savez(folder/'observations.npz',state=states,time_ns=times,packet_id=[f'obs-{t}' for t in times],
             noise_time_ns=np.array([],dtype=np.int64))
    np.savez(folder/'candidates.npz',knots=np.zeros((40,2,6,2)),costs=np.zeros((40,2)),
        statuses=np.full((40,2),'ok'),reasons=np.full((40,2),''),global_ids=np.tile([0,1],(40,1)))
    (folder/'decisions.jsonl').write_text(''.join(json.dumps(dict(time_ns=int(t),
        observation_packet_id=f'obs-{t}',status='ok',selected_global_id=0,selected_knots=np.zeros((6,2)).tolist()))+'\n'
        for t in times))
    request=dict(model_id='node_generic',actuator='direct',seed=0,episode_id=0,phase='confirmatory',
        condition='clean',model_partition_ns=[5_000_000]*8,physics_step_ns=250_000,
        actuator_interval_ns=40_000_000,power_mode=None,checkpoint_sha256='b'*64,
        target=[0.,0.],planner_settings=dict(candidate_count=2,iterations=1,elite_count=2))
    atomic_json(folder/'request.json',request)
    return request


def test_episode_solver_probes_only_saved_observations_and_preserves_files(tmp_path):
    from poc.confirmation_solver import probe_saved_episode
    from poc.io import sha256_file
    request=saved_zero_bank(tmp_path/'episode');folder=tmp_path/'episode'
    hashes={p.name:sha256_file(p) for p in folder.iterdir()}
    local,banks=probe_saved_episode(unit_mass_node(),folder,request,np.zeros(2))
    assert len(local)==40 and len(banks)==3
    assert [r['decision_id'] for r in banks]==[0,20,39]
    assert {r['time_ns'] for r in local}==set(range(0,1_600_000_000,40_000_000))
    assert local[20]['command_seed_entropy']==[9104,0,0,0,20]
    assert all(row['horizon_ns']==5_000_000 for row in local)
    assert all(row['requires_closed_loop_solver_audit'] is False for row in banks)
    assert {p.name:sha256_file(p) for p in folder.iterdir()}==hashes


@pytest.mark.parametrize('mutation',['extra_observation','packet','candidate_id','knots'])
def test_episode_solver_refuses_added_observations_or_changed_saved_plan(tmp_path,mutation):
    import json
    from poc.confirmation_solver import probe_saved_episode
    folder=tmp_path/'episode';request=saved_zero_bank(folder)
    if mutation in ('extra_observation','packet'):
        with np.load(folder/'observations.npz') as data:values={k:data[k].copy() for k in data.files}
        if mutation=='extra_observation':values['time_ns'][1]=5_000_000
        else:values['packet_id'][1]='obs-5000000'
        np.savez(folder/'observations.npz',**values)
    elif mutation=='candidate_id':
        with np.load(folder/'candidates.npz') as data:values={k:data[k].copy() for k in data.files}
        values['global_ids'][20,0]=1
        np.savez(folder/'candidates.npz',**values)
    else:
        values=[json.loads(line) for line in (folder/'decisions.jsonl').read_text().splitlines()]
        values[39]['selected_knots'][0][0]=.1
        (folder/'decisions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in values))
    with pytest.raises(ValueError):probe_saved_episode(unit_mass_node(),folder,request,np.zeros(2))


def test_solver_input_freeze_cannot_be_created_from_partial_primary_observations(tmp_path):
    from poc.confirmation_solver import prepare
    from poc.io import atomic_json
    atomic_json(tmp_path/'config/control_confirmation_solver_audit.json',
                read_json(Path('config/control_confirmation_solver_audit.json')))
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',
                dict(state='running',expected_episodes=1200,completed_episodes=1,episodes=[]))
    with pytest.raises(ValueError,match='Complete paired confirmation'):
        prepare(tmp_path)
    assert not (tmp_path/'config/control_confirmation_solver_inputs.json').exists()


def test_changed_solver_probe_namespace_rejects_even_read_only_readiness(tmp_path):
    from poc.confirmation_solver import main
    from poc.io import atomic_json
    design=read_json(Path('config/control_confirmation_solver_audit.json'));design['command_namespace']=9102
    atomic_json(tmp_path/'config/control_confirmation_solver_audit.json',design)
    with pytest.raises(ValueError,match='registered confirmation solver'):
        main(['--root',str(tmp_path),'--dry-run'])


def test_changed_solver_source_fails_before_reusing_any_completed_inputs(tmp_path):
    from poc.confirmation_solver import _verify_inputs
    from poc.io import sha256_file
    source=tmp_path/'producer.py';source.write_text('original')
    digest=sha256_file(source);source.write_text('changed')
    registry=dict(state='complete',episodes=[{} for _ in range(100)],input_hashes={'producer.py':digest})
    with pytest.raises(ValueError,match='Changed confirmation solver input'):
        _verify_inputs(tmp_path,registry)


def test_changed_solver_source_during_batch_stops_before_model_queries_and_retains_interruption(tmp_path,monkeypatch):
    import poc.confirmation_solver as module
    from poc.io import atomic_json,sha256_file
    atomic_json(tmp_path/'config/control_confirmation_solver_audit.json',
                read_json(Path('config/control_confirmation_solver_audit.json')))
    source=tmp_path/'poc/producer.py';source.parent.mkdir();source.write_text('original')
    registry=dict(state='complete',episodes=[{'cell_id':f'cell-{i}'} for i in range(100)],
        parent_jobs={f'cell-{i}':{} for i in range(100)},input_hashes={'poc/producer.py':sha256_file(source)})
    atomic_json(tmp_path/'config/control_confirmation_solver_inputs.json',registry)
    def prepared(root):
        source.write_text('changed after input verification');return registry
    monkeypatch.setattr(module,'prepare',prepared)
    monkeypatch.setattr(module,'_episode_audit',lambda *args:pytest.fail('Changed source cannot query or reuse a model'))
    with pytest.raises(ValueError,match='Changed frozen solver source'):
        module.main(['--root',str(tmp_path),'--resume'])
    receipt=read_json(tmp_path/'reports/control_confirmation_solver.json')
    assert receipt['state']=='interrupted' and receipt['completed_episodes']==0
    assert receipt['scientific_decision']=='not_evaluated'


def test_changed_cem_elites_are_detected_when_global_best_is_unchanged():
    from poc.confirmation_solver import candidate_elite_ids
    settings=dict(candidate_count=3,iterations=2,elite_count=2)
    coarse=np.array([0.,1.,2.,3.,4.,5.])
    fine=np.array([0.,2.,1.,3.,4.,5.])
    assert np.argmin(coarse)==np.argmin(fine)==0
    assert candidate_elite_ids(coarse,settings)==[[0,1],[3,4]]
    assert candidate_elite_ids(fine,settings)==[[0,2],[3,4]]


def test_cem_elite_solver_audit_preserves_finite_mask_and_stable_ties():
    from poc.confirmation_solver import candidate_elite_ids
    assert candidate_elite_ids(np.array([np.inf,1.,1.,np.inf]),
        dict(candidate_count=2,iterations=2,elite_count=2))==[[1],[2]]
    with pytest.raises(ValueError):
        candidate_elite_ids(np.zeros(3),dict(candidate_count=2,iterations=2,elite_count=1))


def test_actual_solver_elite_change_routes_to_closed_loop_despite_same_best_plan():
    from poc.confirmation_solver import audit_candidate_bank,SolverResolution
    from poc.mpc import rollout_candidates
    model=unit_mass_node()
    with torch.no_grad():
        model.network.weight[2,4]=-4.;model.network.weight[3,5]=-4.
    state=torch.zeros(4,dtype=torch.float64)
    knots=np.zeros((4,6,2));knots[1]=1.;knots[2,0]=1.;knots[3,-1]=1.1
    saved=rollout_candidates(SolverResolution(model,1),state,np.zeros(2),None,knots,(40_000_000,),40_000_000)
    result=audit_candidate_bank(model,state,np.zeros(2),knots,saved['costs'],saved['statuses'],
        saved['reasons'],0,(40_000_000,),dict(candidate_count=4,iterations=1,elite_count=2))
    assert result['best_candidate_ids']==[0,0,0]
    assert result['failed_candidate_count']==[0,0,0]
    assert result['elite_candidate_ids']==[[[0,3]],[[0,2]],[[0,2]]]
    assert result['requires_closed_loop_solver_audit'] is True


def test_undefined_bank_discrepancy_cannot_be_hidden_by_another_finite_bank():
    from poc.confirmation_solver import maximum_solver_cost_shift
    banks=[dict(differences=[dict(maximum_absolute_cost_difference=v) for v in values])
           for values in ([.001,.0001],[None,None],[.002,.0002])]
    assert maximum_solver_cost_shift(banks) is None
    banks[1]['differences']=[dict(maximum_absolute_cost_difference=0.)]*2
    assert maximum_solver_cost_shift(banks)==.002


def test_solver_producer_manifest_passes_real_dependency_verification(tmp_path,monkeypatch):
    """Exercise the producer-to-validator boundary that the live audit rejected."""
    from types import SimpleNamespace
    import poc.confirmation_solver as module
    from poc.control_jobs import SOURCES
    from poc.contracts import verify_run
    from poc.io import atomic_json,sha256_file

    fingerprint={key:'fixture' for key in ('python','architecture','os','kernel','core_count',
        'torch_configuration','torch_threads','torch_interop_threads','deterministic_algorithms')}
    fingerprint['versions']={}
    atomic_json(tmp_path/'locks/environment.json',fingerprint)
    atomic_json(tmp_path/'locks/wheels.json',{'wheels':[]})
    design=read_json(Path('config/control_confirmation_solver_audit.json'))
    atomic_json(tmp_path/'config/control_confirmation_solver_audit.json',design)
    atomic_json(tmp_path/'config/control_confirmation_solver_inputs.json',{'fixture':True})
    for name in set(SOURCES)|set(module.AUDIT_SOURCES):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Metadata-boundary fixture\n')
    folder=tmp_path/'original_episode';request=saved_zero_bank(folder)
    model_path=tmp_path/'selected_model/manifest.json'
    atomic_json(model_path,{'fixture':True})
    atomic_json(model_path.parent/'normalization.json',{'fixture':True})
    (model_path.parent/'weights.pt').write_bytes(b'fixture')
    request.update(model_manifest=str(model_path.relative_to(tmp_path)),
        manifest_sha256=sha256_file(model_path))
    atomic_json(folder/'request.json',request)
    for name in ('summary.json','true_trajectories.npz','action_segments.jsonl'):
        (folder/name).write_text('{}\n')
    atomic_json(folder/'manifest.json',{'fixture':True})
    job_path='config/control_jobs/fixture_solver_parent.json'
    atomic_json(tmp_path/job_path,request)
    entry=dict(cell_id='solver-parent-fixture',manifest='original_episode/manifest.json',
               summary={'target':[0.,0.]})
    monkeypatch.setattr('poc.control_models.load_selected',lambda *args,**kwargs:unit_mass_node())
    monkeypatch.setattr('poc.execution.subprocess.run',lambda argv,**kwargs:
        SimpleNamespace(stdout='fixture' if argv[1]=='rev-parse' else '',returncode=0))

    episode=module._episode_audit(tmp_path,entry,{'path':job_path},design)
    manifest=read_json(tmp_path/episode['manifest'])
    verify_run(tmp_path,manifest)
    assert episode['summary']['cell_id']==entry['cell_id']
    assert episode['summary']['n_local_probes']==40
    assert episode['summary']['n_candidate_banks']==3
