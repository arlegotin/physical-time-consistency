"""Signed control reporting waits for numerical validity and preserves mixed evidence."""
import copy
from pathlib import Path

import pytest
from poc.io import read_json
from test_control_utility import private_episodes


def design():
    return read_json(Path('config/control_confirmation_design.json'))


def refinements(rows):
    return [{**copy.deepcopy(row), 'physics_step_ns': 125_000,
             'checkpoint_hash': row.get('checkpoint_hash', 'a'*64),
             'realized_cost': row['realized_cost']+.00001} for row in rows]


def primary():
    return [{**row,'checkpoint_hash':'a'*64} for row in private_episodes()]


def test_numerical_qualification_accepts_resolved_complete_subset_without_spending_a_look():
    from poc.control_result import qualify_numerics
    rows=primary();fine=refinements([r for r in rows if r['episode_id']<5])
    result=qualify_numerics(rows,fine,design(),look_index=1,scope='subset',solver_cells_required=[])
    assert result['state']=='qualified'
    assert result['required_actions']==[]
    assert result['near_boundary_actuators']==[]
    assert result['statistical_look_reserved'] is False
    assert result['scientific_decision']=='not_evaluated'


def test_unresolved_sign_requires_full_physics_coverage_even_if_subset_is_stable():
    from poc.control_result import qualify_numerics
    rows=primary()
    for row in rows:
        if row['actuator']=='direct' and row['model_id']=='power_correct':
            row['realized_cost']+=1.006
    result=qualify_numerics(rows,refinements([r for r in rows if r['episode_id']<5]),
                          design(),look_index=1,scope='subset',solver_cells_required=[])
    assert result['state']=='additional_numerical_evidence_required'
    assert result['near_boundary_actuators']==['direct']
    assert {'branch':'physics_full','actuators':['direct']} in result['required_actions']


def test_full_stable_physics_can_qualify_an_unresolved_statistical_sign():
    from poc.control_result import qualify_numerics
    rows=primary()
    for row in rows:
        if row['model_id']=='power_correct':row['realized_cost']+=1.006
    result=qualify_numerics(rows,refinements(rows),design(),look_index=1,scope='full',solver_cells_required=[])
    assert result['state']=='qualified'
    assert result['near_boundary_actuators']==['direct','lagged']
    assert result['scientific_decision']=='not_evaluated'


def test_material_full_physics_change_requires_further_convergence_instead_of_a_sign():
    from poc.control_result import qualify_numerics
    rows=primary();fine=refinements(rows)
    for row in fine:
        if row['actuator']=='lagged':row['realized_cost']*=1.1
    result=qualify_numerics(rows,fine,design(),look_index=1,scope='full',solver_cells_required=[])
    assert result['state']=='additional_numerical_evidence_required'
    assert {'branch':'physics_further_refinement','actuators':['lagged']} in result['required_actions']


def test_changed_solver_ranking_blocks_numerical_qualification():
    from poc.control_result import qualify_numerics
    rows=primary();fine=refinements([r for r in rows if r['episode_id']<5])
    result=qualify_numerics(rows,fine,design(),look_index=1,scope='subset',
                          solver_cells_required=['node_generic__direct__seed0__episode0'])
    assert result['state']=='additional_numerical_evidence_required'
    assert {'branch':'solver_closed_loop','cells':['node_generic__direct__seed0__episode0']} in result['required_actions']


def test_verified_closed_loop_solver_expands_envelope_without_replacing_primary_costs():
    from poc.control_result import qualify_numerics
    rows=primary();fine=refinements(rows)
    solver=dict(state='qualified',parent_cells=['declared-node-parent'],required_actions=[],
        envelopes={'utility_control/direct/node_generic/h40000000':dict(lower=.03,upper=.03),
                   'utility_control/lagged/node_generic/h40000000':dict(lower=.04,upper=.04)},
        evidence_verified=True,validation=dict(state='complete',n_fits=10,
            material_nominal_fit_change=False,continuous_field_claim_requires_interpretation=False),
        scope='evaluation_only_same_learned_field')
    result=qualify_numerics(rows,fine,design(),look_index=1,scope='full',
        solver_cells_required=['declared-node-parent'],solver_resolution=solver)
    assert result['state']=='qualified' and result['required_actions']==[]
    assert result['envelopes']['utility_control/direct/node_generic/h40000000']['lower']==pytest.approx(.03002)
    assert result['envelopes']['utility_control/lagged/node_generic/h40000000']['upper']==pytest.approx(.04002)
    assert result['envelopes']['utility_control/direct/mlp_dt/h5000000']['upper']==pytest.approx(.00002)
    assert result['solver_resolution']['scope']=='evaluation_only_same_learned_field'
    assert result['scientific_decision']=='not_evaluated'


@pytest.mark.parametrize('mutation',['partial','coverage','hash','missing_validation','convergence'])
def test_incomplete_solver_resolution_cannot_clear_numerical_prerequisite(mutation):
    from poc.control_result import qualify_numerics
    solver=dict(state='qualified',parent_cells=['declared-node-parent'],required_actions=[],
        envelopes={'utility_control/direct/node_generic/h40000000':dict(lower=.03,upper=.03),
                   'utility_control/lagged/node_generic/h40000000':dict(lower=.04,upper=.04)},
        evidence_verified=True,validation=dict(state='complete',n_fits=10,
            material_nominal_fit_change=False,continuous_field_claim_requires_interpretation=False),
        scope='evaluation_only_same_learned_field')
    if mutation=='partial':solver['state']='running'
    elif mutation=='coverage':solver['parent_cells']=[]
    elif mutation=='hash':solver['evidence_verified']=False
    elif mutation=='missing_validation':solver.pop('validation')
    else:
        solver['state']='additional_solver_evidence_required'
        solver['required_actions']=[dict(branch='solver_resolution',cells=['declared-node-parent'],next_substeps=8)]
    result=qualify_numerics(primary(),refinements(primary()),design(),look_index=1,scope='full',
        solver_cells_required=['declared-node-parent'],solver_resolution=solver)
    assert result['state']=='additional_numerical_evidence_required'
    assert result['required_actions']
    assert result['statistical_look_reserved'] is False


def test_solver_qualified_reporting_keeps_outward_envelope_and_original_costs(tmp_path):
    from poc.control_result import evaluate_evidence
    from poc.io import sha256_file
    evidence=staged_evidence(tmp_path)
    evidence.update(scope='full',refined=refinements(evidence['primary']),solver_cells_required=['declared-node-parent'])
    evidence['solver_resolution']=dict(state='qualified',parent_cells=['declared-node-parent'],required_actions=[],
        envelopes={'utility_control/direct/node_generic/h40000000':dict(lower=.03,upper=.03),
                   'utility_control/lagged/node_generic/h40000000':dict(lower=.04,upper=.04)},
        evidence_verified=True,validation=dict(state='complete',n_fits=10,
            material_nominal_fit_change=True,continuous_field_claim_requires_interpretation=True),
        scope='evaluation_only_same_learned_field')
    original_costs=copy.deepcopy(evidence['seed_costs'])
    result=evaluate_evidence(tmp_path,evidence,report=True)
    assert result['status']=='limited_scope_results'
    assert result['seed_costs']==original_costs
    numerical=read_json(tmp_path/'reports/control_confirmation_numerical.json')
    assert numerical['envelopes']['utility_control/direct/node_generic/h40000000']['lower']==pytest.approx(.03002)
    assert result['solver_resolution']['validation']['material_nominal_fit_change'] is True
    text=(tmp_path/'reports/control_confirmation_result.md').read_text()
    assert 'discretization-specific' in text
    before=sha256_file(tmp_path/'reports/control_confirmation_result.json')
    repeated=evaluate_evidence(tmp_path,evidence,report=True)
    assert repeated==result and sha256_file(tmp_path/'reports/control_confirmation_result.json')==before
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json')))==1


def test_missing_physics_coverage_has_no_numerical_default():
    from poc.control_result import qualify_numerics
    with pytest.raises(ValueError,match='Complete'):
        qualify_numerics(primary(),[],design(),look_index=1,scope='subset',solver_cells_required=[])


def test_report_preserves_gains_losses_unresolved_conditions_and_failure_counts():
    from poc.control_result import render_result
    from poc.control_utility import seed_costs,utility_bounds
    d=design();costs=seed_costs(private_episodes(),d)
    envelopes={c['criterion_id']:dict(lower=.001,upper=.001) for c in d['criteria']}
    result=utility_bounds(costs,d,1,envelopes)
    result['criteria'][0]['status']='higher_cost'
    result['criteria'][1]['status']='unresolved_influence'
    text=render_result(result,costs,dict(physics_fit_episodes=200,zero_command_episodes=200,sim_oracle_episodes=50))
    assert '| direct | mlp_dt | 40 | higher_cost |' in text
    assert '| direct | mlp_dt | 5 | unresolved_influence |' in text
    assert '5 independent training seeds' in text
    assert '1,200' in text
    assert 'Fallback intervals: 60' in text
    assert 'Successful episodes: 60' in text
    assert 'Original PTCO all-claims PASS is not established' in text
    assert 'physics_fit: 200' in text


@pytest.mark.parametrize('mutation',['missing','duplicate','unknown_status'])
def test_result_renderer_rejects_partial_or_mislabelled_results(mutation):
    from poc.control_result import render_result
    from poc.control_utility import seed_costs,utility_bounds
    d=design();costs=seed_costs(private_episodes(),d)
    result=utility_bounds(costs,d,1,{c['criterion_id']:dict(lower=0.,upper=0.) for c in d['criteria']})
    if mutation=='missing':result['criteria'].pop()
    elif mutation=='duplicate':result['criteria'][-1]=copy.deepcopy(result['criteria'][0])
    else:result['criteria'][0]['status']='POC PASS'
    with pytest.raises(ValueError):render_result(result,costs,{})


def staged_evidence(root):
    from poc.io import atomic_json,sha256_file
    from poc.control_utility import seed_costs
    rows=primary();d=design()
    for name in ('config/control_confirmation_cells.json','reports/control_confirmation_batch.json'):
        atomic_json(root/name,{'private_fixture':'No experiment or real frozen checkpoint'})
    atomic_json(root/'config/control_confirmation_design.json',d)
    resources={group:[dict(wall_seconds=1.,cpu_seconds=.5,peak_rss_bytes=100,disk_bytes=50)]
               for group in ('primary','references','refinement','solver')}
    references=[dict(model_id=m,actuator=a,seed=s,episode_id=e,realized_cost=1.)
        for a in ('direct','lagged') for m in ('physics_fit','zero_command','sim_oracle')
        for s in range(5) for e in range(5 if m=='sim_oracle' else 20)]
    names=['config/control_confirmation_cells.json','reports/control_confirmation_batch.json',
           'config/control_confirmation_design.json']
    return dict(design=d,primary=rows,refined=refinements([r for r in rows if r['episode_id']<5]),
        references=references,seed_costs=seed_costs(rows,d),solver_cells_required=[],scope='subset',
        resources=resources,input_hashes={n:sha256_file(root/n) for n in names})


def test_check_ready_does_not_reserve_a_look_and_reporting_is_idempotent(tmp_path,monkeypatch):
    from poc.control_result import evaluate_evidence
    from poc.io import sha256_file,read_json
    e=staged_evidence(tmp_path)
    checked=evaluate_evidence(tmp_path,e,report=False)
    assert checked['state']=='qualified'
    assert not (tmp_path/'decisions/control_utility_looks').exists()
    result=evaluate_evidence(tmp_path,e,report=True)
    assert all(r['status']=='lower_cost' for r in result['criteria'])
    path=tmp_path/'reports/control_confirmation_result.json';before=sha256_file(path)
    repeated=evaluate_evidence(tmp_path,e,report=True)
    assert repeated==result and sha256_file(path)==before
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json')))==1
    assert read_json(tmp_path/'reports/control_confirmation_result.json')['resources']['primary']['wall_seconds']==1.


def test_pending_solver_diagnosis_cannot_reserve_look_or_emit_signed_result(tmp_path):
    from poc.control_result import evaluate_evidence
    e=staged_evidence(tmp_path);e['solver_cells_required']=['changed-node-bank']
    result=evaluate_evidence(tmp_path,e,report=True)
    assert result['state']=='additional_numerical_evidence_required'
    assert not (tmp_path/'decisions/control_utility_looks').exists()
    assert not (tmp_path/'reports/control_confirmation_result.json').exists()


def test_changed_input_is_rejected_before_look_reservation(tmp_path):
    from poc.control_result import evaluate_evidence
    from poc.io import atomic_json
    e=staged_evidence(tmp_path)
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',{'changed':True})
    with pytest.raises(ValueError,match='Changed'):
        evaluate_evidence(tmp_path,e,report=True)
    assert not (tmp_path/'decisions/control_utility_looks').exists()


def test_incomplete_actual_cli_does_not_write_a_scientific_result(tmp_path):
    from poc.control_result import main
    from poc.io import atomic_json
    atomic_json(tmp_path/'reports/control_confirmation_batch.json',dict(state='running'))
    with pytest.raises(ValueError,match='Complete primary'):
        main(['--root',str(tmp_path),'--report'])
    assert not (tmp_path/'reports/control_confirmation_result.json').exists()


@pytest.mark.parametrize('mutation',['references','resources'])
def test_incomplete_reference_or_resource_evidence_cannot_spend_a_look(tmp_path,mutation):
    from poc.control_result import evaluate_evidence
    e=staged_evidence(tmp_path)
    if mutation=='references':e['references'].pop()
    else:e['resources']['primary'][0]['wall_seconds']=float('nan')
    with pytest.raises(ValueError):evaluate_evidence(tmp_path,e,report=True)
    assert not list((tmp_path/'decisions/control_utility_looks').glob('look-*.json'))


def test_changed_declared_evaluation_evidence_keeps_prior_result_and_uses_next_look(tmp_path):
    from poc.control_result import evaluate_evidence
    from poc.io import sha256_file,read_json
    e=staged_evidence(tmp_path)
    first=evaluate_evidence(tmp_path,e,report=True)
    path=tmp_path/'config/independent_numerical_diagnosis.json'
    path.write_text('{"verified_new_diagnostic":true}')
    e['input_hashes']['config/independent_numerical_diagnosis.json']=sha256_file(path)
    second=evaluate_evidence(tmp_path,e,report=True)
    assert second['look_index']==2
    assert read_json(tmp_path/'reports/control_confirmation_look1.json')==first
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json')))==2


def test_full_report_reuses_its_look_after_readonly_subset_requalification(tmp_path):
    from poc.control_result import evaluate_evidence
    from poc.control_utility import seed_costs
    from poc.io import sha256_file
    subset=staged_evidence(tmp_path)
    for row in subset['primary']:
        if row['model_id']=='power_correct':row['realized_cost']+=1.006
    subset['seed_costs']=seed_costs(subset['primary'],subset['design'])
    subset['refined']=refinements([r for r in subset['primary'] if r['episode_id']<5])
    full=copy.deepcopy(subset)
    full.update(scope='full',refined=refinements(full['primary']))
    original=evaluate_evidence(tmp_path,full,report=True)
    path=tmp_path/'reports/control_confirmation_result.json'
    original_hash=sha256_file(path)
    assert evaluate_evidence(tmp_path,subset,report=False)['state']=='additional_numerical_evidence_required'
    repeated=evaluate_evidence(tmp_path,full,report=True)
    assert repeated==original
    assert sha256_file(path)==original_hash
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json')))==1


@pytest.mark.parametrize('mutation',['missing','changed'])
def test_immutable_numerical_archive_corruption_cannot_spend_a_look(tmp_path,mutation):
    from poc.control_result import evaluate_evidence
    from poc.io import sha256_file
    evidence=staged_evidence(tmp_path)
    result=evaluate_evidence(tmp_path,evidence,report=True)
    result_path=tmp_path/'reports/control_confirmation_result.json'
    result_hash=sha256_file(result_path)
    archive=tmp_path/f"decisions/control_utility_numerics/{result['evidence_identity']['numerical_sha256']}.json"
    if mutation=='missing':archive.unlink()
    else:archive.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='immutable utility numerical evidence'):
        evaluate_evidence(tmp_path,evidence,report=True)
    assert sha256_file(result_path)==result_hash
    assert len(list((tmp_path/'decisions/control_utility_looks').glob('look-*.json')))==1


def test_large_saved_bank_cost_shift_requires_solver_branch_even_with_same_winner():
    from poc.control_result import solver_effect_diagnostics
    from poc.control_utility import seed_costs
    from test_confirmation_solver import unit_mass_node
    from poc.confirmation_solver import audit_candidate_bank, SolverResolution
    from poc.mpc import rollout_candidates
    import numpy as np
    import torch
    model=unit_mass_node()
    with torch.no_grad():
        model.network.weight[2,4]=-4.;model.network.weight[3,5]=-4.
    state=torch.zeros(4,dtype=torch.float64)
    knots=np.stack([np.zeros((6,2)),np.ones((6,2))])
    saved=rollout_candidates(SolverResolution(model,1),state,np.zeros(2),None,knots,(40_000_000,),40_000_000)
    bank=audit_candidate_bank(model,state,np.zeros(2),knots,saved['costs'],saved['statuses'],saved['reasons'],0,(40_000_000,))
    assert bank['best_candidate_ids']==[0,0,0]
    assert bank['requires_closed_loop_solver_audit'] is False
    shift=max(d['maximum_absolute_cost_difference'] for d in bank['differences'])
    result=solver_effect_diagnostics(seed_costs(primary(),design()),[
        dict(cell_id='direct-node-40',actuator='direct',model_step_ns=40_000_000,
             maximum_absolute_cost_difference=shift)])
    assert result[0]['requires_closed_loop_solver_audit'] is True
    assert result[0]['criterion_id']=='utility_control/direct/node_generic/h40000000'
    assert result[0]['maximum_absolute_cost_difference']>100


@pytest.mark.parametrize('effect,shift,expected',[(1.,.1,False),(1.,.10001,True),(0.,.001,True),(0.,0.,False)])
def test_solver_cost_diagnostic_uses_absolute_paired_mean_effect(effect,shift,expected):
    from poc.control_result import solver_effect_diagnostics
    from poc.control_utility import seed_costs
    rows=primary()
    for row in rows:
        row['realized_cost']=2.+row['seed']*.01+(effect if row['model_id']=='node_generic' else 0.)
    result=solver_effect_diagnostics(seed_costs(rows,design()),[
        dict(cell_id='private',actuator='direct',model_step_ns=5_000_000,maximum_absolute_cost_difference=shift)])
    assert result[0]['requires_closed_loop_solver_audit'] is expected
    assert result[0]['absolute_mean_episode_effect']==pytest.approx(abs(effect))
