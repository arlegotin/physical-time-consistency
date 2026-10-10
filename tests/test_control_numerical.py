"""Numerical changes must stay paired and cannot silently authorize a sign."""
import copy
import numpy as np
import pytest
from poc.io import read_json
from pathlib import Path


def paired_rows():
    from poc.control_episode import episode_conditions
    primary=[]; refined=[]
    for a in ('direct','lagged'):
        for m in ('mlp_dt','node_generic','power_correct'):
            for s in range(5):
                for h in (40_000_000,5_000_000):
                    for e in range(5):
                        task=episode_conditions(s,e,a,'confirmatory')
                        cost=(2. if m=='power_correct' else 3.)+.01*s+.001*e
                        row=dict(actuator=a,model_id=m,seed=s,episode_id=e,phase='confirmatory',
                            condition='clean',complete=True,physics_step_ns=250_000,
                            actuator_interval_ns=40_000_000,model_partition_ns=[h]*(40_000_000//h),
                            power_mode=None,checkpoint_hash='a'*64,initial_state=task['initial_state'].tolist(),
                            target=task['target'].tolist(),seed_namespaces={'cem':task['cem_seed_entropy'],
                            'disturbance':task['disturbance_seed_entropy']},realized_cost=cost,success=True,
                            fallback_count=0)
                        primary.append(row)
                        refined.append({**copy.deepcopy(row),'physics_step_ns':125_000,'realized_cost':cost+.001})
    return primary,refined


def test_refinement_preserves_pairing_and_reports_empirical_cost_envelopes():
    from poc.control_numerical import compare_refinement
    p,r=paired_rows()
    result=compare_refinement(p,r,read_json(Path('config/control_confirmation_design.json')))
    assert result['audited_episode_pairs']==300
    assert result['full_cells_required']==[]
    assert result['scientific_decision']=='not_evaluated'
    assert len(result['envelopes'])==8
    assert all(e['lower']==pytest.approx(.002) and e['upper']==pytest.approx(.002)
               for e in result['envelopes'].values())
    assert result['envelope_interpretation']=='empirical_discrepancy_not_certified_error_bound'


@pytest.mark.parametrize('mutation',['missing','duplicate','target','initial','checkpoint','namespace','clock','nan'])
def test_changed_workloads_or_incomplete_refinement_cannot_enter_envelopes(mutation):
    from poc.control_numerical import compare_refinement
    p,r=paired_rows()
    if mutation=='missing':r.pop()
    elif mutation=='duplicate':r[-1]=copy.deepcopy(r[0])
    elif mutation=='target':r[0]['target'][0]+=.01
    elif mutation=='initial':r[0]['initial_state'][0]+=.01
    elif mutation=='checkpoint':r[0]['checkpoint_hash']='b'*64
    elif mutation=='namespace':r[0]['seed_namespaces']['cem'][0]=7101
    elif mutation=='clock':r[0]['physics_step_ns']=250_000
    else:r[0]['realized_cost']=float('nan')
    with pytest.raises(ValueError):compare_refinement(p,r,read_json(Path('config/control_confirmation_design.json')))


@pytest.mark.parametrize('mutation',['cost','success','effect'])
def test_material_numerical_change_requires_the_full_affected_plant_cell(mutation):
    from poc.control_numerical import compare_refinement
    p,r=paired_rows()
    if mutation=='cost':
        for row in r:
            if row['actuator']=='direct' and row['model_id']=='power_correct':row['realized_cost']+=.1
    elif mutation=='success':r[0]['success']=False
    else:
        # Each cost changes <2%, but opposing errors change relative reduction >2 points.
        for row in r:
            if row['actuator']=='direct':row['realized_cost']*=.981 if row['model_id']=='mlp_dt' else 1.019
    result=compare_refinement(p,r,read_json(Path('config/control_confirmation_design.json')))
    assert result['full_cells_required']==['direct']
    assert result['qualification']=='expand_full_affected_cells'
    assert result['scientific_decision']=='not_evaluated'


def test_near_boundary_requires_full_cell_even_when_subset_is_stable():
    from poc.control_numerical import compare_refinement
    p,r=paired_rows()
    result=compare_refinement(p,r,read_json(Path('config/control_confirmation_design.json')),near_boundary_actuators=['lagged'])
    assert result['full_cells_required']==['lagged']


def test_full_refinement_remains_required_until_all_twenty_tasks_are_present():
    from poc.control_numerical import compare_refinement
    p,r=paired_rows()
    with pytest.raises(ValueError,match='Complete'):
        compare_refinement(p,r,read_json(Path('config/control_confirmation_design.json')),scope='full')


def test_node_solver_audit_holds_initial_activation_fixed_and_counts_actual_rk4_work():
    import torch
    from poc.models.ode import NeuralODE
    from poc.models.common import Scales
    from poc.analytic import transition
    from poc.control_numerical import solver_probe
    scales=Scales(torch.ones(2,dtype=torch.float64),torch.ones(2,dtype=torch.float64),torch.ones(2,dtype=torch.float64))
    model=NeuralODE('lagged',scales).double()
    # The learned field is replaced by the independent unit-mass fixture, not its solver.
    model.vector_field=lambda qv,command,w:torch.cat((qv[:,2:4],w),dim=1)
    state=torch.tensor([[.2,-.3,.4,-.5,.6,-.7]],dtype=torch.float64)
    command=torch.tensor([[1.1,-1.2]],dtype=torch.float64)
    original=state.clone();u_original=command.clone()
    result=solver_probe(model,state,command,40_000_000)
    assert result['substeps']==[1,2,4]
    assert result['vector_field_calls']==[4,8,16]
    assert torch.equal(state,original) and torch.equal(command,u_original)
    truth=transition(state.numpy()[0],command.numpy()[0],.04,'lagged')
    errors=[np.linalg.norm(np.asarray(x)-truth) for x in result['next_states']]
    assert errors[1]<errors[0]/10 and errors[2]<errors[1]/10
    assert result['command_jacobians'][2][-1][-1]==pytest.approx(1-np.exp(-.5),rel=1e-12)
    assert result['command_jacobians'][2][-2][0]==pytest.approx(1-np.exp(-.5),rel=1e-12)
