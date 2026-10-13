"""Finite-bank scoring clocks, independent truth and complete failure outcomes."""
from poc.calibration_records import read_records
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from poc.calibration_models import build_model
from poc.calibration_study import prepare, open_store, collect_session, prospective_recipe
from poc.calibration_training import fit_model, nominal_reference, load_selected
from poc.models.common import Scales, prediction
from poc.envs import Plant

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert importlib.util.find_spec('poc.calibration_decisions'), 'Decision stage is missing'
    from poc import calibration_decisions
    return calibration_decisions


class CurvedMap(torch.nn.Module):
    """Non-semigroup map: sequential fine queries change its physical endpoint."""
    def predict(self, state, command, h_s):
        h = torch.as_tensor(h_s).reshape(-1, 1)
        out = state.clone()
        out[:, :2] += h**2 + command*h
        return prediction(out, 2, 2)


@pytest.mark.parametrize('step,calls', [(5_000_000, 48), (10_000_000, 24), (20_000_000, 12), (40_000_000, 6)])
def test_auxiliary_clock_preserves_main_map_and_counts_every_query(step, calls):
    m = module()
    bank = np.zeros((3, 6, 2))
    result = m.score_candidates(CurvedMap(), torch.zeros(6, dtype=torch.float64), np.zeros(2), bank, query_step_ns=step)
    expected_endpoint = 6 * (40_000_000 // step) * (step / 1e9)**2
    np.testing.assert_allclose(result['anchors'][:, -1, :2], expected_endpoint)
    q = np.arange(1, 7) * (40_000_000 // step) * (step / 1e9)**2
    expected_cost = .04 * 2 * np.sum(q**2) + 5 * 2 * q[-1]**2
    np.testing.assert_allclose(result['costs'], expected_cost)
    assert result['selected_index'] == 0
    assert result['work']['prediction_batch_calls'] == calls
    assert result['work']['scalar_candidate_evaluations'] == 48*3
    assert result['work']['auxiliary_scalar_predictions'] == (48-calls)*3
    assert result['work']['network_calls'] == 2*calls
    assert result['work']['state_feasibility_checks'] == 49*3
    assert result['samples'].shape == (3, 49, 6)
    for tick in range(1,49):
        start_step = (tick-1) // (step//5_000_000)
        offset_s = ((tick-1) % (step//5_000_000)+1)*.005
        np.testing.assert_allclose(result['samples'][:,tick,:2], start_step*(step/1e9)**2+offset_s**2)
    for query in result['query_log']:
        assert query['offsets_ns'] == list(range(5_000_000, step+1, 5_000_000))
        assert query['endpoint_ns'] == query['start_ns'] + step
        assert query['batch_size'] == 3 * (step // 5_000_000)
    assert result['wall_s'] >= 0 and result['cpu_s'] >= 0


class BetweenAnchorMap(torch.nn.Module):
    def predict(self, state, command, h_s):
        h = torch.as_tensor(h_s).reshape(-1)
        out = state.clone()
        out[:, 0] = torch.where(h == .005, 1.6, 0.)
        return prediction(out, 0)


def test_between_cost_anchor_and_initial_violations_make_no_selection():
    m = module()
    result = m.score_candidates(BetweenAnchorMap(), torch.zeros(6, dtype=torch.float64), np.zeros(2), np.zeros((2,6,2)), query_step_ns=40_000_000)
    assert np.all(result['costs'] == 0) and not result['sampled_feasible'].any()
    assert result['selected_index'] is None and result['all_predicted_infeasible']
    result = m.score_candidates(CurvedMap(), torch.tensor([1.51,0.,0.,0.,0.,0.], dtype=torch.float64), np.zeros(2), np.zeros((2,6,2)), query_step_ns=40_000_000)
    assert not result['sampled_feasible'].any()
    assert not result['initial_feasible']


@pytest.mark.parametrize('failure', ['exception', 'nonfinite', 'status'])
def test_predictor_failures_preserve_all_candidates(failure):
    m = module()
    class Failed(torch.nn.Module):
        def predict(self, state, command, h_s):
            if failure == 'exception':
                raise RuntimeError('intentional unit failure')
            out = state.clone()
            out[:, 0] = float('nan') if failure == 'nonfinite' else 2000.
            return prediction(out, 4, 4)
    result = m.score_candidates(Failed(), torch.zeros(6, dtype=torch.float64), np.zeros(2), np.zeros((256,6,2)), query_step_ns=20_000_000)
    assert len(result['statuses']) == len(result['costs']) == 256
    assert all(s != 'ok' for s in result['statuses'])
    assert np.isinf(result['costs']).all() and not result['sampled_feasible'].any()
    assert result['selected_index'] is None
    assert result['work']['prediction_batch_calls'] >= 1


@pytest.mark.parametrize('model_id', ['affine_flow', 'affine_rk4', 'node_structured', 'node_quadrature', 'node_actuator_aware', 'physics_residual', 'physics_nominal'])
def test_own_model_cost_agrees_with_existing_fixed_anchor_arithmetic(model_id):
    m = module()
    scales = Scales(*(torch.ones(2) for _ in range(3)))
    model = build_model(model_id, scales, 19).double().eval()
    bank = np.array([np.zeros((6,2)), np.full((6,2), .2)])
    state = torch.tensor([.1,-.1,.1,.2,.2,-.1], dtype=torch.float64)
    target = np.array([.2,-.1])
    out = m.score_candidates(model, state, target, bank, query_step_ns=40_000_000)
    from poc.stateless_mpc import rollout_candidates
    if model_id == 'physics_nominal':
        x = state[None].repeat(2,1)
        physical_anchors = []
        with torch.no_grad():
            for k in range(6):
                x = model.predict(x, torch.from_numpy(bank[:,k]), .04).next_state
                physical_anchors.append(x.numpy().copy())
        anchors = np.stack(physical_anchors,axis=1)
        error = anchors[:,:,:2]-target
        costs = .04*np.sum(error**2,axis=(1,2))+.002*np.sum(anchors[:,:,2:4]**2,axis=(1,2))+.00004*np.sum(bank**2,axis=(1,2))+5*np.sum(error[:,-1]**2,axis=1)
    else:
        old = rollout_candidates(model, state, target, None, bank, [40_000_000], 40_000_000)
        costs, anchors = old['costs'], old['anchors']
    np.testing.assert_allclose(out['costs'], costs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out['anchors'], anchors, rtol=1e-12, atol=1e-12)


def test_independent_truth_regret_unsafe_and_false_feasibility_counts():
    m = module()
    result = m.decision_outcome({'costs':np.array([0.,1.,2.]), 'sampled_feasible':np.array([True,False,True]), 'selected_index':0, 'statuses':['ok']*3}, np.array([3.,1.,2.]), np.array([True]*3), np.zeros(3))
    assert result['regret'] == 2 and result['regret_lower'] == result['regret_upper'] == 2
    assert result['false_infeasible_count'] == 1 and result['false_feasible_count'] == 0
    unsafe = m.decision_outcome({'costs':np.zeros(3), 'sampled_feasible':np.ones(3, bool), 'selected_index':0, 'statuses':['ok']*3}, np.array([3.,1.,2.]), np.array([False,True,True]), np.zeros(3))
    assert unsafe['selected_unsafe'] and unsafe['decision_failure'] and unsafe['regret'] is None
    assert unsafe['false_feasible_count'] == 1
    empty = m.decision_outcome({'costs':np.zeros(3), 'sampled_feasible':np.zeros(3, bool), 'selected_index':None, 'statuses':['ok']*3}, np.array([3.,1.,2.]), np.zeros(3, bool), np.zeros(3))
    assert empty['no_true_feasible_bank'] and empty['no_selection'] and empty['decision_failure']


def test_truth_ties_remain_tied_and_regret_has_numerical_interval():
    m = module()
    result = m.decision_outcome({'costs':np.zeros(3), 'sampled_feasible':np.ones(3, bool), 'selected_index':2, 'statuses':['ok']*3}, np.array([1.,1.,2.]), np.ones(3, bool), np.array([.01,.01,.02]))
    assert result['truth_best_indices'] == [0,1]
    assert result['truth_ranking_status'] == 'unresolved'
    assert result['regret'] == 1
    assert result['regret_lower'] == pytest.approx(.97)
    assert result['regret_upper'] == pytest.approx(1.03)


@pytest.fixture
def evaluated(tmp_path):
    module()
    from poc.calibration_predictions import prepare_evaluation
    out = tmp_path / 'disposable-unit-decisions'
    prepare(ROOT, out, unit_fixture={'segment_count':5, 'max_epochs_per_rate':5,
        'evaluation_trajectory_count':2, 'evaluation_anchor_times_ns':[360_000_000,1_200_000_000],
        'decision_candidate_count':4})
    store = open_store(ROOT, out)
    return store, prepare_evaluation(store, root=ROOT)


def arrays(store, receipt, name):
    folder, _ = store.read(receipt)
    with np.load(folder / name, allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def test_bank_uses_actual_primary_anchor_and_shared_observed_pd(evaluated):
    m = module(); store, receipt = evaluated
    data = arrays(store, receipt, 'evaluation.npz')
    assert m.decision_state_ids(data, store.recipe) == [0,3]
    bank = m.candidate_bank(receipt, 3, store)
    assert bank.shape == (4,6,2) and np.abs(bank).max() <= 1.5
    assert np.all(bank[0] == 0)
    pd = np.clip(-data['state'][3,:2] - .25*data['state'][3,2:4], -1.5,1.5)
    np.testing.assert_array_equal(bank[1], np.tile(pd, (6,1)))
    assert bank.tobytes() == m.candidate_bank(receipt, 3, store).tobytes()
    with pytest.raises(ValueError):
        m.candidate_bank(receipt, 5, store)
    recipe = prospective_recipe()
    assert recipe['decisions']['candidate_count'] == 256
    assert recipe['decisions']['bank_channel'] not in [0,1,2,3]
    synthetic = {'trajectory_id':np.repeat(np.arange(64),3), 'primary':np.tile([False,True,True],64),
                 'time_ns':np.tile([0,400_000_000,320_000_000],64)}
    assert m.decision_state_ids(synthetic, recipe) == list(range(2,192,3))


def test_truth_replays_every_candidate_and_keeps_dense_violations(evaluated):
    m = module(); store, evaluation = evaluated
    receipt = m.truth_candidates(store, evaluation, root=ROOT)
    assert receipt == m.truth_candidates(store, evaluation, root=ROOT)
    data = arrays(store, evaluation, 'evaluation.npz')
    truth = m.load_decision_truth(store, evaluation, receipt)
    assert truth['cost'].shape == (2,4) and truth['samples'].shape == (2,4,49,6)
    assert truth['state_id'].tolist() == [0,3]
    assert truth['dense_max_abs'].shape == (2,4,4)
    plant = Plant('lagged',250_000,ROOT/'assets')
    plant.reset(data['state'][0]); bank = m.candidate_bank(evaluation,0,store)
    endpoints = np.stack([plant.advance(command,40_000_000)['state'] for command in bank[2]])
    np.testing.assert_array_equal(truth['base_anchors'][0,2], endpoints)
    error = endpoints[:,:2] - data['target'][0]
    expected = .04*np.sum(error**2)+.002*np.sum(endpoints[:,2:4]**2)+.00004*np.sum(bank[2]**2)+5*np.sum(error[-1]**2)
    assert truth['base_cost'][0,2] == pytest.approx(expected, abs=1e-14)
    _, manifest = store.read(receipt)
    assert manifest['summary']['model_independent']
    assert set(manifest['request']).isdisjoint({'fit_receipt','model_id','budget_s','initialization_id'})
    assert (truth['envelope'] >= 0).all()
    assert truth['dense_violation_count'].shape == (2,4)


def test_measurement_and_failed_fit_keep_complete_coordinates_and_invalid_evidence(evaluated, monkeypatch):
    m = module(); store, evaluation = evaluated
    truth = m.truth_candidates(store, evaluation, root=ROOT)
    collected = collect_session(0, root=ROOT, output_root=store.output_root)
    nominal = nominal_reference(store, collected,5)
    receipt = m.measure_decisions(store, nominal,evaluation,truth)
    folder, manifest = store.read(receipt)
    rows = read_records(folder/'rows.jsonl.gz')
    assert len(rows) == 2*4*4
    assert {(r['state_id'],r['query_step_ns'],r['candidate_id']) for r in rows} == {(s,h,c) for s in [0,3] for h in [5_000_000,10_000_000,20_000_000,40_000_000] for c in range(4)}
    assert len(manifest['summary']['outcomes']) == 8
    assert all(r['comparison'] == 'unfitted_reference' for r in rows)
    from poc import calibration_training
    def fail(*_):
        raise RuntimeError('unit intentionally failed fit')
    monkeypatch.setattr(calibration_training.CalibrationTrainingSession, 'advance_updates',fail)
    fit = fit_model(store,collected,'affine_flow',5,0)
    folder, failed = store.read(m.measure_decisions(store,fit,evaluation,truth))
    failed_rows = read_records(folder/'rows.jsonl.gz')
    assert len(failed_rows) == 32 and all(r['status'] == 'fit_failed' for r in failed_rows)
    assert all(o['no_selection'] and o['decision_failure'] and o['regret'] is None for o in failed['summary']['outcomes'])
    with pytest.raises(ValueError):
        m.measure_decisions(store,{**nominal,'manifest_sha256':'invalid'},evaluation,truth)
    monkeypatch.setattr(calibration_training,'load_selected', lambda *_: (_ for _ in ()).throw(ValueError('invalid checkpoint')))
    with pytest.raises(ValueError,match='invalid checkpoint'):
        m.measure_decisions(store,nominal,evaluation,truth)


def test_cli_decisions_fresh_process_reproduces_real_own_solver(evaluated):
    m = module(); store,evaluation = evaluated
    collected = collect_session(0,root=ROOT,output_root=store.output_root)
    fit = fit_model(store,collected,'affine_rk4',5,0)
    truth = m.truth_candidates(store,evaluation,root=ROOT)
    expected = m.measure_decisions(store,fit,evaluation,truth)
    result = subprocess.run([sys.executable,'-m','poc.calibration_study','decisions','--root',str(ROOT),'--output-root',str(store.output_root),'--model','affine_rk4','--budget','5','--session','0','--initialization','0'],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode == 0,result.stderr
    assert json.loads(result.stdout)['decision_receipts'] == [expected]
    folder, manifest = store.read(expected)
    rows = read_records(folder/'rows.jsonl.gz')
    assert manifest['summary']['canonical_tensor_sha256'] == store.read(fit)[1]['summary']['canonical_tensor_sha256']
    assert all(r['own_solver_trained'] and r['propagator'] == 'affine_rk4' for r in rows)
    data = arrays(store,evaluation,'evaluation.npz')
    predictor = load_selected(store,fit)
    for s in [0,3]:
        scored = m.score_candidates(predictor,torch.from_numpy(data['state'][s]),data['target'][int(data['trajectory_id'][s])],m.candidate_bank(evaluation,s,store),query_step_ns=40_000_000)
        recorded = [r['predicted_cost'] for r in rows if r['state_id']==s and r['query_step_ns']==40_000_000]
        np.testing.assert_array_equal(recorded,scored['costs'])


def test_partial_nonfinite_candidate_does_not_remove_other_candidates():
    m = module()
    class Partial(torch.nn.Module):
        def predict(self,state,command,h_s):
            out = state.clone()
            out[command[:,0] > .1,0] = float('nan')
            return prediction(out,2,2)
    bank = np.zeros((3,6,2)); bank[1] = .2
    result = m.score_candidates(Partial(),torch.zeros(6,dtype=torch.float64),np.zeros(2),bank,query_step_ns=40_000_000)
    assert result['statuses'] == ['ok','nonfinite_prediction','ok']
    assert result['sampled_feasible'].tolist() == [True,False,True]
    assert result['selected_index'] == 0
    assert np.isinf(result['costs'][1]) and result['costs'][0] == result['costs'][2] == 0


def test_dense_truth_violation_can_be_absent_from_sampled_feasibility():
    m = module()
    class DensePlant:
        def reset(self,state):
            self.time = 0
        def advance(self,command,duration_ns,log_every_step):
            times = self.time + np.array([250_000,5_000_000])
            self.time += duration_ns
            trace = np.zeros((2,6)); trace[0,0] = 1.6
            return {'state':np.zeros(6),'trajectory':{'time_ns':times,'state':trace}}
    result = m._replay(DensePlant(),np.zeros(6),np.zeros(2),np.zeros((6,2)))
    assert result['sampled_feasible'] and result['cost'] == 0
    assert result['dense_violation_count'] == 48
    assert result['dense_first_violation_ns'] == 250_000
    assert result['dense_max_abs'][0] == 1.6


def test_truth_refinement_is_bounded_and_keeps_exact_ties(evaluated,monkeypatch):
    m = module(); store,evaluation = evaluated
    # Numerical replay fixture is deliberately identical at every resolution;
    # real receipt production/coordinate coverage/refinement logic remain active.
    def identical(plant,state,target,bank):
        return {'cost':1.,'samples':np.zeros((49,6)), 'dense_max_abs':np.zeros(4),
                'dense_violation_count':0,'dense_first_violation_ns':-1,'sampled_feasible':True}
    monkeypatch.setattr(m,'_replay',identical)
    raw = m.load_decision_truth(store,evaluation,m.truth_candidates(store,evaluation,root=ROOT))
    assert np.all(raw['cost']==1) and np.all(raw['envelope']==0)
    assert np.all(raw['resolution_ns']==31_250)
    assert np.all(raw['refinement_status']=='plateau_unresolved')
    predicted = {'selected_index':0,'costs':np.zeros(4),'sampled_feasible':np.ones(4,bool),'statuses':['ok']*4}
    outcome = m.decision_outcome(predicted,raw['cost'][0],raw['sampled_feasible'][0],raw['envelope'][0])
    assert outcome['truth_best_indices'] == [0,1,2,3]
    assert outcome['truth_ranking_status']=='unresolved'
    assert outcome['regret_lower']==outcome['regret_upper']==0


def test_full_default_failed_fit_retains_64_times_256_times_four_rows(tmp_path,monkeypatch):
    m = module()
    from poc.calibration_predictions import prepare_evaluation
    from poc import calibration_training
    out = tmp_path/'disposable-full-coverage-unit'
    prepare(ROOT,out,unit_fixture={'segment_count':5,'max_epochs_per_rate':5,
        'evaluation_anchor_times_ns':[320_000_000]})
    store = open_store(ROOT,out)
    evaluation = prepare_evaluation(store,root=ROOT)
    def identical(plant,state,target,bank):
        return {'cost':1.,'samples':np.zeros((49,6)), 'dense_max_abs':np.zeros(4),
                'dense_violation_count':0,'dense_first_violation_ns':-1,'sampled_feasible':True}
    monkeypatch.setattr(m,'_replay',identical)
    truth = m.truth_candidates(store,evaluation,root=ROOT)
    collected = collect_session(0,root=ROOT,output_root=out)
    def fail(*_):
        raise RuntimeError('intentional unit failed fit')
    monkeypatch.setattr(calibration_training.CalibrationTrainingSession,'advance_updates',fail)
    fit = fit_model(store,collected,'affine_flow',5,0)
    folder,manifest = store.read(m.measure_decisions(store,fit,evaluation,truth))
    rows = read_records(folder/'rows.jsonl.gz')
    assert len(rows)==64*256*4
    assert len(manifest['summary']['outcomes'])==64*4
    assert {r['trajectory_id'] for r in rows}==set(range(64))
    assert {r['candidate_id'] for r in rows}==set(range(256))
    assert all(r['status']=='fit_failed' and r['predicted_cost'] is None for r in rows)
    assert all(o['decision_failure'] and o['no_selection'] and o['regret'] is None for o in manifest['summary']['outcomes'])


def test_refinement_exhaustion_preserves_numerical_envelopes_and_unresolved_status(evaluated,monkeypatch):
    m = module(); store,evaluation = evaluated
    def close_costs(plant,state,target,bank):
        # Resolution error drops strictly while the candidate separation stays
        # far below its numerical envelope. All five levels must be retained.
        return {'cost':1.+plant.physics_step_ns**2/1e12+float(bank[0,0])*1e-10,
                'samples':np.zeros((49,6)), 'dense_max_abs':np.zeros(4),
                'dense_violation_count':0,'dense_first_violation_ns':-1,'sampled_feasible':True}
    monkeypatch.setattr(m,'_replay',close_costs)
    raw = m.load_decision_truth(store,evaluation,m.truth_candidates(store,evaluation,root=ROOT))
    assert np.all(raw['resolution_ns']==15_625)
    assert np.all(raw['refinement_status']=='exhausted_unresolved')
    assert np.isfinite(raw['cost_by_resolution']).all()
    expected = np.maximum(np.abs(raw['cost_by_resolution'][:,:,4]-raw['cost_by_resolution'][:,:,3]),
                          np.abs(raw['cost_by_resolution'][:,:,3]-raw['cost_by_resolution'][:,:,2]))
    np.testing.assert_array_equal(raw['envelope'],expected)
    assert np.all(raw['envelope']>0)


def test_command_constraint_and_no_selection_with_truly_feasible_bank():
    m = module()
    bank = np.full((2,6,2),2.1)
    out = m.score_candidates(CurvedMap(),torch.zeros(6,dtype=torch.float64),np.zeros(2),bank,query_step_ns=40_000_000)
    assert not out['sampled_feasible'].any() and out['selected_index'] is None
    assert np.isfinite(out['costs']).all()
    outcome = m.decision_outcome(out,np.ones(2),np.ones(2,bool),np.zeros(2))
    assert outcome['no_selection'] and outcome['decision_failure'] and not outcome['no_true_feasible_bank']
    assert outcome['regret'] is None and outcome['false_infeasible_count']==2


def test_shared_bank_survives_actual_session_budget_model_and_initialization_identities(tmp_path):
    m = module()
    from poc.calibration_predictions import prepare_evaluation
    out = tmp_path/'disposable-paired-bank-unit'
    prepare(ROOT,out,'development',unit_fixture={'segment_count':10,'budgets_s':[5,10],
        'max_epochs_per_rate':5,'evaluation_trajectory_count':2,
        'evaluation_anchor_times_ns':[320_000_000], 'decision_candidate_count':4})
    store = open_store(ROOT,out,'development')
    evaluation = prepare_evaluation(store,root=ROOT)
    initial = [m.candidate_bank(evaluation,s,store).tobytes() for s in [0,2]]
    receipts = []
    for session in [0,4]:
        collected = collect_session(session,root=ROOT,output_root=out,matrix='development')
        receipts.extend(nominal_reference(store,collected,b) for b in [5,10])
        if session == 0:
            receipts.extend(fit_model(store,collected,'affine_rk4',b,i) for b,i in [(5,0),(10,1)])
    truth = m.truth_candidates(store,evaluation,root=ROOT)
    for receipt in receipts:
        result = m.measure_decisions(store,receipt,evaluation,truth)
        request = store.read(result)[1]['request']
        assert request['evaluation_receipt'] == evaluation and request['truth_receipt'] == truth
        assert [m.candidate_bank(evaluation,s,store).tobytes() for s in [0,2]] == initial


def test_wide_envelope_on_nonrunnerup_prevents_unique_truth_ranking():
    m = module()
    predicted = {'costs':np.zeros(3),'sampled_feasible':np.ones(3,bool),'selected_index':0,'statuses':['ok']*3}
    outcome = m.decision_outcome(predicted,np.array([1.,2.,3.]),np.ones(3,bool),np.array([0.,0.,2.]))
    assert outcome['truth_best_indices']==[0,2]
    assert outcome['truth_ranking_status']=='unresolved'


@pytest.mark.parametrize('persistent', [False,True])
def test_boundary_feasibility_refines_and_propagates_to_report(evaluated,monkeypatch,persistent):
    m=module();store,evaluation=evaluated
    def boundary(plant,state,target,bank):
        zero=not bank.any(); h=plant.physics_step_ns
        safe=not zero or h not in ({125_000,31_250} if persistent else {125_000})
        samples=np.zeros((49,6)); samples[:,0]=1.5+(-1e-9 if safe else 1e-9)
        return {'cost':1. if zero else 2.+float(np.sum(bank**2)), 'samples':samples,
                'dense_max_abs':np.max(np.abs(samples[:,:4]),axis=0),
                'dense_violation_count':0 if safe else 49,'dense_first_violation_ns':-1 if safe else 0,
                'sampled_feasible':bool(m._safe(samples).all())}
    monkeypatch.setattr(m,'_replay',boundary)
    receipt=m.truth_candidates(store,evaluation,root=ROOT)
    raw=m.load_decision_truth(store,evaluation,receipt)
    assert raw['resolution_ns'][0,0]==(15_625 if persistent else 31_250)
    assert raw['feasibility_disagreement'][0,0]
    assert raw['feasibility_status'][0,0]==('unresolved' if persistent else 'resolved')
    assert raw['feasibility_by_resolution'][0,0].tolist()==([1,0,1,0,1] if persistent else [1,0,1,1,-1])
    collected=collect_session(0,root=ROOT,output_root=store.output_root)
    nominal=nominal_reference(store,collected,5)
    result=m.measure_decisions(store,nominal,evaluation,receipt)
    from poc import calibration_report as report
    folder,manifest=store.read(result)
    outcomes=report._decision_records(store,folder,manifest)
    assert all(o['truth_feasibility_status']==('unresolved' if persistent else 'resolved') for o in outcomes)
    if persistent:
        assert all(o['regret'] is None and o['truth_ranking_status']=='unresolved' for o in outcomes)
        metrics=[{'component':c,'failure_count':0,'conservative_rmse_upper':0.,'conservative_p95_upper':0.} for c in ('q','v')]
        screen=report.configuration_screen(metrics,outcomes)
        assert screen['status']=='unresolved' and screen['truth_feasibility_unresolved']
