"""Disposable timing units; these results are not empirical study evidence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.calibration_models import MODEL_IDS, build_model
from poc.calibration_study import prepare, open_store, collect_session
from poc.calibration_training import nominal_reference
from poc.calibration_predictions import prepare_evaluation
from poc.models.common import Scales

ROOT = Path(__file__).resolve().parents[1]


def module():
    assert importlib.util.find_spec('poc.calibration_timing'), 'Timing stage is missing'
    from poc import calibration_timing
    return calibration_timing


class ManualClock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now


def test_excludes_warmups_rotates_order_and_times_the_complete_operation():
    m = module()
    clock = ManualClock()
    order = []
    def operation(name, elapsed):
        def run():
            order.append(name)
            clock.now += elapsed
            return {'status':'ok', 'work':{'network_calls':1}}
        return run
    cells = [({'model_id':'fast','category':'complete_scorer'},operation('fast',.001)),
             ({'model_id':'slow','category':'complete_scorer'},operation('slow',.25))]
    rows = m.run_rotated_cells(cells, clock=clock, cpu_clock=clock)
    assert len(rows) == 100
    assert order[:4] == ['fast','slow','slow','fast']
    assert order.count('fast') == order.count('slow') == 70
    assert {r['repetition'] for r in rows} == set(range(50))
    summary = m.summarize_timing(rows)
    groups = {g['coordinate']['model_id']:g for g in summary['groups']}
    assert groups['slow']['wall_s']['median'] == pytest.approx(.25)
    assert groups['fast']['wall_s']['p95'] == pytest.approx(.001)
    assert groups['slow']['measured_calls'] == 50
    assert groups['slow']['work_totals']['network_calls'] == 50
    assert groups['slow']['decision_critical']
    with pytest.raises(ValueError, match='Duplicate'):
        m.run_rotated_cells([cells[0],cells[0]],clock=clock,cpu_clock=clock)


@pytest.mark.parametrize('model_id', [*MODEL_IDS,'physics_nominal'])
def test_runtime_cache_preserves_own_exact_map_and_interval_gradients(model_id):
    m = module()
    scales = Scales(*(torch.ones(2,dtype=torch.float64) for _ in range(3)))
    model = build_model(model_id,scales,7).double().eval()
    state = torch.tensor([[.1,-.2,.03,.04,.2,-.1]]*3,dtype=torch.float64)
    command = torch.tensor([[.1,.2]]*3,dtype=torch.float64)
    h = torch.tensor([[.005],[.02],[.04]],dtype=torch.float64)
    with torch.inference_mode():
        expected = model.predict(state,command,h)
    derivative_h = h.clone().requires_grad_()
    original_gradient = torch.autograd.grad(model.predict(state,command,derivative_h).next_state.sum(), derivative_h)[0]
    with m.fixed_coefficient_cache(model) as cache:
        with torch.inference_mode():
            actual = model.predict(state,command,h)
            again = model.predict(state,command,h)
        assert torch.equal(actual.next_state,expected.next_state)
        assert torch.equal(again.next_state,expected.next_state)
        assert actual.status == expected.status
        assert actual.network_calls == expected.network_calls
        assert cache.hits > 0
        assert cache.stats_by_helper['activation']['hits'] > 0
        if model_id=='affine_flow':
            assert cache.stats_by_helper['flow_lag']['hits'] > 0
        if model_id=='node_actuator_aware':
            assert cache.stats_by_helper['response']['hits'] > 0
        derivative_h = h.clone().requires_grad_()
        gradient = torch.autograd.grad(model.predict(state,command,derivative_h).next_state.sum(), derivative_h)[0]
        assert torch.equal(gradient,original_gradient)


def test_cache_key_includes_interval_tau_dtype_and_device():
    m = module()
    cache = m.FixedCoefficientCache()
    h = torch.tensor([[.02]],dtype=torch.float64)
    key = cache.key(h,.08)
    assert key != cache.key(h,.09)
    assert key != cache.key(h.float(),.08)
    assert key != cache.key(h*2,.08)
    # A meta tensor cannot supply values; device identity is explicit in the key.
    assert 'cpu' in str(key)
    from poc.actuator_integrals import response_ratios
    a = cache.response_ratios(h,.08)
    b = cache.response_ratios(h,.09)
    assert torch.equal(a[0],response_ratios(h,.08)[0])
    assert torch.equal(b[0],response_ratios(h,.09)[0])
    assert not torch.equal(a[0],b[0])


def unit_store(tmp_path):
    out = tmp_path/'disposable-timing-unit'
    prepare(ROOT,out,unit_fixture={'segment_count':5,'max_epochs_per_rate':5,
        'evaluation_trajectory_count':2,'evaluation_anchor_times_ns':[360_000_000],
        'decision_candidate_count':2})
    store = open_store(ROOT,out)
    receipt = collect_session(0,root=ROOT,output_root=out)
    fit = nominal_reference(store,receipt,5)
    evaluation = prepare_evaluation(store,root=ROOT)
    return store,fit,evaluation


def test_registry_covers_every_batch_partition_and_real_primary_state(tmp_path):
    m = module()
    store,fit,evaluation = unit_store(tmp_path)
    clock = ManualClock()
    receipt = m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)
    folder,manifest = store.read(receipt)
    registry = json.loads((folder/'registry.json').read_text())
    assert len(registry) == 28  # representative state * 4 partitions * (3 network + 3 predict + 1 scorer)
    assert len({json.dumps(c,sort_keys=True) for c in registry}) == 28
    assert {c['state_id'] for c in registry} == {0}
    assert {c['anchor_ns'] for c in registry} == {360_000_000}
    assert {c['batch_size'] for c in registry if c['category']=='prediction'} == {1,128,256}
    rows = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(rows) == 1400
    complete = [r for r in rows if r['coordinate']['category']=='complete_scorer']
    assert len(complete) == 200
    for row in complete:
        step = row['coordinate']['query_step_ns']
        assert row['work']['scalar_candidate_evaluations'] == 96
        assert row['work']['auxiliary_scalar_predictions'] == (48-240_000_000//step)*2
        assert row['work']['state_feasibility_checks'] == 98
        assert row['selection']['selected_index'] is not None
    assert not manifest['summary']['empirical_claims_allowed']
    assert manifest['summary']['resources']['dtype'] == 'float64'
    assert manifest['summary']['resources']['torch_num_threads'] == 1
    assert manifest['summary']['resources']['synchronization'] == 'synchronous CPU; no device synchronization'
    assert 'audit_wall_s' in manifest['summary']['noncritical']
    assert receipt == m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)


def test_small_real_inference_and_clock_integration():
    m = module()
    model = build_model('affine_flow',Scales(*(torch.ones(2,dtype=torch.float64) for _ in range(3))),0).double().eval()
    state = torch.zeros((1,6),dtype=torch.float64)
    command = torch.zeros((1,2),dtype=torch.float64)
    cells = [({'category':'prediction','model_id':'affine_flow'},lambda:model.predict(state,command,.02))]
    with m.fixed_coefficient_cache(model):
        rows = m.run_rotated_cells(cells)
    assert len(rows)==50 and all(r['wall_s']>0 and r['cpu_s']>0 for r in rows)
    assert all(r['work']['network_calls']==2 for r in rows)


def test_cli_time_reads_existing_selection_and_reports_missing(tmp_path,capsys):
    module()
    from poc.calibration_study import main
    store,fit,evaluation = unit_store(tmp_path)
    assert main(['time','--root',str(ROOT),'--output-root',str(store.output_root),
                 '--model','affine_flow','--phase','unit']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['timing_receipt'] is None
    assert result['missing_coordinates'] == [{'session_id':0,'budget_s':5,'model_id':'affine_flow','initialization_id':0}]
    assert result['evaluation_receipt'] == evaluation


def test_method_rotation_occurs_within_each_comparable_cell():
    m = module()
    order = []
    def op(name):
        def run():
            order.append(name)
            return {'status':'ok','work':{}}
        return run
    cells = [({'category':category,'model_id':model},op(category+model))
             for category in ['prediction','complete_scorer'] for model in ['A','B']]
    m.run_rotated_cells(cells,clock=ManualClock(),cpu_clock=ManualClock())
    assert order[:8] == ['predictionA','predictionB','complete_scorerA','complete_scorerB',
                         'predictionB','predictionA','complete_scorerB','complete_scorerA']


def test_real_scorer_constraints_cost_and_selection_stay_in_outer_duration(monkeypatch):
    m = module()
    from poc import calibration_decisions as decisions
    from poc.models.common import prediction
    clock = ManualClock()
    class FastMap(torch.nn.Module):
        def predict(self,state,command,h):
            clock.now += .001
            return prediction(state.clone(),1,1)
    old_cost,old_safe = decisions._cost,decisions._safe
    def slow_cost(*args):
        clock.now += .5
        return old_cost(*args)
    def slow_safe(*args):
        clock.now += .01
        return old_safe(*args)
    monkeypatch.setattr(decisions,'_cost',slow_cost)
    monkeypatch.setattr(decisions,'_safe',slow_safe)
    state = torch.zeros(6,dtype=torch.float64)
    model = FastMap()
    cells = [({'category':'complete_scorer','query_step_ns':40_000_000},
              lambda:decisions.score_candidates(model,state,np.zeros(2),np.zeros((2,6,2)),query_step_ns=40_000_000))]
    rows = m.run_rotated_cells(cells,clock=clock,cpu_clock=clock)
    # 6 packed predictions + initial feasibility + 6 packed feasibility checks + physical cost.
    assert all(r['wall_s']==pytest.approx(.576) for r in rows)
    assert all(r['selection']['selected_index']==0 for r in rows)
    assert all(r['work']['state_feasibility_checks']==98 for r in rows)


def test_serialization_delay_is_separate_from_complete_scorer(tmp_path,monkeypatch):
    m = module()
    store,fit,evaluation = unit_store(tmp_path)
    clock = ManualClock()
    real_write = m.atomic_bytes
    def delayed_write(*args):
        clock.now += 100.
        return real_write(*args)
    monkeypatch.setattr(m,'atomic_bytes',delayed_write)
    receipt = m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)
    _,manifest = store.read(receipt)
    assert manifest['summary']['noncritical']['logging_serialization_write_wall_s']==100.
    complete = [g for g in manifest['summary']['groups'] if g['coordinate']['category']=='complete_scorer']
    assert all(g['wall_s']['p95']==0. for g in complete)
    assert manifest['summary']['cache_parity_audit']['status']=='exact'


def test_fresh_cli_time_scores_selected_own_affine_map(tmp_path):
    import subprocess
    import sys
    from poc.calibration_training import fit_model,load_selected
    m = module()
    store,_,evaluation = unit_store(tmp_path)
    collected = collect_session(0,root=ROOT,output_root=store.output_root)
    fit = fit_model(store,collected,'affine_flow',5,0)
    expected_hash = store.read(fit)[1]['summary']['canonical_tensor_sha256']
    result = subprocess.run([sys.executable,'-m','poc.calibration_study','time',
        '--root',str(ROOT),'--output-root',str(store.output_root),'--phase','unit',
        '--model','affine_flow','--session','0','--budget','5','--initialization','0'],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report = json.loads(result.stdout)
    assert report['missing_coordinates']==[]
    folder,manifest = store.read(report['timing_receipt'])
    assert manifest['summary']['selected_tensor_hashes']==[expected_hash]
    assert manifest['summary']['cache_parity_audit']['status']=='exact'
    assert manifest['summary']['sample_count']==1400
    assert report['evaluation_receipt']==evaluation
    from poc.calibration_decisions import candidate_bank,score_candidates
    from poc.calibration_predictions import load_evaluation
    arrays = load_evaluation(store,evaluation)
    expected = score_candidates(load_selected(store,fit),torch.from_numpy(arrays['state'][0]),
        arrays['target'][0],candidate_bank(evaluation,0,store),query_step_ns=40_000_000)
    raw = [json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    scored = [r for r in raw if r['coordinate']['category']=='complete_scorer' and r['coordinate']['query_step_ns']==40_000_000]
    assert all(r['selection']['selected_index']==expected['selected_index'] for r in scored)
    assert all(r['work']==expected['work'] for r in scored)


def test_failed_fit_and_absent_network_have_no_valid_latency_summary():
    m = module()
    rows = [{'coordinate':{'category':category},'wall_s':.0001,'cpu_s':.0001,
             'work':{},'status':status,'repetition':0}
            for category,status in [('prediction','fit_failed'),('network_forward','not_applicable')]]
    summary = m.summarize_timing(rows)
    assert all(g['wall_s']['median'] is None and g['cpu_s']['p95'] is None for g in summary['groups'])
    assert all(g['valid_timing_calls']==0 for g in summary['groups'])


def test_cache_rebinding_preserves_qualified_physics_partial_and_restores_instance():
    import functools
    m = module()
    model = build_model('physics_fit',Scales(*(torch.ones(2,dtype=torch.float64) for _ in range(3))),0).double()
    original = functools.partial(model.predict,integration_step_s=.0025)
    model.predict = original
    state = torch.zeros((1,6),dtype=torch.float64)
    command = torch.ones((1,2),dtype=torch.float64)
    expected = original(state,command,.02)
    with pytest.raises(RuntimeError,match='unit exit'):
        with m.fixed_coefficient_cache(model):
            actual = model.predict(state,command,.02)
            assert torch.equal(actual.next_state,expected.next_state)
            assert actual.vector_field_calls==32
            raise RuntimeError('unit exit')
    assert model.predict is original


def test_valid_failed_fit_keeps_full_timing_coverage_and_invalid_evidence_raises(tmp_path,monkeypatch):
    m = module()
    from poc import calibration_training as training
    store,reference,evaluation = unit_store(tmp_path)
    collected = collect_session(0,root=ROOT,output_root=store.output_root)
    def fail(*args):
        raise RuntimeError('intentional disposable failed fit')
    monkeypatch.setattr(training.CalibrationTrainingSession,'advance_updates',fail)
    fit = training.fit_model(store,collected,'affine_flow',5,0)
    clock = ManualClock()
    receipt = m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)
    folder,manifest = store.read(receipt)
    rows = [json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==1400 and all(r['status']=='fit_failed' for r in rows)
    assert all(g['valid_timing_calls']==0 and g['wall_s']['median'] is None for g in manifest['summary']['groups'])
    with pytest.raises(ValueError,match='hash'):
        m.measure_timing(store,[{**reference,'manifest_sha256':'invalid'}],evaluation)
    with pytest.raises(ValueError,match='Duplicate'):
        m.measure_timing(store,[reference,reference],evaluation)


def test_runtime_predictor_exceptions_keep_cells_and_attempted_scorer_work(tmp_path,monkeypatch):
    m = module()
    from poc import calibration_training as training
    store,reference,evaluation = unit_store(tmp_path)
    original_load = training.load_selected
    def unavailable(*args):
        model = original_load(*args)
        def fail(*args):
            raise RuntimeError('intentional unit inference failure')
        model.predict = fail
        return model
    monkeypatch.setattr(training,'load_selected',unavailable)
    clock = ManualClock()
    folder,manifest = store.read(m.measure_timing(store,[reference],evaluation,clock=clock,cpu_clock=clock))
    rows = [json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    predictions = [r for r in rows if r['coordinate']['category']=='prediction']
    assert len(predictions)==600 and all(r['status'].startswith('predictor_exception:RuntimeError') for r in predictions)
    scorers = [r for r in rows if r['coordinate']['category']=='complete_scorer']
    assert len(scorers)==200 and all(r['selection']['prediction_failure_count']==2 for r in scorers)
    assert all(r['work']['prediction_batch_calls']==1 for r in scorers)


def test_processor_identity_discloses_local_hardware_where_available():
    import platform
    import subprocess
    m = module()
    identity = m.processor_identity()
    assert identity
    if platform.system()=='Darwin':
        result = subprocess.run(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True)
        if result.returncode==0:
            assert identity==result.stdout.strip()


def test_timing_final_store_hash_and_receipt_delay_have_numeric_external_category(tmp_path,monkeypatch):
    m = module()
    from poc import calibration_store as store_module
    from poc import calibration_training as training
    store,_,evaluation = unit_store(tmp_path)
    collected = collect_session(0,root=ROOT,output_root=store.output_root)
    def failed(*args):
        raise RuntimeError('disposable failed fit keeps this timing-boundary test bounded')
    monkeypatch.setattr(training.CalibrationTrainingSession,'advance_updates',failed)
    fit = training.fit_model(store,collected,'affine_flow',5,0)
    clock = ManualClock()
    real_hash = store_module.sha256_file
    hash_delay = [10.]
    def delayed_timing_hash(path):
        if '/jobs/timing/' in str(path):
            clock.now += hash_delay[0]
        return real_hash(path)
    monkeypatch.setattr(store_module,'sha256_file',delayed_timing_hash)
    receipt = m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)
    folder,manifest = store.read(receipt)
    events = [json.loads(line) for line in (store.output_root/'events.jsonl').read_text().splitlines()]
    record = [e for e in events if e['event']=='store_finalization' and e['receipt']==receipt][-1]
    assert record['resources']['artifact_hash_wall_s']==40.  # request, registry, samples, resources
    assert record['resources']['metadata_write_hash_wall_s']==10.  # completion marker manifest hash
    assert record['resources']['receipt_create_validate_wall_s']==60.  # receipt + manifest + four artifact hashes
    assert record['resources']['store_finalization_wall_s']==110.
    assert manifest['summary']['noncritical']['store_finalization_record']['event']=='store_finalization'
    raw = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(raw)==1400 and all(row['wall_s']==0. for row in raw)
    assert all(row['cpu_s']==0. for row in raw)
    original = m.load_timing_finalization(store,receipt)
    hash_delay[0] = 20.
    assert m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)==receipt
    assert m.load_timing_finalization(store,receipt)==original
    event_path = store.output_root/'events.jsonl'
    content = event_path.read_text()
    event_path.write_text(content+json.dumps(original)+'\n')
    with pytest.raises(ValueError,match='Ambiguous'):
        m.load_timing_finalization(store,receipt)
    event_path.write_text('\n'.join(line for line in content.splitlines()
        if json.loads(line)['event']!='store_finalization')+'\n')
    with pytest.raises(ValueError,match='Missing'):
        m.load_timing_finalization(store,receipt)
    with pytest.raises(ValueError,match='Missing'):
        m.measure_timing(store,[fit],evaluation,clock=clock,cpu_clock=clock)


def test_residual_network_timing_uses_initial_activation():
    from poc.calibration_timing import _network_inputs
    from poc.calibration_models import build_model
    from poc.models.common import Scales,features
    model=build_model('physics_residual',Scales(*(torch.ones(2) for _ in range(3))),0).double()
    state=torch.tensor([[.1,-.1,.2,-.2,.3,-.3]],dtype=torch.float64)
    command=torch.tensor([[1.,-1.]],dtype=torch.float64)
    h=torch.tensor([[.04]],dtype=torch.float64)
    expected=features(state[:,:4],state[:,4:6],torch.zeros_like(h),model.v_input)[:,:-1]
    torch.testing.assert_close(_network_inputs(model,state,command,h),expected,atol=0,rtol=0)
