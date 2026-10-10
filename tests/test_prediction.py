import numpy as np
import pytest
import torch
from torch import nn

from poc.analytic import transition
from poc.models.common import prediction
from poc.prediction import rollout_path, prediction_metrics
from poc.probes import make_probe_library, summarize_probes


class Exact(nn.Module):
    def __init__(self, stationary=False):
        super().__init__()
        self.stationary = stationary
        self.history_sizes = []

    def predict(self, state, command, h_s, history=None):
        self.history_sizes.append(len(history.eligible().tokens[0]))
        if self.stationary:
            output = state.clone()
        else:
            h = float(h_s)
            output = torch.cat((state[:,:2]+h*state[:,2:4]+.5*h*h*command,
                                state[:,2:4]+h*command), dim=1)
        return prediction(output, 1)


def corpus(count=2, condition='F'):
    duration = np.full((count,32),40_000_000,dtype=np.int64)
    if condition == 'A':
        duration[:,::2] = 20_000_000
    time = np.c_[np.zeros(count,dtype=np.int64),np.cumsum(duration,axis=1)]
    commands = np.tile([.2,-.3],(count,32,1))
    states = np.zeros((count,33,4),dtype=np.float64)
    for i in range(count):
        for j in range(32):
            states[i,j+1] = transition(states[i,j],commands[i,j],duration[i,j]/1e9,'direct')
    return {'state':states,'command':commands,'duration_ns':duration,'time_ns':time,
            'trajectory_id':np.arange(count,dtype=np.int64)}


def test_cold_warm_probes_choose_first_completed_anchor_and_fixed_history():
    arrays = corpus(condition='A')
    library = make_probe_library(arrays,'direct','A','development')
    assert len(library['states']) == 4
    warm = np.flatnonzero(library['strata'] == 'warm')
    for i in warm:
        anchor = library['anchor_ids'][i]
        trajectory = library['trajectory_ids'][i]
        assert arrays['time_ns'][trajectory,anchor] >= 320_000_000
        assert arrays['time_ns'][trajectory,anchor-1] < 320_000_000
        assert len(library['histories'][i].eligible().tokens[0]) >= 8
    assert max(abs(library['commands']).ravel()) <= 1.5
    replay = make_probe_library(arrays,'direct','A','development')
    np.testing.assert_array_equal(library['commands'],replay['commands'])


def test_confirmation_library_has_exactly_512_unique_probe_ids():
    library = make_probe_library(corpus(256),'direct','F','confirmation')
    assert len(library['probe_ids']) == 512
    assert len(set(library['probe_ids'])) == 512
    assert np.count_nonzero(library['strata'] == 'cold') == 256


@pytest.mark.parametrize('step',[40_000_000,20_000_000,10_000_000,5_000_000,2_500_000,'jitter'])
def test_exact_rollout_scores_only_truth_anchors_and_commits_no_substep_observations(step):
    arrays = corpus()
    model = Exact()
    result = rollout_path(model,arrays['state'][0,0],arrays['command'][0],
                          arrays['duration_ns'][0],step,seed_entropy=(1201,0,0,4))
    np.testing.assert_allclose(result['states'],arrays['state'][0],atol=1e-12,rtol=1e-12)
    assert result['time_ns'].tolist() == arrays['time_ns'][0].tolist()
    assert result['observations_received'] == 1
    assert len(result['history_commits']) == 32
    assert all(event['provenance'] == 'imagined' for event in result['history_commits'])
    first_hold_calls = 8 if step == 'jitter' else 40_000_000//step
    sizes = [len(event['history_packet_ids']) for event in result['traces']]
    assert sizes[:first_hold_calls] == [0]*first_hold_calls
    assert sizes[first_hold_calls:2*first_hold_calls] == [1]*first_hold_calls
    assert model.history_sizes == []  # audit copies do not mutate the supplied model


def test_consistent_stationary_predictor_exposes_large_truth_error():
    arrays = corpus()
    model = Exact(stationary=True)
    coarse = rollout_path(model,arrays['state'][0,0],arrays['command'][0],arrays['duration_ns'][0],40_000_000)
    fine = rollout_path(model,arrays['state'][0,0],arrays['command'][0],arrays['duration_ns'][0],5_000_000)
    result = prediction_metrics(fine,arrays['state'][0],coarse,{'scale_q':[1.,1.],'scale_v':[1.,1.]})
    assert result['partition_discrepancy'] == 0
    assert result['q_rmse_rad'] > .01
    assert result['v_rmse_rad_s'] > .01


def test_model_failure_preserves_all_future_anchor_rows():
    class Failed(Exact):
        def predict(self,*args,**kwargs):
            return prediction(torch.full_like(args[0],float('nan')),1)
    arrays = corpus()
    result = rollout_path(Failed(),arrays['state'][0,0],arrays['command'][0],arrays['duration_ns'][0],5_000_000)
    assert len(result['states']) == 33
    assert list(result['statuses'][1:]) == ['model_failure']*32
    assert result['observations_received'] == 1


def test_probe_summary_never_pools_cold_warm_or_missing_models():
    rows = {'cell_id':np.array(['a','a','a','a']), 'stratum':np.array(['cold','cold','warm','warm']),
            'h_ns':np.full(4,5_000_000), 'error_q':np.array([0.,2.,4.,6.]),
            'error_v':np.array([2.,4.,6.,8.]),'status':np.array(['ok']*4)}
    result = summarize_probes(rows,['a'])
    assert result['a']['cold']['5000000']['primary_error'] == 2
    assert result['a']['warm']['5000000']['primary_error'] == 6
    with pytest.raises(ValueError,match='coverage'):
        summarize_probes(rows,['a','b'])
    rows['status'][0] = 'bad'
    with pytest.raises(ValueError,match='failed|missing'):
        summarize_probes(rows,['a'])


def test_reference_cache_preserves_raw_independent_branches_and_rejects_corruption(tmp_path):
    from pathlib import Path
    from poc.probes import reference_library
    from poc.io import read_json
    import shutil
    project = Path(__file__).resolve().parents[1]
    for path in ('poc/probes.py','poc/model_sensitivity.py','poc/envs.py','poc/oracle_rigidbody.py',
                 'assets/two_link_direct.xml'):
        target = tmp_path/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/path,target)
    library = make_probe_library(corpus(1),'direct','F','development')
    numerical = read_json(project/'config/numerical.json')
    first = reference_library(library,'direct',[5_000_000],numerical,tmp_path)
    second = reference_library(library,'direct',[5_000_000],numerical,tmp_path)
    np.testing.assert_array_equal(first['jacobians'],second['jacobians'])
    assert first['endpoints'].shape == (2,1,3,2,2,4)
    assert first['oracle_endpoints'].shape == (2,1,2,2,4)
    assert np.all(first['numerical_envelope'] > 0)
    path = next((tmp_path/'data/audit_cache/reference').glob('*/*.npz'))
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        reference_library(library,'direct',[5_000_000],numerical,tmp_path)


def test_e01_producer_writes_all_typed_coordinates_and_raw_derivative_evidence(tmp_path,monkeypatch):
    from pathlib import Path
    import shutil
    from poc.io import read_json
    import poc.probes
    import poc.audit_jobs
    from poc.model_sensitivity import run_e01
    project = Path(__file__).resolve().parents[1]
    sources = ['poc/probes.py','poc/model_sensitivity.py','poc/envs.py','poc/oracle_rigidbody.py',
        'poc/history.py','assets/two_link_direct.xml','config/numerical.json','locks/environment.json']
    sources += ['poc/models/'+name+'.py' for name in ('common','power','mlp','sequence','ode','factory')]
    for name in sources:
        destination = tmp_path/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project/name,destination)
    parameters = {'model_id':'exact_fixture','actuator':'direct','condition':'F','loss_mode':'one_step',
                  'seed':0,'corpus_manifest':'unit-development-fixture'}
    metadata = {'model':parameters,'corpus':{'actuator':'direct','condition':'F','role':'development'},
                'checkpoint_hash':'a'*64,'input_hashes':{'unit_history':'b'*64}}
    monkeypatch.setattr(poc.probes,'load_audit_inputs',lambda *_:(Exact(),corpus(1),
                 {'scale_q':[1.,1.],'scale_v':[1.,1.]},metadata))
    def execute(_root,_experiment,_job,_metadata,_checks,producer):
        folder = tmp_path/'unit-audit'
        folder.mkdir(exist_ok=True)
        return producer(folder)
    monkeypatch.setattr(poc.audit_jobs,'audit_job',execute)
    checks,summary = run_e01({'semantic_inputs':parameters,'phase':'unit'},tmp_path)
    assert checks[0]['status'] == 'pass' and summary['failed_rows'] == 0
    assert summary['row_count'] == 12
    with np.load(tmp_path/'unit-audit/jacobians.npz',allow_pickle=False) as arrays:
        assert arrays['q_jac'].shape == (12,2,2)
        assert arrays['central_endpoints'].shape == (12,2,2,2,4)
        for h,jacobian in zip(arrays['h_ns'],arrays['q_jac']):
            np.testing.assert_allclose(jacobian,np.eye(2)*.5*(h/1e9)**2,atol=1e-14,rtol=1e-12)
    slopes = read_json(tmp_path/'unit-audit/slopes.json')['rows']
    assert len(slopes) == 4
    assert all(row['status'] == 'ok' for row in slopes)
    assert (tmp_path/'unit-audit/metrics_by_seed.csv').exists()
    caches = tmp_path/'data/audit_cache/model_jacobians'
    assert len(list(caches.iterdir())) == 1
    metadata['input_hashes']['unit_history'] = 'c'*64
    run_e01({'semantic_inputs':parameters,'phase':'unit'},tmp_path)
    assert len(list(caches.iterdir())) == 2  # The same physical probes may have different preceding histories.
