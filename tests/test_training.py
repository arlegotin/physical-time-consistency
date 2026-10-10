"""Matched supervised exposure, actual-time rollout and exact continuation."""
import importlib
import numpy as np
import pytest
import torch

from poc.models.common import Scales
from poc.models.factory import build_model


def module():
    assert importlib.util.find_spec('poc.train'), 'Matched supervised training is not implemented'
    return importlib.import_module('poc.train')


def scales():
    return Scales(torch.tensor([2.,3.]),torch.tensor([4.,5.]),torch.ones(2))


def corpus(count=2,steps=32):
    rng = np.random.Generator(np.random.PCG64(5521))
    durations = rng.choice(np.array([20,30,40,50,60],dtype=np.int64)*1_000_000,(count,steps))
    return {'state':rng.uniform(-1,1,(count,steps+1,4)),
            'command':rng.uniform(-1,1,(count,steps,2)),'duration_ns':durations,
            'time_ns':np.concatenate((np.zeros((count,1),dtype=np.int64),np.cumsum(durations,axis=1)),axis=1),
            'trajectory_id':np.arange(count,dtype=np.int64)}


def test_checkpoint_window_storage_count_does_not_grow_with_training_corpus(tmp_path):
    import zipfile
    training = module()
    counts = []
    for n in (2,1024):
        session = training.TrainingSession(build_model('mlp_dt','direct',scales(),1),corpus(n),scales(),1)
        path = tmp_path/f'{n}.pt'
        session.save(path)
        with zipfile.ZipFile(path) as archive:
            counts.append(sum('/data/' in name for name in archive.namelist()))
        restored = training.TrainingSession(build_model('mlp_dt','direct',scales(),1),corpus(n),scales(),1)
        restored.restore(path)
        assert len(restored.windows) == len(session.windows)
        for before,after in zip(session.windows,restored.windows):
            np.testing.assert_array_equal(before,after)
    assert counts[1]-counts[0] <= 2


def test_packed_and_legacy_ragged_windows_resume_identical_optimizer_updates(tmp_path):
    import io
    from poc.io import atomic_bytes,atomic_json,read_json,sha256_file
    training = module()
    arrays = corpus(3,31)
    original = training.TrainingSession(build_model('mlp_dt','direct',scales(),7),arrays,scales(),7)
    original.advance_updates(2)
    path = tmp_path/'legacy.pt'
    original.save(path)
    saved = torch.load(path,weights_only=True,map_location='cpu')
    flat,offsets = saved['windows'],saved.pop('window_offsets')
    saved['windows'] = [flat[int(a):int(b)].clone() for a,b in zip(offsets[:-1],offsets[1:])]
    saved.pop('checkpoint_format_version')
    buffer = io.BytesIO(); torch.save(saved,buffer)
    atomic_bytes(path,buffer.getvalue())
    marker = read_json(path.with_suffix('.complete.json'))
    marker['checkpoint_sha256'] = sha256_file(path)
    atomic_json(path.with_suffix('.complete.json'),marker)
    resumed = training.TrainingSession(build_model('mlp_dt','direct',scales(),7),arrays,scales(),7)
    resumed.restore(path)
    original.advance_updates(2)
    resumed.advance_updates(2)
    assert resumed.tensor_hash() == original.tensor_hash()
    assert resumed.valid_target_count == original.valid_target_count
    assert resumed.training_rows == original.training_rows
    assert any(len(window) == 7 for window in resumed.windows)


def test_epoch_checkpoint_commits_after_complete_updates_and_resumes_exactly(tmp_path,monkeypatch):
    training = module()
    arrays = corpus(17)
    original = training.TrainingSession(build_model('mlp_dt','direct',scales(),8),arrays,scales(),8)
    calls = []
    save = original.save
    def observed(path):
        calls.append((original.updates,original.window_position))
        save(path)
    monkeypatch.setattr(original,'save',observed)
    path = tmp_path/'epoch.pt'
    original.advance_epoch(path)
    assert calls == [(3,len(original.windows))]
    assert original.valid_target_count == 17*32
    restored = training.TrainingSession(build_model('mlp_dt','direct',scales(),8),arrays,scales(),8)
    restored.restore(path)
    original.advance_updates(2)
    restored.advance_updates(2)
    assert restored.tensor_hash() == original.tensor_hash()
    assert restored.training_rows == original.training_rows


def test_endpoint_loss_gradient_and_known_activation_exclusion():
    predicted = torch.tensor([[1.,2.,3.,4.,99.,-99.]],dtype=torch.float64,requires_grad=True)
    target = torch.zeros_like(predicted)
    loss = module().endpoint_loss(predicted,target,scales())
    expected = ((1/2)**2+(2/3)**2+(3/4)**2+(4/5)**2)/4
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    torch.testing.assert_close(predicted.grad,torch.tensor([[1/8,1/9,3/32,2/25,0.,0.]],dtype=torch.float64))


def test_exact_adamw_update_matches_independent_scalar_arithmetic():
    parameter = torch.nn.Parameter(torch.tensor([.7],dtype=torch.float64))
    optimizer = module().optimizer_for([parameter],learning_rate=.001)
    loss = (parameter-2).square().sum()
    loss.backward()
    optimizer.step()
    gradient = 2*(.7-2)
    expected = .7*(1-.001*1e-6)-.001*gradient/(abs(gradient)+1e-8)
    assert parameter.item() == pytest.approx(expected,abs=1e-15)
    assert optimizer.defaults['betas'] == (.9,.999)
    assert optimizer.defaults['foreach'] is False and optimizer.defaults['fused'] is False


def test_epoch_windows_cover_every_start_once_and_keep_last_three_one_step_targets():
    training = module()
    arrays = corpus(3)
    windows = training.epoch_starts(arrays,np.random.Generator(np.random.PCG64(1)))
    starts = np.concatenate(windows)
    assert len(starts) == 3*32
    assert set(map(tuple,starts)) == {(trajectory,start) for trajectory in range(3) for start in range(32)}
    assert sum(start+4 <= 32 for _,start in starts) == 3*29
    assert all(len(window) == 8 for window in windows)


def test_actual_duration_rollout_closed_loop_and_weighting():
    training = module()
    arrays = corpus(1)
    model = build_model('power_correct','direct',scales(),1)
    result = training.batch_objective(model,arrays,np.array([[0,0],[0,30],[0,31]]),scales(),'shared')
    assert result['one_count'] == 3 and result['four_count'] == 1
    torch.testing.assert_close(result['loss'],result['one_loss']+.25*result['four_loss'])
    state = torch.tensor(arrays['state'][0,0:1],dtype=torch.float32)
    scores = []
    for step in range(4):
        command = torch.tensor(arrays['command'][0,step:step+1],dtype=torch.float32)
        state = model.predict(state,command,arrays['duration_ns'][0,step]/1e9).next_state
        scores.append(training.endpoint_loss(state,torch.tensor(arrays['state'][0,step+1:step+2],dtype=torch.float32),scales()))
    torch.testing.assert_close(result['four_loss'],torch.stack(scores).mean())


def test_four_step_sequence_history_uses_predicted_states_and_dataset_anchors():
    training = module()
    arrays = corpus(1)
    model = build_model('gru_dt','direct',scales(),2)
    history = training.histories_for(arrays,np.array([[0,4]]))
    assert [token.duration_ns for token in history.tokens[0]] == arrays['duration_ns'][0,:4].tolist()
    result = training.batch_objective(model,arrays,np.array([[0,4]]),scales(),'shared')
    result['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert result['imagined_history_anchors'] == arrays['time_ns'][0,5:9].tolist()
    assert result['imagined_tokens'] == 4


@pytest.mark.parametrize('loss_mode',['one_step','shared'])
def test_cached_and_uncached_history_preserve_exact_optimizer_evolution(monkeypatch,loss_mode):
    import poc.models.sequence as sequence
    training = module()
    arrays = corpus(4)
    cached = training.TrainingSession(build_model('gru_dt','direct',scales(),21),arrays,scales(),21,
                                     loss_mode=loss_mode,effective_batch=32)
    cached.advance_updates(8)
    original = sequence.history_features
    def uncached(*args,cache=None):
        return original(*args,cache=None)
    monkeypatch.setattr(sequence,'history_features',uncached)
    reference = training.TrainingSession(build_model('gru_dt','direct',scales(),21),arrays,scales(),21,
                                        loss_mode=loss_mode,effective_batch=32)
    reference.advance_updates(8)
    assert cached.tensor_hash() == reference.tensor_hash()
    assert cached.training_rows == reference.training_rows
    assert cached.valid_target_count == reference.valid_target_count
    for name,values in cached.optimizer.state_dict()['state'].items():
        for key,value in values.items():
            torch.testing.assert_close(value,reference.optimizer.state_dict()['state'][name][key],rtol=0,atol=0)


def test_imagined_history_keeps_exact_integer_duration_instead_of_rounding_float32_seconds():
    training = module()
    arrays = corpus(1)
    arrays['duration_ns'][0,:4] = 30_000_000
    arrays['time_ns'][0,1:] = np.cumsum(arrays['duration_ns'][0])
    model = build_model('gru_dt','direct',scales(),2)
    original, seen = model.predict,[]
    def record(state,command,h_s,history=None):
        seen.extend(token.duration_ns for tokens in history.tokens for token in tokens if token.provenance == 'imagined')
        return original(state,command,h_s,history=history)
    model.predict = record
    training.batch_objective(model,arrays,np.array([[0,0]]),scales(),'shared')
    assert seen and set(seen) == {30_000_000}


def test_checkpoint_rejects_changed_optimizer_recipe_and_mismatched_rng_sidecar(tmp_path):
    from poc.io import read_json,atomic_json
    training = module()
    arrays = corpus(1,8)
    session = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4)
    session.advance_updates(1)
    path = tmp_path / 'resume.pt'
    session.save(path)
    other = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4,learning_rate=.0003)
    with pytest.raises(ValueError,match='recipe|inputs'):
        other.restore(path)
    sidecar = path.with_suffix('.rng.json')
    rng = read_json(sidecar)
    rng['numpy']['state']['state'] += 1
    atomic_json(sidecar,rng)
    with pytest.raises(ValueError,match='hash|checkpoint'):
        session.restore(path)


def test_selected_checkpoint_ties_choose_earlier_epoch():
    selection = module().select_checkpoint([{'epoch':5,'objective':1.},
           {'epoch':10,'objective':1.-5e-9},{'epoch':15,'objective':1.2}])
    assert selection['epoch'] == 5
    assert module().select_checkpoint([{'epoch':5,'objective':1.},{'epoch':10,'objective':.9}])['epoch'] == 10


def test_weights_optimizer_rng_permutation_and_target_counts_resume_exactly(tmp_path):
    training = module()
    arrays = corpus(2,8)
    reference = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4,
                                         loss_mode='shared',effective_batch=8)
    interrupted = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4,
                                           loss_mode='shared',effective_batch=8)
    reference.advance_updates(4)
    interrupted.advance_updates(1)
    interrupted.save(tmp_path / 'resume.pt')
    resumed = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4,
                                      loss_mode='shared',effective_batch=8)
    resumed.restore(tmp_path / 'resume.pt')
    resumed.advance_updates(3)
    assert reference.valid_target_count == resumed.valid_target_count == 32
    assert reference.epoch == resumed.epoch and reference.window_position == resumed.window_position
    assert reference.tensor_hash() == resumed.tensor_hash()
    assert reference.training_rows == resumed.training_rows


def test_nominal_plateau_moving_windows_and_still_improving_are_distinct():
    training = module()
    flat = [{'epoch':i,'objective':1.,'one_step':.2} for i in range(5,201,5)]
    improving = [{'epoch':i,'objective':1/i,'one_step':1/i} for i in range(5,201,5)]
    assert training.nominal_convergence(flat,200)['stabilized']
    assert not training.nominal_convergence(improving,200)['stabilized']
    assert not training.nominal_convergence(flat[:10],50)['stabilized']


def test_canonical_tensor_hash_ignores_archive_metadata_and_detects_changed_weights(tmp_path):
    training = module()
    model = build_model('mlp_dt','direct',scales(),9)
    first = training.canonical_tensor_hash(model.state_dict())
    torch.save(model.state_dict(),tmp_path / 'first.pt')
    torch.save(model.state_dict(),tmp_path / 'second.pt')
    loaded = torch.load(tmp_path / 'second.pt',weights_only=True,map_location='cpu')
    assert training.canonical_tensor_hash(loaded) == first
    with torch.no_grad():
        next(model.parameters()).flatten()[0] += .001
    assert training.canonical_tensor_hash(model.state_dict()) != first


@pytest.mark.parametrize('actuator',['direct','lagged'])
def test_analytic_learning_control_labels_are_independent_exact_cold_transitions(actuator):
    training = module()
    assert hasattr(training,'analytic_control_arrays'), 'Analytic learning controls are not implemented'
    from poc.analytic import transition
    arrays = training.analytic_control_arrays(actuator,'tiny')
    assert arrays['state'].shape == (16,2,4 if actuator == 'direct' else 6)
    assert np.all(arrays['duration_ns'] == 40_000_000)
    for initial,command,next_state in zip(arrays['state'][:,0],arrays['command'][:,0],arrays['state'][:,1]):
        np.testing.assert_allclose(next_state,transition(initial,command,.04,actuator),atol=0,rtol=0)
    assert not np.array_equal(training.analytic_control_arrays(actuator,'train')['state'][:16,0],arrays['state'][:,0])


def test_nominal_validation_scores_each_label_once_and_training_resume_has_saved_accumulators():
    training = module()
    arrays = corpus(2,8)
    model = build_model('mlp_dt','direct',scales(),4)
    assert hasattr(training,'validate_nominal'), 'Nominal validation is not implemented'
    scores = training.validate_nominal(model,arrays,scales(),'shared')
    assert scores['one_count'] == 16 and scores['four_count'] == 10
    assert scores['objective'] == pytest.approx(scores['one_step']+.25*scores['four_step'])


def test_confirmatory_training_job_cannot_open_confirmation_as_training_labels(tmp_path):
    from poc.io import atomic_json
    training = module()
    assert hasattr(training,'fit'), 'Artifact-producing training fits are not implemented'
    manifest = tmp_path / 'data/direct/F/confirmation/manifest.json'
    atomic_json(manifest,{'role':'confirmation'})
    job = {'protocol_version':'PTCO-1.0','milestone_id':'M30','phase':'confirmatory','purpose':'training',
           'expected_cells':[], 'semantic_inputs':{'train_manifest':'data/direct/F/confirmation/manifest.json',
              'validation_manifest':'data/direct/F/validation/manifest.json','model_id':'mlp_dt','actuator':'direct',
              'condition':'F','seed':0,'loss_mode':'one_step','epochs':100,'learning_rate':.001}}
    with pytest.raises(ValueError,match='train|confirmation'):
        training.fit(job,tmp_path)


def test_training_control_command_explains_exact_model_and_actuator_arguments():
    import subprocess,sys
    result = subprocess.run([sys.executable,'-m','poc.train','controls','--help'],capture_output=True,text=True)
    assert result.returncode == 0
    assert '--model' in result.stdout and '--actuator' in result.stdout


def test_interrupted_checkpoint_write_preserves_previous_committed_optimizer_boundary(tmp_path,monkeypatch):
    training = module()
    arrays = corpus(1,8)
    session = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4)
    session.advance_updates(1)
    path = tmp_path / 'resume.pt'
    session.save(path)
    previous_hash = session.tensor_hash()
    session.advance_updates(1)
    original = training.atomic_json
    def interrupt(destination,value):
        if destination == path.with_suffix('.rng.json'):
            raise OSError('Interrupted after checkpoint rename')
        return original(destination,value)
    monkeypatch.setattr(training,'atomic_json',interrupt)
    with pytest.raises(OSError):
        session.save(path)
    restored = training.TrainingSession(build_model('mlp_dt','direct',scales(),4),arrays,scales(),seed=4)
    assert hasattr(restored,'restore_latest'), 'Previous committed checkpoint recovery is not implemented'
    chosen = restored.restore_latest(path)
    assert chosen == tmp_path / 'resume.prev.pt'
    assert restored.tensor_hash() == previous_hash and restored.updates == 1


def test_partial_training_job_selection_requires_exact_identity_and_never_uses_mtime(tmp_path):
    from poc.io import atomic_json
    import poc.training_jobs as jobs
    assert hasattr(jobs,'find_partial'), 'Compatible interrupted-job lookup is not implemented'
    first = tmp_path / 'evidence/M30/first'
    atomic_json(first / 'request.json',{'job_key':'a'*64})
    atomic_json(tmp_path / 'evidence/M30/newest/request.json',{'job_key':'b'*64})
    assert jobs.find_partial(tmp_path,'a'*64) == first
    atomic_json(tmp_path / 'evidence/M30/conflicting/request.json',{'job_key':'a'*64})
    with pytest.raises(ValueError,match='conflict|multiple'):
        jobs.find_partial(tmp_path,'a'*64)


def test_training_input_hashes_become_live_dependencies_and_missing_controls_cannot_qualify(tmp_path):
    from poc.io import sha256_file
    import poc.training_jobs as jobs
    assert hasattr(jobs,'validate_data_inputs'), 'Training input dependency validation is not implemented'
    path = tmp_path / 'corpus.npz'
    path.write_bytes(b'Original saved observations')
    hashes = {'corpus.npz':sha256_file(path)}
    assert jobs.validate_data_inputs(tmp_path,hashes) == hashes
    path.write_bytes(b'Modified observations')
    with pytest.raises(ValueError,match='hash'):
        jobs.validate_data_inputs(tmp_path,hashes)
    assert hasattr(module(),'qualified_learning_control'), 'Required learning-control closure is not implemented'
    with pytest.raises(ValueError,match='control'):
        module().qualified_learning_control(tmp_path,'mlp_dt','direct')
