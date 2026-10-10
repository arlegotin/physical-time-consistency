"""Control reloads exact selected tensors and keeps oracle branches independent."""
from pathlib import Path
import numpy as np
import pytest
import torch

from poc.io import atomic_json, sha256_file
from poc.models.common import Scales
from poc.models.factory import build_model
from poc.reference_jobs import save_tensors, tensor_hash


def test_oracle_matches_independent_plant_branches_without_mutating_query():
    from poc.control_models import SimOracle
    from poc.envs import Plant
    model = SimOracle('lagged', 250_000, Path('assets'))
    states = torch.tensor([[.3, -.5, 1., -.2, .4, -.7], [-.2, .1, -.3, .7, -.1, .2]], dtype=torch.float64)
    commands = torch.tensor([[.8, -.9], [-.5, .3]], dtype=torch.float64)
    saved = states.clone()
    result = model.predict(states, commands, .005)
    truth = Plant('lagged', 250_000)
    for index in range(2):
        truth.reset(states[index].numpy())
        truth.advance(commands[index].numpy(), 5_000_000)
        np.testing.assert_array_equal(result.next_state[index].numpy(), truth.state())
    assert torch.equal(saved, states)
    assert model.physics_steps == 40
    assert result.network_calls == result.vector_field_calls == 0
    reordered = model.predict(states.flip(0), commands.flip(0), .005)
    torch.testing.assert_close(result.next_state, reordered.next_state.flip(0), rtol=0, atol=0)


def test_simulator_warning_invalidates_reference_instead_of_becoming_model_fallback(monkeypatch):
    from poc.control_models import SimOracle, PlantReferenceError
    from poc.mpc import rollout_candidates
    model = SimOracle('direct', 250_000, Path('assets'))
    def failed_advance(*args, **kwargs):
        raise RuntimeError('simulator warning fixture')
    monkeypatch.setattr(model.plant, 'advance', failed_advance)
    with pytest.raises(PlantReferenceError, match='simulator warning fixture'):
        rollout_candidates(model, torch.zeros(4, dtype=torch.float64), [0., 0.],
                           None, np.zeros((2, 6, 2)), (40_000_000,), 40_000_000)
    # The producer envelope catches AssertionError as invalid evidence;
    # MPC catches only RuntimeError/ValueError as intrinsic model failure.
    assert issubclass(PlantReferenceError, AssertionError)
    assert not issubclass(PlantReferenceError, (RuntimeError, ValueError))


def test_selected_model_reload_rejects_changed_weights_and_training_identity(tmp_path, monkeypatch):
    from poc.control_models import load_selected
    monkeypatch.setattr('poc.control_models.verify_run', lambda root, record: None)
    folder = tmp_path / 'evidence/M30/unit'
    folder.mkdir(parents=True)
    scale = Scales(torch.ones(2), torch.ones(2), torch.ones(2))
    original = build_model('power_correct', 'direct', scale, 9000)
    save_tensors(folder / 'weights.pt', original.state_dict())
    atomic_json(folder / 'normalization.json', {'scale_q': [1., 1.], 'scale_v': [1., 1.], 'velocity_input_rms': [1., 1.]})
    metadata = {'milestone_id': 'M30', 'completion_status': 'complete', 'phase': 'unit',
                'job_inputs': {'parameters': {'model_id': 'power_correct', 'actuator': 'direct', 'seed': 9000}},
                'model_hashes': {'canonical_tensor_sha256': tensor_hash(original.state_dict())}}
    atomic_json(folder / 'manifest.json', metadata)
    manifest_hash = sha256_file(folder / 'manifest.json')
    weights_hash = sha256_file(folder / 'weights.pt')
    restored = load_selected(folder / 'manifest.json', manifest_hash, weights_hash,
                             model_id='power_correct', actuator='direct', seed=9000)
    x, u = torch.zeros((1, 4), dtype=torch.float64), torch.ones((1, 2), dtype=torch.float64)
    torch.testing.assert_close(restored.predict(x, u, .005).next_state,
                               original.double().predict(x, u, .005).next_state, rtol=0, atol=0)
    with pytest.raises(ValueError, match='identity'):
        load_selected(folder / 'manifest.json', manifest_hash, weights_hash,
                      model_id='power_correct', actuator='direct', seed=0)
    with (folder / 'weights.pt').open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        load_selected(folder / 'manifest.json', manifest_hash, weights_hash,
                      model_id='power_correct', actuator='direct', seed=9000)
