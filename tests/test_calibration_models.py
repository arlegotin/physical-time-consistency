"""Public comparator recipes preserve priors, paired seeds and solver work."""
import json

import pytest
import torch

from poc.calibration_models import MODEL_IDS, PhysicsResidual, build_model, model_metadata
from poc.models.common import Prediction, Scales
from poc.physics_fit import PhysicsFit


FITTED_IDS = ('affine_flow', 'affine_rk4', 'node_structured', 'node_quadrature',
              'node_actuator_aware', 'physics_fit', 'physics_residual')
NOMINAL = [.10, .02, .02, .05, .05]


def scales(dtype=torch.float32):
    return Scales(torch.tensor([.03, .05], dtype=dtype),
                  torch.tensor([.2, .3], dtype=dtype),
                  torch.tensor([.7, .9], dtype=dtype))


def query(dtype=torch.float32):
    return (torch.tensor([[.2, -.7, .4, -.3, .6, -.5],
                          [-.4, .1, -.2, .5, -.3, .7]], dtype=dtype),
            torch.tensor([[-.4, .8], [.9, -.6]], dtype=dtype))


@pytest.mark.parametrize('model_id', FITTED_IDS + ('physics_nominal',))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_registry_constructs_fresh_lagged_predictors_with_valid_work(model_id, dtype):
    assert MODEL_IDS == FITTED_IDS
    model = build_model(model_id, scales(dtype), 710)
    fresh = build_model(model_id, scales(dtype), 710)
    assert model is not fresh and model.actuator == 'lagged'
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), fresh.parameters()))
    state, command = query(dtype)
    before = state.clone()
    packet = model.predict(state, command, .04)
    assert isinstance(packet, Prediction)
    assert packet.next_state.shape == (2, 6) and packet.next_state.dtype == dtype
    assert packet.status == ('ok', 'ok')
    expected_w = state[:, 4:] + (command - state[:, 4:]) * (1 - torch.exp(torch.tensor(-.5, dtype=dtype)))
    torch.testing.assert_close(packet.next_state[:, 4:], expected_w)
    assert torch.equal(state, before)
    assert packet.scalar_candidate_evaluations == 2


def test_registry_rejects_unregistered_names():
    with pytest.raises(ValueError, match='Unknown'):
        build_model('node', scales(), 710)


@pytest.mark.parametrize('model_id', FITTED_IDS + ('physics_nominal',))
def test_construction_preserves_the_callers_rng(model_id):
    torch.manual_seed(718)
    before = torch.random.get_rng_state().clone()
    build_model(model_id, scales(), 710)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize('family', [('affine_flow', 'affine_rk4'),
                                    ('node_structured', 'node_quadrature', 'node_actuator_aware')])
def test_family_seed_pairs_network_tensors_without_aliasing(family):
    models = [build_model(model_id, scales(), 710) for model_id in family]
    reference = models[0].network.state_dict()
    for model in models[1:]:
        assert reference.keys() == model.network.state_dict().keys()
        for name, value in model.network.state_dict().items():
            assert torch.equal(reference[name], value)
            assert reference[name].data_ptr() != value.data_ptr()
    other_seed = build_model(family[0], scales(), 711)
    assert any(not torch.equal(value, other_seed.network.state_dict()[name])
               for name, value in reference.items())


def test_residual_optimizer_updates_only_the_1410_network_parameters():
    model = build_model('physics_residual', scales(), 710)
    assert isinstance(model, PhysicsResidual) and isinstance(model.nominal, PhysicsFit)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) == 1410
    assert {id(p) for p in trainable} == {id(p) for p in model.network.parameters()}
    assert all(not p.requires_grad for p in model.nominal.parameters())
    torch.testing.assert_close(model.nominal.physical_parameters(), torch.tensor(NOMINAL))
    before = model.nominal.raw.detach().clone()
    optimizer = torch.optim.Adam(trainable, lr=.001)
    state, command = query()
    model.predict(state, command, .04).next_state[:, :4].square().sum().backward()
    optimizer.step()
    assert torch.equal(model.nominal.raw, before) and model.nominal.raw.grad is None


def test_nominal_reference_is_unfitted_and_physics_fit_remains_trainable():
    nominal = build_model('physics_nominal', scales(), 710)
    fitted = build_model('physics_fit', scales(), 710)
    assert sum(p.numel() for p in fitted.parameters() if p.requires_grad) == 5
    assert not any(p.requires_grad for p in nominal.parameters())
    torch.testing.assert_close(fitted.physical_parameters(), torch.tensor(NOMINAL))
    torch.testing.assert_close(nominal.physical_parameters(), torch.tensor(NOMINAL))


def test_zero_residual_matches_nominal_at_the_same_rk4_resolution():
    model = PhysicsResidual(scales(torch.float64))
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
    state, command = query(torch.float64)
    for h in (.005, .04, .08):
        actual = model.predict(state, command, h)
        expected = model.nominal.predict(state, command, h, integration_step_s=h)
        torch.testing.assert_close(actual.next_state, expected.next_state, rtol=0, atol=0)


def test_residual_uses_dense_effort_features_and_physical_acceleration_scaling():
    model = PhysicsResidual(scales(torch.float64))
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.zero_()
        model.network[0].weight[0, 6] = 1
        model.network[0].weight[1, 7] = 1
        model.network[2].weight[0, 0] = 1
        model.network[2].weight[1, 1] = 1
        model.network[4].weight[0, 1] = 1
        model.network[4].weight[1, 0] = 1
    state, _ = query(torch.float64)
    effort = torch.tensor([[.6, -.8], [.4, 1.]], dtype=torch.float64)
    correction = model.acceleration(state[:, :4], effort) - model.nominal.acceleration(state[:, :4], effort)
    expected = torch.tanh(torch.tanh(effort / 2)).flip(1) * torch.tensor([5., 7.5], dtype=torch.float64)
    torch.testing.assert_close(correction, expected, rtol=1e-13, atol=1e-13)


def test_residual_prediction_gradients_reach_each_output_and_record_real_calls():
    model = build_model('physics_residual', scales(), 710)
    state, command = query()
    command.requires_grad_()
    calls = []
    handle = model.network.register_forward_hook(lambda *args: calls.append(1))
    try:
        packet = model.predict(state, command, .04)
    finally:
        handle.remove()
    assert len(calls) == packet.network_calls == packet.vector_field_calls == 4
    packet.next_state[:, :4].square().sum().backward()
    assert (model.network[-1].weight.grad.abs().sum(1) > 0).all()
    assert (model.network[-1].bias.grad.abs() > 0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.network.parameters())
    assert command.grad is not None and torch.isfinite(command.grad).all()


def test_residual_zero_duration_and_invalid_queries():
    model = PhysicsResidual(scales())
    state, command = query()
    packet = model.predict(state, command, 0.)
    assert torch.equal(packet.next_state, state) and packet.next_state.data_ptr() != state.data_ptr()
    assert packet.network_calls == packet.vector_field_calls == 0
    mixed = model.predict(state, command, torch.tensor([[0.], [.04]]))
    assert torch.equal(mixed.next_state[0], state[0]) and mixed.network_calls == 4
    for h in (-.04, float('nan')):
        with pytest.raises(ValueError):
            model.predict(state, command, h)
    with pytest.raises(ValueError):
        model.predict(state[:, :4], command, .04)
    with pytest.raises(ValueError):
        model.predict(state, command, .04, power_mode='fast')


@pytest.mark.parametrize('model_id,calls,fields,parameters', [
    ('affine_flow', 2, 2, 18182), ('affine_rk4', 4, 4, 18182),
    ('node_structured', 4, 4, 17922), ('node_quadrature', 2, 2, 17922),
    ('node_actuator_aware', 2, 2, 17922), ('physics_fit', 0, 128, 5),
    ('physics_residual', 4, 4, 1410), ('physics_nominal', 0, 128, 0)])
def test_metadata_exposes_solver_work_and_trainable_counts(model_id, calls, fields, parameters):
    model = build_model(model_id, scales(), 710)
    metadata = model_metadata(model)
    json.dumps(metadata, allow_nan=False)
    assert metadata['model_id'] == model_id
    assert metadata['actuator'] == 'lagged' and metadata['tau_s'] == .08
    assert metadata['parameter_counts']['trainable'] == parameters
    assert metadata['field_work']['network_calls_per_40ms_query'] == calls
    assert metadata['field_work']['vector_field_calls_per_40ms_query'] == fields
    state, command = query()
    packet = model.predict(state, command, .04)
    assert packet.network_calls == calls and packet.vector_field_calls == fields
    if model_id.startswith('physics'):
        assert metadata['nominal_prior_parameters'] == NOMINAL
    if model_id == 'physics_residual':
        assert metadata['architecture'] == {'input_features': 8, 'hidden_widths': [32, 32],
                                            'output_features': 2, 'activation': 'tanh'}
        assert metadata['parameter_counts'] == {'total': 1415, 'trainable': 1410, 'frozen': 5}
        assert metadata['solver']['steps_per_interval'] == 1
    if model_id in ('physics_fit', 'physics_nominal'):
        assert metadata['solver']['default_max_step_s'] == .00125
        assert 'ceil' in metadata['field_work']['vector_field_calls_rule']


def test_metadata_rejects_models_outside_the_registry():
    with pytest.raises(ValueError, match='Unsupported'):
        model_metadata(torch.nn.Linear(2, 2))


HYBRID_IDS = ('hybrid_affine_flow', 'hybrid_affine_rk4', 'hybrid_affine_aware',
              'hybrid_accel_rk4', 'hybrid_accel_aware')


@pytest.mark.parametrize('family,count,inputs,outputs', [
    (HYBRID_IDS[:3], 1478, 6, 6), (HYBRID_IDS[3:], 1410, 8, 2)])
def test_hybrid_families_pair_initial_tensors_frozen_prior_and_zero_outputs(family, count, inputs, outputs):
    torch.manual_seed(718)
    before = torch.random.get_rng_state().clone()
    models = [build_model(model_id, scales(), 710) for model_id in family]
    assert torch.equal(torch.random.get_rng_state(), before)
    reference = models[0].network.state_dict()
    for model, model_id in zip(models, family):
        metadata = model_metadata(model)
        assert metadata['model_id'] == model_id and metadata['nominal_physics_frozen']
        assert metadata['architecture'] == {'input_features': inputs, 'hidden_widths': [32, 32],
                                             'output_features': outputs, 'activation': 'tanh'}
        assert metadata['parameter_counts'] == {'total': count + 5, 'trainable': count, 'frozen': 5}
        assert not any(p.requires_grad for p in model.nominal.parameters())
        torch.testing.assert_close(model.nominal.physical_parameters(), torch.tensor(NOMINAL))
        assert torch.count_nonzero(model.network[-1].weight) == 0
        assert torch.count_nonzero(model.network[-1].bias) == 0
        for name, value in model.network.state_dict().items():
            assert torch.equal(reference[name], value)
            if model is not models[0]:
                assert reference[name].data_ptr() != value.data_ptr()
        state, command = query()
        packet = model.predict(state, command, .04)
        assert metadata['field_work']['network_calls_per_40ms_query'] == packet.network_calls
        assert metadata['field_work']['vector_field_calls_per_40ms_query'] == packet.vector_field_calls
        json.dumps(metadata, allow_nan=False)
    other_seed = build_model(family[0], scales(), 711)
    assert not torch.equal(models[0].network[0].weight, other_seed.network[0].weight)


def test_new_registry_and_propagation_families_leave_legacy_recipe_intact():
    import poc.calibration_models as registry
    assert getattr(registry, 'HYBRID_MODEL_IDS', ()) == HYBRID_IDS
    assert registry.ALL_MODEL_IDS == FITTED_IDS + HYBRID_IDS
    assert registry.MODEL_IDS == FITTED_IDS
    assert registry.propagation_family('affine_flow') == ('affine_flow', 'affine_rk4')
    assert registry.propagation_family('affine_rk4') == ('affine_flow', 'affine_rk4')
    for model_id in HYBRID_IDS[:3]:
        assert registry.propagation_family(model_id) == HYBRID_IDS[:3]
    for model_id in HYBRID_IDS[3:] + ('physics_nominal', 'physics_residual', 'node_structured', 'unknown'):
        assert registry.propagation_family(model_id) == ()
