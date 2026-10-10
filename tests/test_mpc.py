"""Search mechanics and predictions checked against independent known answers."""
import numpy as np
import pytest
import torch

from poc.action_path import compile_knots
from poc.control import anchor_cost
from poc.history import HistoryBatch, HistoryToken
from poc.models.common import Prediction
from poc import mpc


class ExactIntegrator:
    def __init__(self):
        self.contexts = []

    def predict(self, state, command, h_s, history=None):
        self.contexts.append(history)
        q, v = state[:, :2], state[:, 2:4]
        value = torch.cat((q + h_s * v + .5 * h_s**2 * command, v + h_s * command), 1)
        return Prediction(value, ('ok',) * len(state), 0, 0, len(state), 0)


def settings(candidates=8, iterations=2):
    return {'candidate_count': candidates, 'iterations': iterations, 'elite_count': candidates // 8,
            'initial_std': 1., 'std_floor': .10}


def independent_rollout(state, knots, interval):
    x, anchors = np.array(state, copy=True), []
    for segment in compile_knots(knots, interval):
        h, u = segment.duration_ns / 1e9, segment.command
        q, v = x[:2].copy(), x[2:].copy()
        x = np.r_[q + h * v + .5 * h * h * u, v + h * u]
        if (segment.start_ns + segment.duration_ns) % 40_000_000 == 0:
            anchors.append(x.copy())
    return np.stack(anchors)


@pytest.mark.parametrize('interval', [40_000_000, 20_000_000, 10_000_000])
@pytest.mark.parametrize('step', [40_000_000, 5_000_000])
def test_candidate_prediction_has_same_path_and_six_cost_anchors(interval, step):
    model = ExactIntegrator()
    state, target = torch.tensor([.2, -.3, .1, -.2], dtype=torch.float64), np.array([.5, -.2])
    bank = np.random.default_rng(4).uniform(-2, 2, (3, 6, 2))
    result = mpc.rollout_candidates(model, state, target, None, bank,
                                    (step,) * (40_000_000 // step), interval)
    assert result['statuses'] == ('ok',) * 3
    assert result['anchors'].shape == (3, 6, 4)
    for index, knots in enumerate(bank):
        expected = independent_rollout(state.numpy(), knots, interval)
        np.testing.assert_allclose(result['anchors'][index], expected, atol=1e-14)
        assert result['costs'][index] == pytest.approx(anchor_cost(expected, target, compile_knots(knots, interval)))
    # Physical history commits occur at macro anchors, never model substeps.
    for context in model.contexts:
        time = int(context.query_time_ns[0])
        for sequence in context.tokens:
            assert len(sequence) == time // 40_000_000
            assert all(t.duration_ns == 40_000_000 and t.provenance == 'imagined' for t in sequence)


def test_candidate_order_and_repeated_queries_preserve_real_context():
    token = HistoryToken(0, np.zeros(4), np.zeros(2), 40_000_000, 'observed', 'real-0')
    history = HistoryBatch(((token,),), np.array([40_000_000], dtype=np.int64))
    model, state = ExactIntegrator(), torch.zeros(4, dtype=torch.float64)
    bank = np.random.default_rng(8).uniform(-2, 2, (3, 6, 2))
    first = mpc.rollout_candidates(model, state, np.zeros(2), history, bank, (5_000_000,) * 8, 40_000_000)
    order = [2, 0, 1]
    second = mpc.rollout_candidates(model, state, np.zeros(2), history, bank[order], (5_000_000,) * 8, 40_000_000)
    np.testing.assert_array_equal(first['anchors'][order], second['anchors'])
    assert history.tokens == ((token,),)
    assert history.query_time_ns[0] == 40_000_000


def test_scalar_and_batched_candidate_predictions_agree():
    bank = np.random.default_rng(17).uniform(-2, 2, (3, 6, 2))
    state, target = torch.tensor([.1, -.1, .2, 0.], dtype=torch.float64), np.array([.2, .3])
    batched = mpc.rollout_candidates(ExactIntegrator(), state, target, None, bank,
                                     (5_000_000,) * 8, 20_000_000)
    for i in range(3):
        scalar = mpc.rollout_candidates(ExactIntegrator(), state, target, None, bank[i:i+1],
                                        (5_000_000,) * 8, 20_000_000)
        np.testing.assert_array_equal(batched['anchors'][i], scalar['anchors'][0])
        assert batched['costs'][i] == scalar['costs'][0]


def stub_evaluator(cost_fn):
    def evaluate(model, state, target, history, knots, partition, actuator_interval):
        costs = np.asarray(cost_fn(knots), dtype=float)
        count = len(knots)
        return {'costs': costs, 'anchors': np.zeros((count, 6, 4)),
                'statuses': tuple('ok' if np.isfinite(c) else 'model_failure' for c in costs),
                'reasons': tuple('' if np.isfinite(c) else 'fixture failure' for c in costs),
                'path_checksums': tuple('fixture' for _ in costs),
                'work': {'network_calls': 1, 'vector_field_calls': 0,
                         'scalar_candidate_evaluations': count, 'history_tokens_processed': 0}}
    return evaluate


def search(normals, previous=None):
    return mpc.search(None, torch.zeros(4), torch.zeros(2), None, previous, normals,
                      settings(normals.shape[1], normals.shape[0]), (40_000_000,), 40_000_000)


def test_cem_uses_evaluated_candidates_fixed_normal_ids_population_std_and_floor(monkeypatch):
    monkeypatch.setattr(mpc, 'rollout_candidates', stub_evaluator(lambda k: np.sum(k * k, axis=(1, 2))))
    normals = np.random.default_rng(9).normal(size=(2, 8, 6, 2))
    old = np.arange(12, dtype=float).reshape(6, 2) / 10
    result = search(normals, old)
    candidates = result['candidates']
    np.testing.assert_array_equal(candidates['knots'][0], np.r_[old[1:], old[-1:]])
    np.testing.assert_array_equal(candidates['knots'][1], np.zeros((6, 2)))
    np.testing.assert_array_equal(candidates['knots'][2], np.clip(np.r_[old[1:], old[-1:]] + normals[0, 2], -2, 2))
    np.testing.assert_array_equal(result['proposals'][0]['next_mean'], np.zeros((6, 2)))
    np.testing.assert_array_equal(result['proposals'][0]['next_std'], np.full((6, 2), .1))
    assert candidates['global_ids'].tolist() == list(range(16))
    assert result['selected_global_id'] == 1
    assert result['cost'] == 0
    np.testing.assert_array_equal(result['knots'], candidates['knots'][result['selected_global_id']])


def test_multiple_elites_use_population_standard_deviation_without_smoothing(monkeypatch):
    monkeypatch.setattr(mpc, 'rollout_candidates', stub_evaluator(lambda k: np.sum(k * k, axis=(1, 2))))
    normals = np.ones((1, 16, 6, 2))
    normals[0, 2] = .2
    result = search(normals, np.full((6, 2), .4))
    assert result['proposals'][0]['elite_global_ids'] == [1, 0]
    np.testing.assert_allclose(result['proposals'][0]['next_mean'], .2)
    np.testing.assert_allclose(result['proposals'][0]['next_std'], .2)


def test_equal_costs_choose_earlier_iteration_then_lower_id(monkeypatch):
    monkeypatch.setattr(mpc, 'rollout_candidates', stub_evaluator(lambda k: np.ones(len(k))))
    result = search(np.zeros((2, 8, 6, 2)))
    assert result['selected_global_id'] == 0


def test_partial_and_all_candidate_failures_remain_logged(monkeypatch):
    monkeypatch.setattr(mpc, 'rollout_candidates', stub_evaluator(
        lambda k: [1.] + [np.inf] * (len(k) - 1)))
    result = search(np.zeros((2, 8, 6, 2)))
    assert result['status'] == 'ok'
    assert np.isinf(result['candidates']['costs']).sum() == 14
    assert all(p['elite_global_ids'] == [p['iteration'] * 8] for p in result['proposals'])
    monkeypatch.setattr(mpc, 'rollout_candidates', stub_evaluator(lambda k: [np.inf] * len(k)))
    result = search(np.zeros((2, 8, 6, 2)))
    assert result['status'] == 'all_candidates_failed'
    assert result['fallback'] == 'zero_command'
    assert result['cost'] is None
    np.testing.assert_array_equal(result['knots'], np.zeros((6, 2)))
    assert len(result['candidates']['costs']) == 16


def test_nonfinite_or_runaway_model_rows_fail_without_poisoning_other_candidates():
    class Failing(ExactIntegrator):
        def predict(self, state, command, h_s, history=None):
            packet = super().predict(state, command, h_s, history)
            value = packet.next_state.clone()
            value[command[:, 0] > 0, 0] = float('nan')
            value[command[:, 1] > 0, 2] = 1001
            return Prediction(value, packet.status, 0, 0, len(state), 0)
    bank = np.zeros((3, 6, 2))
    bank[1, :, 0], bank[2, :, 1] = 1, 1
    result = mpc.rollout_candidates(Failing(), torch.zeros(4), np.zeros(2), None,
                                    bank, (5_000_000,) * 8, 40_000_000)
    assert result['statuses'] == ('ok', 'model_failure', 'model_failure')
    assert np.isfinite(result['costs'][0]) and np.isinf(result['costs'][1:]).all()
    assert result['reasons'][1] and result['reasons'][2]


@pytest.mark.parametrize('interval', [40_000_000, 20_000_000, 10_000_000])
def test_box_optimum_is_bounded_qualified_and_matches_independent_rollout(interval):
    state, target = np.array([.2, -.4, .1, -.1]), np.array([-.2, .1])
    result = mpc.analytic_box_optimum(state, target, interval)
    assert result['success']
    assert np.max(np.abs(result['knots'])) <= 2
    assert result['projected_optimality'] <= 1e-6
    anchors = independent_rollout(state, result['knots'], interval)
    observed = anchor_cost(anchors, target, compile_knots(result['knots'], interval))
    assert observed == pytest.approx(result['cost'], rel=1e-10, abs=1e-12)
    assert observed == pytest.approx(2 * result['solver_half_squared_cost'], rel=1e-10, abs=1e-12)
    assert result['cost'] <= result['zero_cost']


def test_wrong_model_partitions_and_incomplete_random_banks_fail():
    with pytest.raises(ValueError):
        mpc.rollout_candidates(ExactIntegrator(), torch.zeros(4), np.zeros(2), None,
                               np.zeros((2, 6, 2)), (5_000_000,), 40_000_000)
    with pytest.raises(ValueError):
        search(np.zeros((2, 8, 5, 2)))


def test_power_modes_use_shared_weights_and_produce_same_nominal_search():
    from poc.models.common import Scales
    from poc.models.factory import build_model
    scales = Scales(torch.ones(2), torch.ones(2), torch.ones(2))
    model = build_model('power_correct', 'direct', scales, seed=8100).double()
    normals = np.random.default_rng(32).normal(size=(1, 8, 6, 2))
    state, target = torch.zeros(4, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    results = [mpc.search(model, state, target, None, None, normals, settings(8, 1),
                           (40_000_000,), 40_000_000, power_mode=mode)
               for mode in ('low', 'direct', 'lagged')]
    for result in results[1:]:
        np.testing.assert_allclose(result['candidates']['costs'], results[0]['candidates']['costs'],
                                   atol=1e-12, rtol=1e-12)
        assert result['selected_global_id'] == results[0]['selected_global_id']
        np.testing.assert_array_equal(result['knots'], results[0]['knots'])
