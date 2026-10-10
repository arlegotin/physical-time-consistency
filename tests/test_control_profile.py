import numpy as np
import torch
import pytest
from poc.control_profile import prediction_latency
from poc.models.common import prediction


class Model:
    def __init__(self, fail=False): self.calls, self.fail = 0, fail
    def predict(self, state, command, h, history=None):
        self.calls += 1
        output = state.clone()
        if self.fail: output[:] = float('nan')
        return prediction(output, 2)


def test_prediction_profile_counts_both_head_passes_and_excludes_warmup_from_samples():
    model = Model()
    ticks = iter(range(0, 100_000, 1000))
    result = prediction_latency(model, torch.zeros((1, 4)), torch.zeros((1, 2)), .005,
                                warmup=2, repeats=4, timer=lambda: next(ticks))
    assert model.calls == 6
    assert result['samples'] == 4
    assert result['network_calls_per_prediction'] == 2
    np.testing.assert_allclose(result['elapsed_seconds'], [1e-6] * 4)
    assert result['median_seconds'] == pytest.approx(1e-6)


def test_failed_model_profile_cannot_hide_bad_calls_in_latency_summary():
    with pytest.raises(ValueError, match='failed'):
        prediction_latency(Model(fail=True), torch.zeros((1, 4)), torch.zeros((1, 2)), .005,
                           warmup=1, repeats=3)
