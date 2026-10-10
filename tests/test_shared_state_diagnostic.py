"""Catch branch contamination, signed decomposition mistakes and omitted probes."""
import numpy as np
import pytest
import torch

from poc import shared_state_diagnostic as diagnostic
from poc.models.common import activation, prediction


def test_signed_errors_cancel_without_disappearing_from_components():
    # Truth zero=0, truth commanded=3. Model zero=2, commanded=3.
    # A wrong sign or addition of norms would invent total error 4.
    zero = np.zeros((1, 4))
    model_zero = np.array([[2., -2., 2., -2.]])
    truth_command = np.full((1, 4), 3.)
    values = diagnostic.decompose(model_zero, truth_command, zero, truth_command, truth_command)
    np.testing.assert_array_equal(values['drift_error'], model_zero)
    np.testing.assert_array_equal(values['command_error'], -model_zero)
    np.testing.assert_array_equal(values['total_error'], zero)
    np.testing.assert_array_equal(values['composition_discrepancy'], zero)
    np.testing.assert_array_equal(values['composed_error'], zero)


class ExactLag:
    """Physical lagged unit-mass map, recording inputs to detect wrong branches."""
    def __init__(self):
        self.calls = []

    def predict(self, state, command, h_s):
        self.calls.append((state.clone(), command.clone(), h_s))
        h = torch.as_tensor(h_s, dtype=state.dtype)
        kw = -torch.expm1(-h/.08)
        kv, kq = h-.08*kw, h*h/2-.08*h+.08*.08*kw
        q, v, w = state[:, :2], state[:, 2:4], state[:, 4:]
        result = torch.cat((q+h*v+h*h*w/2+(command-w)*kq,
                            v+h*w+(command-w)*kv, activation(state, command, h)), 1)
        return prediction(result, 0)


def test_zero_branch_keeps_initial_activation_and_composition_keeps_command():
    state = np.array([[.3, -.4, .2, -.1, .7, -.8]])
    command = np.array([[-.5, .9]])
    model = ExactLag()
    values = diagnostic.model_endpoints(model, state, command, 40_000_000)
    np.testing.assert_array_equal(model.calls[0][0], state)
    np.testing.assert_array_equal(model.calls[1][0], state)
    np.testing.assert_array_equal(model.calls[0][1], np.zeros((1, 2)))
    np.testing.assert_array_equal(model.calls[1][1], command)
    np.testing.assert_array_equal(model.calls[2][1], command)
    np.testing.assert_array_equal(model.calls[3][1], command)
    np.testing.assert_allclose(values['composed'], values['command'], atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(state, np.array([[.3, -.4, .2, -.1, .7, -.8]]))


def test_decomposition_rejects_missing_or_nonfinite_endpoint():
    values = np.zeros((2, 4))
    with pytest.raises(ValueError):
        diagnostic.decompose(values, values[:1], values, values, values)
    invalid = values.copy(); invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic.decompose(values, invalid, values, values, values)


def test_probe_selection_keeps_paired_first_trajectories_and_rejects_truncation():
    library = dict(states=np.zeros((6, 4)), commands=np.zeros((6, 2)),
                   time_ns=np.array([0, 330_000_000, 0, 340_000_000, 0, 350_000_000]),
                   strata=np.array(['cold', 'warm']*3), trajectory_ids=np.repeat([4, 5, 6], 2),
                   anchor_ids=np.array([0, 9]*3),
                   probe_ids=np.array(['cold4', 'warm4', 'cold5', 'warm5', 'cold6', 'warm6']))
    selected = diagnostic.select_probes(library, count=2)
    np.testing.assert_array_equal(selected['probe_ids'], ['cold4', 'cold5', 'warm4', 'warm5'])
    with pytest.raises(ValueError):
        diagnostic.select_probes(library, count=4)
    library['trajectory_ids'][3] = 99
    with pytest.raises(ValueError):
        diagnostic.select_probes(library, count=2)


def test_command_normalization_expands_by_both_branch_envelopes():
    errors = np.array([[3., 4.]])
    truth = np.array([[3., 4.]])
    actual = diagnostic.normalized_error(errors, truth, np.array([.04]))
    np.testing.assert_allclose(actual[0], [1.])
    np.testing.assert_allclose(actual[1], [4.96/5.04])
    np.testing.assert_allclose(actual[2], [5.04/4.96])
    with pytest.raises(ValueError, match='signal'):
        diagnostic.normalized_error(errors, np.zeros((1, 2)), np.array([.04]))


def test_reference_envelope_uses_refinement_and_independent_error():
    base = np.zeros((2, 4))
    fine = base.copy(); fine[:, 0] = 3e-8
    finest = base.copy(); finest[:, 2] = 4e-7
    oracle = base.copy(); oracle[:, 0] = 8e-8
    envelope = diagnostic.reference_envelope(base, fine, finest, oracle)
    np.testing.assert_allclose(envelope[:, 0], [8e-8, 8e-8])
    np.testing.assert_allclose(envelope[:, 1], [4e-7, 4e-7])
    oracle[:, 0] = 2e-7
    with pytest.raises(ValueError, match='budget'):
        diagnostic.reference_envelope(base, fine, finest, oracle)


def _measurement_fixture():
    zeros = np.zeros((4, 128, 3, 4))
    ref = np.zeros((128, 3, 4)); ref[:, :, :] = 1.
    raw = dict(model_ids=np.array(['power_correct','mlp_h','node_generic','node_structured']),
               horizons_ns=np.array([5_000_000, 10_000_000, 40_000_000]),
               probe_ids=np.array(['p'+str(i) for i in range(128)]),
               strata=np.array(['cold']*64+['warm']*64), model_zero=zeros.copy(),
               model_command=zeros.copy(), model_composed=zeros.copy(),
               truth_zero=np.zeros_like(ref), truth_command=ref,
               uncertainty_zero=np.zeros((128, 3, 2)), uncertainty_command=np.zeros((128, 3, 2)))
    coordinates = {key:raw[key].tolist() for key in ('probe_ids', 'strata')}
    freeze = {'coordinates':{'direct':coordinates}}
    return raw, freeze


def test_raw_summary_rejects_missing_baseline_and_truncated_endpoint():
    raw, freeze = _measurement_fixture()
    summary = diagnostic.summarize_measurements(raw, freeze, 'direct')
    assert len(summary['rows']) == 48
    raw['model_ids'] = raw['model_ids'][:3]
    with pytest.raises(ValueError, match='coordinate'):
        diagnostic.summarize_measurements(raw, freeze, 'direct')
    raw, freeze = _measurement_fixture()
    raw['model_composed'] = raw['model_composed'][:, :127]
    with pytest.raises(ValueError, match='endpoint'):
        diagnostic.summarize_measurements(raw, freeze, 'direct')


def test_raw_summary_reports_cross_term_and_errors_not_just_total():
    raw, freeze = _measurement_fixture()
    raw['model_zero'][:] = 1.
    raw['model_command'][:] = 1.
    raw['model_composed'][:] = 1.
    row = diagnostic.summarize_measurements(raw, freeze, 'direct')['rows'][0]
    assert row['drift_q_squared_mean'] == 2.
    assert row['command_q_squared_mean'] == 2.
    assert row['cross_q_squared_mean'] == -4.
    assert row['total_q_squared_mean'] == 0.


def test_development_context_reuses_identical_corpus_and_rejects_changed_copy(tmp_path):
    root_corpus = tmp_path/'data/direct/A/development'
    context_corpus = tmp_path/'study/control-development/data/direct/A/development'
    for directory in (root_corpus, context_corpus):
        directory.mkdir(parents=True)
        (directory/'manifest.json').write_text('{"role":"development"}')
        (directory/'trajectories.npz').write_bytes(b'unchanged historical corpus')
    path = diagnostic.development_manifest(tmp_path,'direct','study/control-development')
    assert path == context_corpus/'manifest.json'
    (context_corpus/'trajectories.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identical'):
        diagnostic.development_manifest(tmp_path,'direct','study/control-development')


def test_verification_rejects_edited_reader_report(tmp_path):
    from poc.io import atomic_json
    result = dict(rows=[],comparisons=[],manifest_path='evidence/example/manifest.json')
    target = tmp_path/'reports/shared_state_diagnostic_result.md'
    target.parent.mkdir()
    atomic_json(tmp_path/'reports/shared_state_diagnostic_result.json',result)
    target.write_bytes(diagnostic.render_report(result))
    diagnostic.verify_saved_reports(tmp_path,result)
    target.write_bytes(target.read_bytes()+b'An invented result.\n')
    with pytest.raises(ValueError, match='Markdown'):
        diagnostic.verify_saved_reports(tmp_path,result)
