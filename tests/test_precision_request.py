"""A continuation is prospective, paired and confined to the existing design."""
import importlib
import pytest


def api():
    return importlib.import_module('poc.precision_request').build_request


def inputs():
    parent = {'decision_id': 'parent', 'gate_id': 'GR', 'outcome': 'INCONCLUSIVE',
              'validity_status': 'valid', 'phase': 'confirmatory', 'look_index': 3,
              'permitted_next_actions': [{'milestone_id': 'MX', 'branch': 'precision'}]}
    scope = {'seed_ids': list(range(20)), 'negative_only': True, 'statistical_family_size': 280,
             'actuator': 'direct', 'condition': 'A', 'loss_mode': 'shared', 'competitor': 'node_generic'}
    evidence = {'path': 'evidence/MR/complete/manifest.json', 'sha256': 'a'*64,
                'role': 'manifest', 'producing_run_id': 'complete'}
    resources = {'measured_new_training_attempts': 30, 'measured_new_training_wall_seconds': 5400,
                 'measured_peak_rss_bytes': 700000000}
    return parent, scope, evidence, resources


def test_twenty_to_forty_preserves_prior_seeds_look_and_effect_floor():
    parent, scope, evidence, resources = inputs()
    request = api()(parent, scope, evidence, resources, request_id='MX-precision-0007',
                    context_relative='study/working')
    assert request['action_parameters']['seed_ids'] == list(range(40))
    assert request['action_parameters']['new_seed_ids'] == list(range(20, 40))
    assert request['action_parameters']['next_look'] == 4
    assert request['action_parameters']['alpha'] == .003125
    assert request['statistical_family_size'] == 280
    assert request['resource_projection']['projected_new_training_wall_seconds'] == 10800
    assert request['resource_projection']['new_selected_training_fits'] == 40
    assert request['parent_decision_id'] == 'parent'
    assert request['prior_manifest_sha256'] == 'a'*64
    assert set(request['contract']) == {str(i) for i in range(1, 13)}
    assert all(request['contract'].values())
    assert request['status'] == 'prospective'


@pytest.mark.parametrize('change', [
    {'outcome': 'NO-GO'}, {'validity_status': 'invalid'}, {'phase': 'exploratory'},
    {'permitted_next_actions': []}])
def test_no_extra_samples_after_rejection_or_without_a_precision_transition(change):
    parent, scope, evidence, resources = inputs()
    parent.update(change)
    with pytest.raises(ValueError, match='precision'):
        api()(parent, scope, evidence, resources, request_id='MX-precision-0007',
              context_relative='study/working')


def test_missing_measured_resources_are_not_replaced_by_an_invented_forecast():
    parent, scope, evidence, resources = inputs()
    resources['measured_new_training_wall_seconds'] = None
    with pytest.raises(ValueError, match='resource'):
        api()(parent, scope, evidence, resources, request_id='MX-precision-0007',
              context_relative='study/working')
