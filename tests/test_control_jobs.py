"""Only explicit frozen single-episode jobs may reach robot control."""
import pytest


def request():
    return {'schema_version': 1, 'phase': 'calibration', 'model_id': 'sim_oracle', 'actuator': 'direct',
            'seed': 9000, 'episode_id': 0, 'condition': 'clean', 'model_partition_ns': [40_000_000],
            'actuator_interval_ns': 40_000_000, 'physics_step_ns': 250_000,
            'power_mode': None, 'model_manifest': None, 'manifest_sha256': None, 'checkpoint_sha256': None,
            'planner_settings': {'candidate_count': 128, 'iterations': 4, 'elite_count': 16,
                                 'initial_std': 1., 'std_floor': .1},
            'resource_projection': {'wall_seconds': None, 'disk_bytes': 10_000_000,
                                    'reason': 'Representative episode measurement before full batch'}}


def test_single_episode_request_keeps_all_physical_clocks_and_complete_identity():
    from poc.control_jobs import validate_request
    item = request()
    assert validate_request(item) == item
    item['model_partition_ns'] = [5_000_000] * 8
    assert validate_request(item)['actuator_interval_ns'] == 40_000_000


@pytest.mark.parametrize('mutation', ['partial_horizon', 'discovery_confirmation', 'extra_observations', 'missing_hash', 'unknown'])
def test_bad_requests_fail_before_model_or_plant_access(mutation):
    from poc.control_jobs import validate_request
    item = request()
    if mutation == 'partial_horizon':
        item['model_partition_ns'] = [5_000_000] * 7
    elif mutation == 'discovery_confirmation':
        item['phase'] = 'confirmatory'
    elif mutation == 'extra_observations':
        item['observation_interval_ns'] = 5_000_000
    elif mutation == 'missing_hash':
        item['model_id'] = 'power_correct'
        item['model_manifest'] = 'evidence/M30/selected/manifest.json'
    else:
        item['model_id'] = 'future_model'
    with pytest.raises(ValueError):
        validate_request(item)


def test_true_qualification_flags_do_not_bypass_stale_evidence(monkeypatch):
    from poc.control_jobs import main
    def stale(root):
        raise ValueError('stale qualification evidence fixture')
    monkeypatch.setattr('poc.planner_qualification.verify_frozen', stale)
    with pytest.raises(ValueError, match='stale qualification evidence fixture'):
        main(['config/control_jobs/development__direct__A__shared__power_correct__default__h5000000__episode0.json', '--dry-run'])
