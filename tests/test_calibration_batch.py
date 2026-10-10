from pathlib import Path
import copy
import pytest
from poc.io import read_json
from poc.calibration_batch import validate_registry


def test_registry_contains_exact_frozen_original_tasks_and_requests():
    root = Path.cwd()
    registry = read_json(root / 'config/control_calibration_jobs.json')
    assert len(validate_registry(root, registry)) == 96


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'identity'])
def test_mutated_batch_cannot_launch_even_its_first_job(mutation):
    registry = copy.deepcopy(read_json(Path('config/control_calibration_jobs.json')))
    if mutation == 'missing': registry['jobs'].pop()
    if mutation == 'duplicate': registry['jobs'].append(copy.deepcopy(registry['jobs'][0]))
    if mutation == 'hash': registry['jobs'][0]['sha256'] = '0' * 64
    if mutation == 'identity': registry['jobs'][0]['episode_id'] = 1
    with pytest.raises(ValueError): validate_registry(Path.cwd(), registry)
