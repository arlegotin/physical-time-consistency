import copy
from pathlib import Path

import pytest
from poc.development_batch import validate_jobs
from poc.io import read_json


def test_original_direct_core_workload_has_all_ten_tasks_and_both_partitions():
    root = Path.cwd()
    registry = read_json(root / 'config/control_development_direct_jobs.json')
    assert len(validate_jobs(root, registry)) == 60


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'design'])
def test_partial_or_changed_workload_cannot_start(mutation):
    root = Path.cwd()
    registry = copy.deepcopy(read_json(root / 'config/control_development_direct_jobs.json'))
    if mutation == 'missing': registry['jobs'].pop()
    if mutation == 'duplicate': registry['jobs'][-1] = copy.deepcopy(registry['jobs'][0])
    if mutation == 'hash': registry['jobs'][0]['sha256'] = '0' * 64
    if mutation == 'design': registry['design_sha256'] = '0' * 64
    with pytest.raises(ValueError): validate_jobs(root, registry)
