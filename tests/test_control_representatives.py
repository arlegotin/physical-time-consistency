import copy
from pathlib import Path

import pytest

from poc.io import read_json
from poc.control_representatives import validate_jobs


def test_exact_three_models_and_two_partitions_are_prospective():
    root = Path.cwd()
    registry = read_json(root / 'config/control_development_representative_jobs.json')
    assert len(validate_jobs(root, registry)) == 6


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'hash', 'design'])
def test_unverified_representative_workload_cannot_start(mutation):
    root = Path.cwd()
    registry = copy.deepcopy(read_json(root / 'config/control_development_representative_jobs.json'))
    if mutation == 'missing': registry['jobs'].pop()
    if mutation == 'duplicate': registry['jobs'][-1] = copy.deepcopy(registry['jobs'][0])
    if mutation == 'hash': registry['jobs'][0]['sha256'] = '0' * 64
    if mutation == 'design': registry['design_sha256'] = '0' * 64
    with pytest.raises(ValueError): validate_jobs(root, registry)
