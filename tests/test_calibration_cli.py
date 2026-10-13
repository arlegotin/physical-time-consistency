"""Study CLI uses public assets and explicitly labeled disposable unit stores."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def study_module():
    assert importlib.util.find_spec('poc.calibration_study'), 'Study CLI is missing'
    from poc import calibration_study
    return calibration_study


@pytest.fixture
def public_root(tmp_path):
    root = tmp_path / 'public-checkout'
    shutil.copytree(ROOT / 'poc', root / 'poc', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT / 'assets', root / 'assets')
    return root


def test_prepare_freezes_unit_recipe_and_expansion_reuses_fit_identity(public_root, tmp_path):
    study = study_module()
    output = tmp_path / 'out'
    pilot = study.prepare(public_root, output, unit_fixture={'segment_count': 600})
    assert pilot == study.prepare(public_root, output, unit_fixture={'segment_count': 600})
    assert pilot['recipe']['phase'] == 'unit'
    assert pilot['recipe']['collector']['segment_count'] == 600
    assert pilot['recipe']['metrics']['rmse'] == 'sqrt(mean squared component error over endpoints and both joints)'
    assert pilot['recipe']['metrics']['p95'] == '95th percentile of two-joint Euclidean endpoint-error norms'
    assert len(pilot['coordinates']) == 21
    development = study.prepare(public_root, output, 'development', unit_fixture={'segment_count': 600})
    assert len(development['coordinates']) == 195
    assert pilot['recipe_receipt'] == development['recipe_receipt']
    assert pilot['recipe'] == development['recipe']
    assert all(c in development['coordinates'] for c in pilot['coordinates'])
    assert all('matrix' not in c for c in development['coordinates'])
    assert pilot['matrix_receipt'] != development['matrix_receipt']


@pytest.mark.parametrize('change', ['source', 'asset', 'added-scientific-module', 'root', 'recipe'])
def test_stale_prepared_bindings_cannot_resume(public_root, tmp_path, change):
    study = study_module()
    output = tmp_path / 'out'
    study.prepare(public_root, output, unit_fixture={'segment_count': 5})
    if change == 'source':
        with (public_root / 'poc/calibration_models.py').open('a') as file:
            file.write('\n# unit fixture changed source\n')
    elif change == 'asset':
        with (public_root / 'assets/two_link_lagged.xml').open('a') as file:
            file.write('\n<!-- unit fixture changed asset -->\n')
    elif change == 'added-scientific-module':
        (public_root / 'poc/calibration_training.py').write_text('# later scientific module\n')
    elif change == 'root':
        other = tmp_path / 'other-checkout'
        shutil.copytree(public_root, other)
        public_root = other
    else:
        with pytest.raises(ValueError, match='binding|recipe'):
            study.prepare(public_root, output, unit_fixture={'segment_count': 10})
        return
    with pytest.raises(ValueError, match='binding|stale'):
        study.open_store(public_root, output)


def test_collect_requires_prepare_and_propagates_output_root(public_root, tmp_path):
    study = study_module()
    output = tmp_path / 'chosen'
    with pytest.raises(ValueError, match='prepare'):
        study.collect_session(0, root=public_root, output_root=output)
    prepared = study.prepare(public_root, output, unit_fixture={'segment_count': 5})
    receipt = study.collect_session(0, root=public_root, output_root=output)
    assert receipt == study.collect_session(0, root=public_root, output_root=output)
    store = study.open_store(public_root, output)
    _, manifest = store.read(receipt)
    assert manifest['summary']['phase'] == 'unit'
    assert manifest['summary']['segment_count'] == 5
    assert manifest['summary']['recipe_receipt'] == prepared['recipe_receipt']
    execution = manifest['summary']['execution']
    assert execution['torch_num_threads'] == 1
    assert execution['torch_num_interop_threads'] == 1
    assert set(execution['environment_threads']) == {'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                     'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'}
    assert set(execution['environment_threads'].values()) == {'1'}
    assert not (public_root / 'study').exists()
    with pytest.raises(ValueError):
        study.collect_session(1, root=public_root, output_root=output)
    status = study.status(public_root, output)
    assert status['completed_by_stage']['collect'] == 1
    assert status['phase'] == 'unit'


def test_cli_prepare_collect_status_in_fresh_process(public_root, tmp_path):
    study_module()
    output = tmp_path / 'cli-output'
    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'poc.calibration_study', *args,
                                 '--root', str(public_root), '--output-root', str(output)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    prepared = run('prepare', '--matrix', 'pilot', '--unit-fixture-segments', '5')
    assert prepared['recipe']['phase'] == 'unit'
    receipt = run('collect', '--session', '0')
    assert (output / receipt['path']).exists()
    report = run('status', '--matrix', 'pilot')
    assert report['completed_by_stage']['collect'] == 1
    assert report['output_root'] == str(output.resolve())


def test_execution_source_is_bound_even_when_public_root_is_a_copy(public_root, tmp_path, monkeypatch):
    study = study_module()
    executing = tmp_path / 'executing-poc'
    shutil.copytree(ROOT / 'poc', executing, ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(study, '__file__', str(executing / 'calibration_study.py'))
    output = tmp_path / 'out'
    study.prepare(public_root, output, unit_fixture={'segment_count': 5})
    with (executing / 'calibration_models.py').open('a') as file:
        file.write('\n# changed executing source; root copy is unchanged\n')
    with pytest.raises(ValueError, match='stale|binding'):
        study.open_store(public_root, output)


def test_runtime_change_cannot_reuse_prepared_results(public_root, tmp_path, monkeypatch):
    study = study_module()
    output = tmp_path / 'out'
    study.prepare(public_root, output, unit_fixture={'segment_count': 5})
    monkeypatch.setattr(study.np, '__version__', 'unit-different-runtime')
    with pytest.raises(ValueError, match='stale|binding'):
        study.open_store(public_root, output)


def test_status_exposes_retained_failed_jobs_and_missing_sessions(public_root, tmp_path):
    study = study_module()
    output = tmp_path / 'out'
    study.prepare(public_root, output, unit_fixture={'segment_count': 5})
    store = study.open_store(public_root, output)
    def failed(_):
        raise RuntimeError('explicit unit failed attempt')
    with pytest.raises(RuntimeError):
        store.produce('collect', {'session_id': 0, 'phase': 'unit'}, failed)
    report = study.status(public_root, output)
    assert report['missing_session_ids'] == [0]
    assert len(report['incomplete_jobs']) == 1
    assert len(report['failed_attempts']) == 1
    assert report['failed_attempts'][0]['error_type'] == 'RuntimeError'


def test_cli_no_filter_collect_and_train_cover_expected_matrix(public_root, tmp_path, monkeypatch):
    study = study_module()
    output = tmp_path / 'unit-matrix'
    study.prepare(public_root, output, 'development', unit_fixture={
        'segment_count': 5, 'max_epochs_per_rate': 5, 'physical_epochs': 5})
    # Collection is real; only expensive fitting is replaced for CLI dispatch coverage.
    assert study.main(['collect', '--root', str(public_root), '--output-root', str(output),
                       '--matrix', 'development']) == 0
    assert study.status(public_root, output, 'development')['missing_session_ids'] == []
    from poc import calibration_training
    seen, references = [], []
    def fitted(store, receipt, model_id, budget_s, initialization_id):
        _, manifest = store.read(receipt)
        seen.append((manifest['summary']['session_id'], budget_s, model_id, initialization_id))
        return receipt
    def nominal(store, receipt, budget_s):
        _, manifest = store.read(receipt)
        references.append((manifest['summary']['session_id'], budget_s))
        return receipt
    monkeypatch.setattr(calibration_training, 'fit_model', fitted)
    monkeypatch.setattr(calibration_training, 'nominal_reference', nominal)
    assert study.main(['train', '--root', str(public_root), '--output-root', str(output),
                       '--matrix', 'development']) == 0
    assert len(seen) == len(set(seen)) == 65
    assert sum(c[2] == 'physics_fit' for c in seen) == 5
    assert references == [(i, 5) for i in range(5)]
    seen.clear(); references.clear()
    assert study.main(['train', '--root', str(public_root), '--output-root', str(output),
                       '--matrix', 'development', '--session', '2', '--budget', '5',
                       '--model', 'affine_rk4', '--initialization', '1']) == 0
    assert seen == [(2, 5, 'affine_rk4', 1)]


def test_fresh_cli_no_filter_train_closes_unit_fits_and_nominal_separately(public_root, tmp_path):
    output = tmp_path / 'cli-unit-fits'
    def run(*args):
        result = subprocess.run([sys.executable, '-m', 'poc.calibration_study', *args,
                                 '--root', str(public_root), '--output-root', str(output)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    run('prepare', '--unit-fixture-segments', '5', '--unit-fixture-max-epochs', '5',
        '--unit-fixture-physical-epochs', '5')
    assert len(run('collect')['session_receipts']) == 1
    trained = run('train', '--phase', 'unit')
    assert len(trained['fit_receipts']) == 7 and len(trained['nominal_receipts']) == 1
    assert run('train', '--phase', 'unit') == trained
    report = run('status')
    assert report['completed_by_stage']['fit'] == report['expected_fit_count'] == 7
    assert report['completed_by_stage']['nominal'] == 1
