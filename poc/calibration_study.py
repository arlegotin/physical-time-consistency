"""Prospective recipes and CLI coordination for fresh PTCO-CAL-0.1 studies."""
from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import platform

# Install numerical runtime settings before imports in fresh CLI processes.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for _thread_variable in THREAD_VARIABLES:
    os.environ[_thread_variable] = '1'

import mujoco
import numpy as np
import scipy
import torch

from .calibration_data import PHYSICS_STEP_NS, collect_arrays
from .calibration_models import ALL_MODEL_IDS, MODEL_IDS, HYBRID_MODEL_IDS, NOMINAL_PARAMETERS, propagation_family
from .calibration_store import CalibrationStore, PROTOCOL_ID, Receipt
from .io import atomic_bytes, atomic_json, canonical_json, job_key, job_lock, read_json, sha256_file


def prospective_recipe(*, unit_fixture: dict | None = None, matrix: str = 'pilot') -> dict:
    """Portable acquisition and analysis choices, frozen before scored outcomes.

    Reduced fixtures are conspicuously marked unit and cannot support study claims.
    Matrix membership lives separately, so expansion retains the pilot fit inputs.
    """
    if matrix not in {'pilot', 'development', 'hybrid'}:
        raise ValueError('Unknown calibration matrix')
    recipe = {
        'protocol_id': PROTOCOL_ID, 'phase': 'exploratory', 'empirical_claims_allowed': True,
        'raw_record_encoding': {'predictions':'rows.jsonl.gz', 'decisions':'rows.jsonl.gz',
                                'format':'gzip-jsonl', 'compression_level':1, 'mtime':0},
        'actuator': {'kind': 'lagged', 'tau_s': .08, 'state_order': ['q1', 'q2', 'v1', 'v2', 'w1', 'w2'],
                     'command_unit': 'N*m', 'joint_count': 2},
        'seed_namespace': [20261012, 1],
        'seed_coordinates': ['role', 'session', 'trajectory', 'channel'],
        'role_codes': {'train-session': 0, 'development-evaluation': 1, 'confirmation': 2},
        'initialization_seed': '2026101200 + 2*session_id + initialization_id',
        'collector': {'segment_count': 600, 'segment_seconds': 1, 'commands_per_segment': 25,
                      'observation_tick_ns': 40_000_000, 'physics_step_ns': PHYSICS_STEP_NS,
                      'initial_q': [-.5, .5], 'initial_v': [-.25, .25], 'initial_w': [-.3, .3],
                      'law': 'clip(-q - .25*v + d, -1.5, 1.5)', 'd_bounds': [-.75, .75],
                      'd_hold_ticks': 5, 'initial_state_channel': 0, 'excitation_channel': 1,
                      'independent_simulator_resets': True, 'reset_settling_overhead_counted': False,
                      'telemetry': '40 ms states and held-command log; dense truth is evaluator-only'},
        'budgets_s': [30, 120, 600], 'validation_rule': 'every fifth complete segment (zero-based id % 5 == 4)',
        'models': list(MODEL_IDS), 'nominal_prior_parameters': list(NOMINAL_PARAMETERS),
        'training': {'dtype': 'float32', 'device': 'cpu', 'threads': 1, 'own_solver': True,
                     'independent_initialization_per_budget': True, 'optimizer': 'AdamW',
                     'effective_batch': 256, 'objective': 'one_step + .25 * four_step',
                     'shuffle': 'shuffle eight-start windows, flatten, batch up to 256 starts',
                     'learning_rates': [.001, .0003], 'validation_every_epochs': 5,
                     'epoch_block': 100, 'max_epochs_per_rate': 1000,
                     'plateau': 'existing nominal plateau rule after at least 100 epochs',
                     'selection': 'all valid rates/checkpoints by nominal validation and existing deterministic tie rule',
                     'physics_fit': 'existing endpoint fitting recipe and solver-resolution check',
                     'physical_epochs': 100,
                     'failed_attempts': 'retain; never replace draws or discard poor fits',
                     'qualification': 'historical constant-velocity threshold does not reject study fits'},
        'evaluation': {'trajectory_count': 64, 'trajectory_seconds': 1.6, 'role_code': 1,
                       'anchor_start_ns': 320_000_000, 'anchor_stop_ns': 1_200_000_000,
                       'anchor_tick_ns': 40_000_000, 'anchors_per_trajectory': 23,
                       'query_steps_ns': [5_000_000, 10_000_000, 20_000_000, 40_000_000],
                       'endpoint_horizons_ns': [5_000_000, 10_000_000, 20_000_000, 40_000_000],
                       'rollout_horizons_ns': [80_000_000, 160_000_000, 400_000_000],
                       'stress_held_horizons_ns': [80_000_000, 160_000_000],
                       'truth_step_ns': 250_000, 'refinement_steps_ns': [125_000, 62_500],
                       'refinement_coordinates': 'first and last evaluation trajectories',
                       'envelope_fraction_limit': .01, 'scoring_dtype': 'float64',
                       'seed_session_id': 0, 'held_probe_channel': 2, 'target_channel': 3,
                       'held_probe_bounds': [-1.5, 1.5], 'target_bounds': [-.5, .5],
                       'held_probe_law': 'independent uniform two-joint draw at each of 41 observation coordinates; shared across horizons',
                       'target_law': 'one independent uniform two-joint target per trajectory',
                       'drift_command': [0., 0.], 'initial_secondary': True,
                       'intervention': 'immutable observed q/v/w initial state for new-command and zero-command branches',
                       'composition': 'held P(h) versus P(h/2) composed twice with identical command',
                       'adaptive_refinement_steps_ns': [31_250, 15_625],
                       'adaptive_refinement_policy': 'refine cases above 1% error; last-two-resolution discrepancy envelope; stop on nondecreasing discrepancy or exhausted integer grid; retain unresolved equality'},
        'decisions': {'candidate_count': 256, 'knot_count': 6, 'knot_duration_ns': 40_000_000,
                      'anchor_ns': 320_000_000, 'random_command_bounds': [-1.5, 1.5],
                      'bank_channel': 4,
                      'bank_law': 'PCG64 role=1/session=0/trajectory/channel=4; independent uniform six-knot paths at each 40 ms observation coordinate',
                      'truth_refinement': 'all candidates 250000/125000/62500 ns; unresolved simulator-cost separations refine at 31250/15625 ns; plateau or exhaustion retains ties',
                      'included_candidates': ['zero', 'collector current PD repeated'],
                      'target_bounds': [-.5, .5], 'position_cost': .04, 'velocity_cost': .002,
                      'command_cost': .00004, 'terminal_cost': 5., 'sampled_feasibility_tick_ns': 5_000_000,
                      'q_bound': 1.5, 'v_bound': 3., 'u_bound': 2.,
                      'auxiliary_queries': 'query offsets from main-step start; only main endpoint advances state',
                      'tie_selection': 'lowest candidate index among numerically resolved predicted-cost ties',
                      'regret': 'truth selected cost - minimum truly feasible cost; conditional on valid decisions',
                      'invalid_selection': 'decision failure, no benign zero regret',
                      'empty_true_feasible_bank': 'shared bank-design failure; screen unresolved'},
        'timing': {'batches': [1, 128, 256], 'warmup_calls': 20, 'measured_repetitions': 50,
                   'threads': 1, 'method_order': 'rotated', 'measurements': ['median', 'p95', 'cpu', 'wall'],
                   'representative_input': 'earliest primary decision anchor from lowest trajectory identity; fixed common state target and bank',
                   'cache': 'detached fixed interval coefficients equally available to applicable models; differentiable h bypasses cache',
                   'complete_scorer': 'includes auxiliary queries, batching, constraints and selection'},
        'metrics': {'rmse': 'sqrt(mean squared component error over endpoints and both joints)',
                    'p95': '95th percentile of two-joint Euclidean endpoint-error norms'},
        'screen': {'horizon_ns': 160_000_000, 'q_rmse_rad': .01, 'v_rmse_rad_s': .10,
                   'q_p95_rad': .03, 'v_p95_rad_s': .30, 'prediction_failures': 0,
                   'decision_failure_fraction': .01, 'valid_regret_fraction_of_zero_cost': .01,
                   'undefined_zero_cost_le': 1e-8, 'same_main_partition_required': True},
        'aggregation': {'independent_cluster': 'calibration session',
                        'initializations': 'average within session', 'bootstrap_draws': 10000,
                        'bootstrap': 'paired session cluster; fixed namespace', 'bootstrap_namespace': [20261012, 1, 1, 0, 0, 5],
                        'bootstrap_rng': 'PCG64 SeedSequence([20261012,1,role=1,session=0,trajectory=0,channel=5])', 'interval_percent': 95},
    }
    if matrix == 'hybrid':
        recipe['models'] = list(HYBRID_MODEL_IDS)
        recipe['budgets_s'] = [30, 120]
        recipe['evaluation']['seed_session_id'] = 1
        recipe['decisions']['bank_law'] = recipe['decisions']['bank_law'].replace('session=0/', 'session=1/')
        recipe['hybrid_component'] = {
            'primary_model_id': 'hybrid_affine_flow', 'anchor_model_id': 'hybrid_accel_rk4',
            'session_ids': [1, 2, 3], 'initialization_ids': [0], 'primary_budget_s': 30,
            'primary_query_step_ns': 40_000_000, 'error_ratio_limit': 1.05,
            'median_ratio_limit': .75, 'p95_ratio_limit': 1.0}
    if unit_fixture is not None:
        if set(unit_fixture)-{'segment_count', 'budgets_s', 'max_epochs_per_rate', 'physical_epochs',
                              'evaluation_trajectory_count', 'evaluation_anchor_times_ns', 'decision_candidate_count'}:
            raise ValueError('Unknown unit recipe override')
        count = unit_fixture.get('segment_count', 5)
        if type(count) is not int or count <= 0 or count % 5:
            raise ValueError('Unit sessions require complete five-segment role groups')
        budgets = unit_fixture.get('budgets_s', [b for b in recipe['budgets_s'] if b <= count] or [count])
        if (not isinstance(budgets, list) or not budgets or len(set(budgets)) != len(budgets) or
                any(type(b) is not int or b <= 0 or b > count or b % 5 for b in budgets)):
            raise ValueError('Unit budgets require unique complete five-segment role groups')
        recipe.update({'phase': 'unit', 'empirical_claims_allowed': False, 'budgets_s': sorted(budgets)})
        recipe['collector']['segment_count'] = count
        if 'evaluation_trajectory_count' in unit_fixture:
            value = unit_fixture['evaluation_trajectory_count']
            if type(value) is not int or not 2 <= value <= 64:
                raise ValueError('Unit evaluation requires 2 through 64 trajectories including first/last audits')
            recipe['evaluation']['trajectory_count'] = value
        if 'evaluation_anchor_times_ns' in unit_fixture:
            values = unit_fixture['evaluation_anchor_times_ns']
            allowed = list(range(320_000_000, 1_200_000_001, 40_000_000))
            if (not isinstance(values, list) or not values or
                    any(type(v) is not int or v not in allowed for v in values) or values != sorted(set(values))):
                raise ValueError('Unit anchors must be an ordered nonempty subset of the primary population')
            recipe['evaluation']['unit_anchor_times_ns'] = values
        if 'decision_candidate_count' in unit_fixture:
            value = unit_fixture['decision_candidate_count']
            if type(value) is not int or not 2 <= value <= 256:
                raise ValueError('Unit candidate count requires 2 through 256, including both references')
            recipe['decisions']['candidate_count'] = value
        for key in ('max_epochs_per_rate', 'physical_epochs'):
            if key in unit_fixture:
                value = unit_fixture[key]
                if type(value) is not int or value <= 0 or value % 5 or value > 1000:
                    raise ValueError('Unit epoch controls require positive five-epoch units up to 1000')
                recipe['training'][key] = value
    return recipe


def study_bindings(root: Path, recipe: dict) -> dict[str, str]:
    """Hash the complete available scientific module set at execution time."""
    root = Path(root).resolve()
    source_paths = sorted((root / 'poc').rglob('*.py'))
    asset_paths = sorted((root / 'assets').rglob('*.xml'))
    if not source_paths or not (root / 'assets/two_link_lagged.xml').is_file():
        raise ValueError('A public checkout with scientific source and lagged plant asset is required')
    sources = {str(path.relative_to(root)): sha256_file(path) for path in source_paths}
    assets = {str(path.relative_to(root)): sha256_file(path) for path in asset_paths}
    executing_root = Path(__file__).resolve().parent
    executing_sources = {str(path.relative_to(executing_root)): sha256_file(path)
                         for path in sorted(executing_root.rglob('*.py'))}
    return {'root': str(root), 'source': hashlib.sha256(canonical_json(sources)).hexdigest(),
            'executing_source': hashlib.sha256(canonical_json(executing_sources)).hexdigest(),
            'asset': hashlib.sha256(canonical_json(assets)).hexdigest(),
            'runtime': hashlib.sha256(canonical_json(_runtime_versions())).hexdigest(),
            'recipe': hashlib.sha256(canonical_json(recipe)).hexdigest()}


def _runtime_versions() -> dict:
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'mujoco': mujoco.__version__, 'scipy': scipy.__version__}


def _execution_settings() -> dict:
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return {'device': 'cpu', 'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'environment_threads': {name: os.environ.get(name) for name in THREAD_VARIABLES},
            **_runtime_versions()}


def _coordinates(recipe: dict, matrix: str) -> list[dict]:
    if matrix not in {'pilot', 'development', 'hybrid'}:
        raise ValueError('Unknown calibration matrix')
    coordinates = []
    sessions = recipe['hybrid_component']['session_ids'] if matrix == 'hybrid' else range(1 if matrix == 'pilot' else 5)
    for session in sessions:
        for budget in recipe['budgets_s']:
            for model in recipe['models']:
                initializations = ([None] if model == 'physics_fit' else
                                   recipe['hybrid_component']['initialization_ids'] if matrix == 'hybrid' else
                                   range(1 if matrix == 'pilot' else 2))
                for initialization in initializations:
                    coordinates.append({'session_id': session, 'budget_s': budget,
                                        'model_id': model, 'initialization_id': initialization})
    return coordinates


def prepare(root: Path, output_root: Path, matrix: str = 'pilot', *,
            unit_fixture: dict | None = None) -> dict:
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    recipe = prospective_recipe(unit_fixture=unit_fixture, matrix=matrix)
    coordinates = _coordinates(recipe, matrix)
    bindings = study_bindings(root, recipe)
    store = CalibrationStore(output_root, bindings)
    settings = _execution_settings()
    with job_lock(job_key({'prepare_output_root': str(output_root)}), output_root):
        path = output_root / 'prepared.json'
        if path.exists():
            frozen = read_json(path)
            if frozen['bindings'] != bindings or frozen['recipe'] != recipe:
                raise ValueError('Prepared study binding/recipe is stale; use a new output root')
            store.read(frozen['recipe_receipt'])
            recipe_receipt = frozen['recipe_receipt']
        else:
            def freeze(folder):
                atomic_json(folder / 'recipe.json', recipe)
                source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'poc').rglob('*.py'))}
                asset_hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / 'assets').rglob('*.xml'))}
                atomic_json(folder / 'source_hashes.json', source_hashes)
                atomic_json(folder / 'asset_hashes.json', asset_hashes)
                executing_root = Path(__file__).resolve().parent
                atomic_json(folder / 'executing_source_hashes.json', {
                    str(p.relative_to(executing_root)): sha256_file(p)
                    for p in sorted(executing_root.rglob('*.py'))})
                return {'phase': recipe['phase'], 'recipe': recipe, 'execution': settings}
            recipe_receipt = store.produce('recipe', {'recipe': recipe}, freeze)
            frozen = {'protocol_id': PROTOCOL_ID, 'recipe': recipe, 'bindings': bindings,
                      'recipe_receipt': recipe_receipt}
            atomic_json(path, frozen)
        matrix_request = {'matrix': matrix, 'recipe_receipt': recipe_receipt, 'coordinates': coordinates}
        def membership(folder):
            atomic_json(folder / 'matrix.json', matrix_request)
            return {'matrix': matrix, 'phase': recipe['phase'], 'fit_count': len(coordinates)}
        matrix_receipt = store.produce('matrix', matrix_request, membership)
        result = {**frozen, 'matrix': matrix, 'coordinates': coordinates, 'matrix_receipt': matrix_receipt}
        matrix_path = output_root / 'matrices' / f'{matrix}.json'
        if matrix_path.exists():
            if read_json(matrix_path) != result:
                raise ValueError('Prepared matrix membership is immutable')
        else:
            atomic_json(matrix_path, result)
        return result


def open_store(root: Path, output_root: Path, matrix: str = 'pilot') -> CalibrationStore:
    output_root = Path(output_root).resolve()
    path = output_root / 'prepared.json'
    if not path.exists():
        raise ValueError('Run prepare before calibration study execution')
    frozen = read_json(path)
    current = study_bindings(root, frozen['recipe'])
    if frozen['bindings'] != current or frozen['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Prepared study binding is stale; use a new output root')
    store = CalibrationStore(output_root, current)
    _, recipe_manifest = store.read(frozen['recipe_receipt'])
    if recipe_manifest['summary']['recipe'] != frozen['recipe']:
        raise ValueError('Prepared recipe binding mismatch')
    matrix_path = output_root / 'matrices' / f'{matrix}.json'
    if matrix not in {'pilot', 'development', 'hybrid'} or not matrix_path.is_file():
        raise ValueError('Run prepare for the requested matrix')
    membership = read_json(matrix_path)
    _, matrix_manifest = store.read(membership['matrix_receipt'])
    expected = {'matrix': matrix, 'recipe_receipt': frozen['recipe_receipt'],
                'coordinates': _coordinates(frozen['recipe'], matrix)}
    if matrix_manifest['request'] != expected or membership != {**frozen, **expected,
                                                               'matrix_receipt': membership['matrix_receipt']}:
        raise ValueError('Prepared matrix coordinate binding mismatch')
    # Public context for downstream stage functions; immutable receipts remain authoritative.
    store.recipe = frozen['recipe']
    store.recipe_receipt = frozen['recipe_receipt']
    store.matrix = matrix
    store.coordinates = membership['coordinates']
    store.execution_settings = _execution_settings()
    return store


def collect_session(session_id: int, *, root: Path, output_root: Path, matrix: str = 'pilot') -> Receipt:
    store = open_store(root, output_root, matrix)
    if type(session_id) is not int or session_id not in {c['session_id'] for c in store.coordinates}:
        raise ValueError('Session is outside prepared matrix membership')
    recipe = store.recipe
    request = {'session_id': session_id, 'role': 'train-session', 'phase': recipe['phase'],
               'recipe_receipt': store.recipe_receipt, 'collector': recipe['collector']}
    def collect(folder):
        arrays = collect_arrays(session_id, root=root, segment_count=recipe['collector']['segment_count'],
                                phase=recipe['phase'], resume_dir=folder / 'segments')
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        atomic_bytes(folder / 'session.npz', buffer.getvalue())
        return {'session_id': session_id, 'phase': recipe['phase'],
                'segment_count': len(arrays['trajectory_id']), 'budget_seconds': recipe['budgets_s'],
                'motion_seconds': len(arrays['trajectory_id']), 'transitions': arrays['command'].shape[0]*25,
                'recipe_receipt': store.recipe_receipt, 'execution': store.execution_settings,
                'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
                'reset_settling_overhead_counted': False, 'dense_labels_available': False}
    return store.produce('collect', request, collect)


def status(root: Path, output_root: Path, matrix: str = 'pilot') -> dict:
    store = open_store(root, output_root, matrix)
    from .calibration_report import verify
    coverage = verify(store, matrix)
    completed: dict[str, int] = {}
    receipts = []
    collected_sessions = set()
    for marker in sorted(store.output_root.glob('jobs/*/*/COMPLETED')):
        folder = marker.parent
        try:
            receipt = {'job_key': folder.name, 'path': str(folder.relative_to(store.output_root)),
                       'manifest_sha256': read_json(marker)['manifest_sha256']}
            _, manifest = store.read(receipt)
        except (ValueError, OSError, KeyError, TypeError):
            # Invalid evidence is explicit in coverage, never counted as completed.
            continue
        stage = manifest['stage']
        completed[stage] = completed.get(stage, 0)+1
        receipts.append(receipt)
        if stage == 'collect':
            collected_sessions.add(manifest['request']['session_id'])
    incomplete = [{'path': str(path.parent.relative_to(store.output_root)),
                   **read_json(path)}
                  for path in sorted(store.output_root.glob('jobs/*/*/request.json'))
                  if not (path.parent / 'COMPLETED').exists()]
    return {'coverage': coverage, 'protocol_id': PROTOCOL_ID, 'output_root': str(store.output_root), 'matrix': matrix,
            'phase': store.recipe['phase'], 'completed_by_stage': completed,
            'receipts': receipts, 'expected_fit_count': len(store.coordinates),
            'missing_session_ids': sorted({c['session_id'] for c in store.coordinates}-collected_sessions),
            'incomplete_jobs': incomplete, 'failed_attempts': coverage['failed_attempts']}


def train_coordinates(root: Path, output_root: Path, matrix: str = 'pilot', *,
                      session_id: int | None = None, model_id: str | None = None,
                      budget_s: int | None = None, initialization_id: int | None = None) -> dict:
    from .calibration_training import fit_model, nominal_reference
    store = open_store(root, output_root, matrix)
    coordinates = [c for c in store.coordinates if
                   (session_id is None or c['session_id'] == session_id) and
                   (model_id in {None, 'physics_nominal'} or c['model_id'] == model_id) and
                   (budget_s is None or c['budget_s'] == budget_s) and
                   (initialization_id is None or c['initialization_id'] == initialization_id)]
    if not coordinates or (model_id == 'physics_nominal' and initialization_id is not None):
        raise ValueError('Requested training filters match no prepared coordinates')
    collections = {}
    for receipt in status(root, output_root, matrix)['receipts']:
        _, manifest = store.read(receipt)
        if manifest['stage'] == 'collect':
            collections[manifest['summary']['session_id']] = receipt
    pairs = sorted({(c['session_id'], c['budget_s']) for c in coordinates})
    if any(session not in collections for session, _ in pairs):
        raise ValueError('Run collect for all requested sessions before train')
    references = [nominal_reference(store, collections[session], budget) for session, budget in pairs]
    fits = [] if model_id == 'physics_nominal' else [
        fit_model(store, collections[c['session_id']], c['model_id'], c['budget_s'], c['initialization_id'])
        for c in coordinates]
    return {'phase': store.recipe['phase'], 'fit_receipts': fits, 'nominal_receipts': references}


def _measure_coordinates(root: Path, output_root: Path, matrix: str = 'pilot', *,
                        session_id: int | None = None, model_id: str | None = None,
                        budget_s: int | None = None, initialization_id: int | None = None, decisions: bool = False,
                        timing: bool = False) -> dict:
    from .calibration_predictions import (prepare_evaluation, prediction_truth,
                                         measure_prediction, propagation_diagnostic)
    store = open_store(root, output_root, matrix)
    expected = list(store.coordinates)
    expected += [{'session_id': sid, 'budget_s': budget, 'model_id': 'physics_nominal',
                  'initialization_id': None}
                 for sid, budget in sorted({(c['session_id'], c['budget_s']) for c in store.coordinates})]
    expected = [c for c in expected if
                (session_id is None or c['session_id'] == session_id) and
                (model_id is None or c['model_id'] == model_id) and
                (budget_s is None or c['budget_s'] == budget_s) and
                (initialization_id is None or c['initialization_id'] == initialization_id)]
    if not expected:
        raise ValueError('Prediction filters are outside prepared matrix membership')
    selected = {}
    for marker in sorted(store.output_root.glob('jobs/*/*/COMPLETED')):
        folder = marker.parent
        receipt = {'job_key': folder.name, 'path': str(folder.relative_to(store.output_root)),
                   'manifest_sha256': read_json(marker)['manifest_sha256']}
        _, manifest = store.read(receipt)
        if manifest['stage'] not in {'fit', 'nominal'}:
            continue
        coordinate = {k: manifest['request'][k] for k in
                      ('session_id', 'budget_s', 'model_id', 'initialization_id')}
        if coordinate in expected:
            key = canonical_json(coordinate)
            if key in selected:
                raise ValueError('Duplicate selected prediction coordinate')
            selected[key] = receipt
    evaluation = prepare_evaluation(store, root=root)
    if timing:
        from .calibration_timing import measure_timing
        receipts = [selected[canonical_json(c)] for c in expected if canonical_json(c) in selected]
        return {'evaluation_receipt':evaluation,
                'timing_receipt':measure_timing(store,receipts,evaluation) if receipts else None,
                'missing_coordinates':[c for c in expected if canonical_json(c) not in selected]}
    if decisions:
        from .calibration_decisions import truth_candidates, measure_decisions
        truth = truth_candidates(store, evaluation, root=root)
    else:
        truth = prediction_truth(store, evaluation, root=root)
    predictions, diagnostics, missing = [], [], []
    for coordinate in expected:
        receipt = selected.get(canonical_json(coordinate))
        if receipt is None:
            missing.append(coordinate)
            continue
        measure = measure_decisions if decisions else measure_prediction
        predictions.append(measure(store, receipt, evaluation, truth))
        if not decisions and propagation_family(coordinate['model_id']):
            diagnostics.append(propagation_diagnostic(store, receipt, evaluation, truth))
    if decisions:
        return {'evaluation_receipt': evaluation, 'truth_receipt': truth,
                'decision_receipts': predictions, 'missing_coordinates': missing}
    return {'evaluation_receipt': evaluation, 'truth_receipt': truth,
            'prediction_receipts': predictions, 'diagnostic_receipts': diagnostics,
            'missing_coordinates': missing}


def predict_coordinates(root: Path, output_root: Path, matrix: str = 'pilot', *,
                        session_id: int | None = None, model_id: str | None = None,
                        budget_s: int | None = None, initialization_id: int | None = None) -> dict:
    return _measure_coordinates(root, output_root, matrix, session_id=session_id, model_id=model_id,
                                budget_s=budget_s, initialization_id=initialization_id)


def decisions_coordinates(root: Path, output_root: Path, matrix: str = 'pilot', *,
                          session_id: int | None = None, model_id: str | None = None,
                          budget_s: int | None = None, initialization_id: int | None = None) -> dict:
    return _measure_coordinates(root, output_root, matrix, decisions=True, session_id=session_id,
                                model_id=model_id, budget_s=budget_s, initialization_id=initialization_id)


def time_coordinates(root: Path, output_root: Path, matrix: str = 'pilot', *,
                     session_id: int | None = None, model_id: str | None = None,
                     budget_s: int | None = None, initialization_id: int | None = None) -> dict:
    return _measure_coordinates(root,output_root,matrix,timing=True,session_id=session_id,
                                model_id=model_id,budget_s=budget_s,initialization_id=initialization_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'collect', 'train', 'predict', 'decisions', 'time', 'report', 'verify', 'status'):
        subparser = subparsers.add_parser(command)
        subparser.add_argument('--root', type=Path, default=Path('.'))
        subparser.add_argument('--output-root', type=Path, default=Path('study/short-calibration'))
        subparser.add_argument('--matrix', choices=('pilot', 'development', 'hybrid'), default='pilot')
        subparser.add_argument('--phase', choices=('unit', 'exploratory'), help='check prepared phase')
        if command == 'report':
            subparser.add_argument('--destination', type=Path, help='default: selected output root / reports')
        if command == 'prepare':
            subparser.add_argument('--unit-fixture-segments', type=int,
                                   help='explicit nonempirical fixture; never valid for research claims')
            subparser.add_argument('--unit-fixture-max-epochs', type=int)
            subparser.add_argument('--unit-fixture-physical-epochs', type=int)
            subparser.add_argument('--unit-fixture-evaluation-trajectories', type=int)
            subparser.add_argument('--unit-fixture-evaluation-anchors-ns', nargs='+', type=int)
            subparser.add_argument('--unit-fixture-decision-candidates', type=int)
        if command in {'collect', 'train', 'predict', 'decisions', 'time'}:
            subparser.add_argument('--session', type=int)
        if command in {'train', 'predict', 'decisions', 'time'}:
            subparser.add_argument('--model', choices=(*ALL_MODEL_IDS, 'physics_nominal'))
            subparser.add_argument('--budget', type=int)
            subparser.add_argument('--initialization', type=int)
    args = parser.parse_args(argv)
    if args.phase is not None and args.command != 'prepare':
        if open_store(args.root, args.output_root, args.matrix).recipe['phase'] != args.phase:
            raise ValueError('Requested phase differs from prepared recipe')
    if args.command == 'prepare':
        unit = {'segment_count': args.unit_fixture_segments} if args.unit_fixture_segments is not None else None
        for key, value in (('max_epochs_per_rate', args.unit_fixture_max_epochs),
                           ('physical_epochs', args.unit_fixture_physical_epochs),
                           ('evaluation_trajectory_count', args.unit_fixture_evaluation_trajectories),
                           ('evaluation_anchor_times_ns', args.unit_fixture_evaluation_anchors_ns),
                           ('decision_candidate_count', args.unit_fixture_decision_candidates)):
            if value is not None:
                if unit is None:
                    raise ValueError('Reduced training controls require --unit-fixture-segments')
                unit[key] = value
        if args.phase is not None and args.phase != ('unit' if unit else 'exploratory'):
            raise ValueError('Requested prepare phase differs from explicit fixture controls')
        result = prepare(args.root, args.output_root, args.matrix, unit_fixture=unit)
    elif args.command == 'collect':
        if args.session is not None:
            result = collect_session(args.session, root=args.root, output_root=args.output_root, matrix=args.matrix)
        else:
            store = open_store(args.root, args.output_root, args.matrix)
            result = {'session_receipts': [collect_session(session, root=args.root, output_root=args.output_root,
                                                         matrix=args.matrix)
                                          for session in sorted({c['session_id'] for c in store.coordinates})]}
    elif args.command == 'train':
        result = train_coordinates(args.root, args.output_root, args.matrix, session_id=args.session,
                                   model_id=args.model, budget_s=args.budget, initialization_id=args.initialization)
    elif args.command in {'predict', 'decisions', 'time'}:
        measure = {'predict':predict_coordinates,'decisions':decisions_coordinates,'time':time_coordinates}[args.command]
        result = measure(args.root, args.output_root, args.matrix, session_id=args.session,
                                     model_id=args.model, budget_s=args.budget, initialization_id=args.initialization)
    elif args.command in {'report', 'verify'}:
        from .calibration_report import aggregate, verify, write_report
        store = open_store(args.root, args.output_root, args.matrix)
        if args.command == 'verify':
            result = verify(store, args.matrix)
        else:
            report = aggregate(store, args.matrix)
            paths = write_report(report, args.destination or store.output_root / 'reports')
            result = {'paths': [str(p) for p in paths], 'coverage': report['coverage'], 'gate': report['gate']}
    else:
        result = status(args.root, args.output_root, args.matrix)
    print(canonical_json(result).decode('utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
