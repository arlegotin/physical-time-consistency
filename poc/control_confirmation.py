"""Freeze only the explicitly registered amended control comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .io import atomic_json, read_json, safe_path, sha256_file, utc_now

ACTUATORS = ('direct', 'lagged')
MODELS = ('mlp_dt', 'node_generic', 'power_correct')
STEPS = (40_000_000, 5_000_000)
REGISTRY_SOURCES = ('poc/control_confirmation.py', 'poc/control_confirmation_batch.py',
                    'poc/control_utility.py', 'poc/statistics.py')


def validate_design(design: dict) -> dict:
    fixed = {'protocol_id': 'UTILITY-2026-10-02-control-v1',
             'actuators': list(ACTUATORS), 'models': list(MODELS),
             'training_condition': 'A', 'loss_mode': 'shared',
             'initial_training_seed_ids': list(range(5)), 'initial_episode_ids': list(range(20)),
             'episode_seed_root': 6101, 'episode_ns': 1_600_000_000,
             'model_steps_ns': list(STEPS), 'physics_step_ns': 250_000,
             'hardware_clocks_ns': {'observation': 40_000_000, 'replanning': 40_000_000,
                                   'actuator': 40_000_000}}
    if any(design.get(k) != v for k, v in fixed.items()):
        raise ValueError('Changed prospectively registered control design')
    for name in ('initial_training_seed_ids', 'initial_episode_ids'):
        if any(type(x) is not int for x in design[name]):
            raise ValueError('Control coordinates require integer identities')
    expected = {(a, m, h) for a in ACTUATORS for m in MODELS[:2] for h in STEPS}
    found = []
    for criterion in design.get('criteria', []):
        coordinate = (criterion.get('actuator'), criterion.get('competitor'), criterion.get('model_step_ns'))
        if (criterion.get('criterion_id') != f'utility_control/{coordinate[0]}/{coordinate[1]}/h{coordinate[2]}'
                or criterion.get('metric') != 'mean_realized_episode_cost'
                or criterion.get('contrast') != 'baseline_minus_corrected'
                or type(criterion.get('effect_floor')) not in (int, float)
                or criterion['effect_floor'] != 0):
            raise ValueError('Changed registered signed cost contrast')
        found.append(coordinate)
    statistics = design.get('statistical_design', {})
    if (len(found) != 8 or set(found) != expected or statistics.get('family_size') != 8
            or statistics.get('n_initial_training_seeds') != 5
            or statistics.get('episodes_per_seed') != 20
            or statistics.get('percentage_floor') is not None):
        raise ValueError('Incomplete cost family or changed independent-cluster design')
    return design


def expected_checkpoints(design: dict) -> set:
    validate_design(design)
    return {(a, m, s) for a in ACTUATORS for m in MODELS
            for s in design['initial_training_seed_ids']}


def _check_registered_inputs(root: Path, design: dict) -> None:
    if sha256_file(root/'config/planner.json') != design['planner_sha256']:
        raise ValueError('Changed common planner since prospective registration')
    if sha256_file(safe_path(root, design['owner_objective_revision'])) != design['owner_objective_revision_sha256']:
        raise ValueError('Changed owner objective since prospective registration')


def _check_selected_fit(root: Path, item: dict) -> None:
    from .control_models import load_selected
    path = safe_path(root, item['manifest'])
    record = read_json(path)
    parameters = record['job_inputs']['parameters']
    if (record['phase'] != 'confirmatory' or record['milestone_id'] != 'M30'
            or any(parameters.get(k) != v for k, v in
                   [('condition', 'A'), ('loss_mode', 'shared'), ('actuator', item['actuator']),
                    ('model_id', item['model_id']), ('seed', item['seed'])])):
        raise ValueError('Selected checkpoint is outside the registered confirmatory fit identity')
    # Completion, nominal adequacy, predictor source, archive and canonical tensors.
    load_selected(path, item['manifest_sha256'], item['checkpoint_sha256'],
                  model_id=item['model_id'], actuator=item['actuator'], seed=item['seed'])


def freeze_cells(root: Path, design: dict, selected: list[dict]) -> dict:
    root = Path(root).resolve()
    expected = expected_checkpoints(design)
    coordinates = [(x['actuator'], x['model_id'], x['seed']) for x in selected]
    if len(coordinates) != 30 or set(coordinates) != expected:
        raise ValueError('Incomplete or duplicated qualified checkpoint coverage')
    target = root / 'config/control_confirmation_cells.json'
    if target.exists():
        raise ValueError('Confirmation checkpoint freeze already exists; verify rather than overwrite')
    if read_json(root/'config/control_confirmation_design.json') != design:
        raise ValueError('Provided design differs from the registered design file')
    _check_registered_inputs(root, design)
    from .control_jobs import SOURCES, validate_request
    from .planner_qualification import verify_frozen
    verify_frozen(root)
    for item in selected:
        _check_selected_fit(root, item)
    settings = read_json(root/'config/planner.json')['settings']
    evaluator = {p: sha256_file(root/p) for p in (*SOURCES, *REGISTRY_SOURCES)}
    jobs = []
    for item in sorted(selected, key=lambda x: (x['actuator'], x['model_id'], x['seed'])):
        for step in STEPS:
            for episode in design['initial_episode_ids']:
                cell = f"{item['actuator']}__A__shared__{item['model_id']}__seed{item['seed']}__h{step}__episode{episode}"
                request = validate_request(dict(schema_version=1, phase='confirmatory',
                    model_id=item['model_id'], actuator=item['actuator'], seed=item['seed'], episode_id=episode,
                    condition='clean', model_partition_ns=[step]*(40_000_000//step),
                    actuator_interval_ns=40_000_000, physics_step_ns=250_000, power_mode=None,
                    model_manifest=item['manifest'], manifest_sha256=item['manifest_sha256'],
                    checkpoint_sha256=item['checkpoint_sha256'], planner_settings=settings,
                    resource_projection=design['resource_projection']))
                path = root/'config/control_jobs'/f'confirmatory__{cell}.json'
                if path.exists() and read_json(path) != request:
                    raise ValueError('Conflicting prospective confirmation request')
                atomic_json(path, request)
                jobs.append(dict(cell_id=cell, path=str(path.relative_to(root)), sha256=sha256_file(path)))
    if any(sha256_file(root/p) != h for p, h in evaluator.items()):
        raise ValueError('Control evaluator changed during freeze')
    result = dict(schema_version=1, protocol_id=design['protocol_id'], state='complete',
        created_utc=utc_now(), design_sha256=sha256_file(root/'config/control_confirmation_design.json'),
        selected=selected, evaluator_hashes=evaluator, expected_episodes=1200, jobs=jobs)
    atomic_json(target, result)
    return result


def verify_freeze(root: Path, design: dict, freeze: dict) -> None:
    from .control_jobs import SOURCES, validate_request
    if (freeze.get('state') != 'complete' or freeze.get('expected_episodes') != 1200
            or freeze.get('protocol_id') != design['protocol_id']
            or freeze.get('design_sha256') != sha256_file(root/'config/control_confirmation_design.json')):
        raise ValueError('Incomplete or stale control checkpoint freeze')
    _check_registered_inputs(root, design)
    selected = freeze.get('selected', [])
    coordinates = [(s['actuator'], s['model_id'], s['seed']) for s in selected]
    if len(coordinates) != 30 or set(coordinates) != expected_checkpoints(design):
        raise ValueError('Incomplete selected checkpoint coverage')
    by_key = dict(zip(coordinates, selected))
    for item in selected:
        _check_selected_fit(root, item)
    sources = freeze.get('evaluator_hashes', {})
    if set(sources) != set(SOURCES) | set(REGISTRY_SOURCES):
        raise ValueError('Incomplete frozen evaluator coverage')
    for name, expected in sources.items():
        if sha256_file(safe_path(root, name)) != expected:
            raise ValueError('Changed frozen control evaluator: ' + name)
    seen = []
    for job in freeze.get('jobs', []):
        path = safe_path(root, job['path'])
        if sha256_file(path) != job['sha256']:
            raise ValueError('Changed frozen episode request')
        request = validate_request(read_json(path))
        key = request['actuator'], request['model_id'], request['seed']
        selected_item = by_key[key]
        step = request['model_partition_ns'][0]
        if (request['phase'] != 'confirmatory' or request['condition'] != 'clean'
                or request['actuator_interval_ns'] != 40_000_000 or request['physics_step_ns'] != 250_000
                or step not in STEPS or request['model_partition_ns'] != [step]*(40_000_000//step)
                or request['power_mode'] is not None
                or request['planner_settings'] != read_json(root/'config/planner.json')['settings']
                or any(request[k] != selected_item[v] for k, v in
                       [('model_manifest', 'manifest'), ('manifest_sha256', 'manifest_sha256'),
                        ('checkpoint_sha256', 'checkpoint_sha256')])):
            raise ValueError('Frozen episode differs from its registered cell')
        seen.append((*key, step, request['episode_id']))
    expected_jobs = {(*key, h, e) for key in expected_checkpoints(design)
                     for h in STEPS for e in design['initial_episode_ids']}
    if len(seen) != 1200 or set(seen) != expected_jobs:
        raise ValueError('Incomplete or duplicated frozen episode coverage')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    design = validate_design(read_json(root/'config/control_confirmation_design.json'))
    path = root/'config/control_confirmation_cells.json'
    freeze = read_json(path) if path.exists() else None
    if freeze is not None:
        verify_freeze(root, design, freeze)
    if not args.dry_run:
        if freeze is None:
            raise ValueError('Freeze qualified checkpoints before executing the confirmation registry')
        from .io import job_key, job_lock
        from .control_confirmation_batch import run_jobs
        registry_hash = sha256_file(path)
        with job_lock(job_key({'control_registry_sha256': registry_hash}), root):
            run_jobs(root, freeze['jobs'], registry_hash, resume=args.resume,
                     frozen_sources=freeze['evaluator_hashes'])
        return 0
    print(json.dumps(dict(execute=False, expected_episodes=1200,
        checkpoint_freeze_complete=bool(freeze and freeze.get('state') == 'complete'),
        missing_checkpoint_count=0 if freeze and freeze.get('state') == 'complete' else len(expected_checkpoints(design)),
        required_action='Complete original nominal-only fits and freeze their exact checkpoint/evaluator hashes',
        resource_projection=design['resource_projection'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
