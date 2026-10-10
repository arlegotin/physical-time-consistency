"""Complete seven original development fits using the unchanged nominal-only engine."""
from __future__ import annotations
import argparse
from pathlib import Path

from .io import append_event, atomic_json, read_json, run_directory, sha256_file, utc_now


def validate_cells(values):
    from .mechanism import cell, CORE
    expected = [cell('direct', 'A', 'shared', m, 9000) for m in ('gru_dt', 'node_structured')] + [
        cell('lagged', 'A', 'shared', m, 9000) for m in CORE]
    if values != expected:
        raise ValueError('Only the original seven unfinished development fits may run')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    owner = args.root.resolve()
    request_path = owner / 'config/control_development.json'
    request_hash = sha256_file(request_path)
    request = read_json(request_path)
    cells = validate_cells(request['nominal_fits_to_complete'])
    root = owner / request['nominal_fit_context']
    if (root != owner / 'study/control-development' or request['phase'] != 'exploratory'
            or sha256_file(owner / request['objective_revision']) != request['objective_revision_sha256']):
        raise ValueError('Changed owner scope or writable nominal-training context')
    context = read_json(root / 'CONTROL_CONTEXT.json')
    if sha256_file(root / 'INITIAL_CONTEXT.json') != context['copied_inventory_sha256']:
        raise ValueError('Working context creation provenance changed')
    if args.dry_run:
        print({'execute': False, 'cells': cells, 'engine': request['nominal_fit_engine'],
               'source_context': context['source_context'], 'context': str(root),
               'control_outcomes_used_for_selection': False,
               'resource_projection': 'Preserved nominal fit histories; measure each next 100-epoch block',
               'authorization': read_json(owner / 'config/authorization.json')})
        return 0
    from .mechanism import train_cell
    from .train import qualify_nominal
    folder = root / 'control_nominal_completion'
    folder.mkdir(exist_ok=True)
    if not (folder / 'events.jsonl').exists():
        (folder / 'events.jsonl').touch()
    atomic_json(folder / 'request.json', {'request': request, 'sha256': request_hash})
    results = []
    definition = None
    try:
        for definition in cells:
            if sha256_file(request_path) != request_hash:
                raise ValueError('Prospective nominal completion request changed during execution')
            append_event(folder / 'events.jsonl', {'event': 'nominal_cell_start', 'cell': definition, 'utc': utc_now()})
            record = train_cell(root, folder, definition, 'exploratory', resume=True)
            qualification = qualify_nominal(record, root)
            path = run_directory(root, record) / 'manifest.json'
            results.append({'cell': definition, 'manifest': str(path.relative_to(owner)),
                'manifest_sha256': sha256_file(path), 'qualification': qualification})
            atomic_json(owner / 'reports/control_nominal_completion.json', {
                'schema_version': 1, 'state': 'running', 'request_sha256': request_hash,
                'selection': 'Original nominal training/validation engine only', 'cells': results})
            if not qualification['qualified']:
                raise RuntimeError('Nominal optimization remains unqualified; retain this fit and diagnose its existing grid')
    except Exception as error:
        unqualified = bool(results and not results[-1]['qualification']['qualified'])
        atomic_json(owner / 'reports/control_nominal_completion.json', {
            'schema_version': 1, 'state': 'unqualified' if unqualified else 'interrupted',
            'request_sha256': request_hash, 'active_cell': definition, 'cells': results,
            'selection': 'Original nominal training/validation engine only',
            'reason': f'{type(error).__name__}: {error}'})
        append_event(folder / 'events.jsonl', {'event': 'nominal_completion_stopped',
            'cell': definition, 'reason': f'{type(error).__name__}: {error}', 'utc': utc_now()})
        raise
    atomic_json(owner / 'reports/control_nominal_completion.json', {
        'schema_version': 1, 'state': 'complete', 'request_sha256': request_hash,
        'selection': 'Original nominal training/validation engine only', 'cells': results})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
