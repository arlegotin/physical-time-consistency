"""Immutable PTCO-CAL-0.1 artifacts with resumable work and retained events."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import re
import resource
import time
from typing import Callable

from .calibration_resources import ResourceTracker
from .io import (append_event, atomic_json, canonical_json, job_key, job_lock,
                 read_json, safe_path, sha256_file, utc_now)

PROTOCOL_ID = 'PTCO-CAL-0.1'
Receipt = dict[str, str]


@contextmanager
def _resource_interval(resources, category, clock, cpu_clock):
    if resources is None:
        yield
        return
    wall,cpu = clock(),cpu_clock()
    try:
        yield
    finally:
        resources[category+'_wall_s'] += clock()-wall
        resources[category+'_cpu_s'] += cpu_clock()-cpu


class CalibrationStore:
    def __init__(self, output_root: Path, bindings: dict[str, str]):
        self.output_root = Path(output_root).resolve()
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in bindings.items()):
            raise ValueError('Study bindings must be strings')
        self.bindings = {**bindings, 'protocol': PROTOCOL_ID, 'output_root': str(self.output_root)}

    def _identity(self, stage: str, request: dict) -> dict:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', stage):
            raise ValueError('Unsafe study stage')
        coordinates = request.get('coordinates', [])
        if len({canonical_json(c) for c in coordinates}) != len(coordinates):
            raise ValueError('Duplicate study coordinates')
        return json.loads(canonical_json({'protocol_id': PROTOCOL_ID, 'stage': stage,
                                         'bindings': self.bindings, 'request': request}))

    def _receipt(self, key: str, folder: Path) -> Receipt:
        return {'job_key': key, 'path': str(folder.relative_to(self.output_root)),
                'manifest_sha256': sha256_file(folder / 'manifest.json')}

    def produce(self, stage: str, request: dict, producer: Callable[[Path], dict], *,
                measure_finalization: bool = False, clock=None,
                cpu_clock=None) -> Receipt:
        """Optionally log actual finalization work after immutable completion.

        Numeric observations live outside the immutable product, in events bound
        to its exact final receipt. The observation record's own persistence is
        explicitly excluded; no manifest is rewritten to record its own cost.
        Reuse produces a separate reused observation, never new original costs.
        """
        clock, cpu_clock = clock or time.perf_counter, cpu_clock or time.process_time
        identity = self._identity(stage, request)
        key = job_key(identity)
        folder = self.output_root / 'jobs' / stage / key
        resources = {category+'_'+kind+'_s':0. for category in
            ('artifact_hash','metadata_write_hash','receipt_create_validate','completion_event_write')
            for kind in ('wall','cpu')} if measure_finalization else None
        def record_finalization(receipt,reused,wall,cpu):
            if resources is None:
                return
            resources.update({'store_finalization_wall_s':clock()-wall,
                              'store_finalization_cpu_s':cpu_clock()-cpu,
                              'resource_record_write_measured':False})
            append_event(self.output_root/'events.jsonl',{
                'event':'store_finalization','protocol_id':PROTOCOL_ID,'stage':stage,'job_key':key,
                'receipt':receipt,'reused':reused,'resources':resources,'utc':utc_now(),
                'measurement_envelope':'actual post-producer hash/metadata/receipt/completed-event work; observation record persistence excluded'})
        with job_lock(key, self.output_root):
            if (folder / 'COMPLETED').exists():
                final_wall,final_cpu = (clock(),cpu_clock()) if measure_finalization else (0.,0.)
                with _resource_interval(resources,'receipt_create_validate',clock,cpu_clock):
                    receipt = self._receipt(key, folder)
                    self.read(receipt)
                record_finalization(receipt,True,final_wall,final_cpu)
                return receipt
            if (folder / 'manifest.json').exists():
                raise ValueError('Final manifest has no valid completion marker; preserve evidence for diagnosis')
            folder.mkdir(parents=True, exist_ok=True)
            request_path = folder / 'request.json'
            if request_path.exists() and read_json(request_path) != identity:
                raise ValueError('Changed job binding is not a resumption')
            atomic_json(request_path, identity)
            event = {'stage': stage, 'job_key': key, 'request': identity['request']}
            append_event(self.output_root / 'events.jsonl', {**event, 'event': 'started', 'utc': utc_now()})
            tracker = ResourceTracker(folder / 'resources.json', clock=clock, cpu_clock=cpu_clock, producer_key=key)
            try:
                with tracker:
                    summary = producer(folder)
                if not isinstance(summary, dict):
                    raise ValueError('Study producer must return a JSON object summary')
                final_wall,final_cpu = (clock(),cpu_clock()) if measure_finalization else (0.,0.)
                with _resource_interval(resources,'artifact_hash',clock,cpu_clock):
                    files = {str(path.relative_to(folder)): sha256_file(safe_path(folder, str(path.relative_to(folder))))
                             for path in sorted(folder.rglob('*')) if path.is_file()
                             and path not in {folder / 'manifest.json', folder / 'COMPLETED'}}
                with _resource_interval(resources,'metadata_write_hash',clock,cpu_clock):
                    manifest = {**identity, 'job_key': key, 'files': files, 'summary': summary,
                                'resources': {**tracker.summary(),
                                              'peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                              'peak_rss_native_unit': 'bytes on macOS; KiB on Linux',
                                              'produced_bytes': sum((folder / p).stat().st_size for p in files)},
                                'completed_utc': utc_now()}
                    atomic_json(folder / 'manifest.json', manifest)
                    atomic_json(folder / 'COMPLETED', {'manifest_sha256': sha256_file(folder / 'manifest.json')})
                with _resource_interval(resources,'receipt_create_validate',clock,cpu_clock):
                    receipt = self._receipt(key, folder)
                    self.read(receipt)
            except BaseException as error:
                append_event(self.output_root / 'events.jsonl', {
                    **event, 'event': 'failed', 'utc': utc_now(),
                    'error_type': type(error).__name__, 'error': str(error),
                    'resources':tracker.summary()})
                raise
            with _resource_interval(resources,'completion_event_write',clock,cpu_clock):
                append_event(self.output_root / 'events.jsonl', {**event, 'event': 'completed',
                                                         'receipt': receipt, 'utc': utc_now()})
            record_finalization(receipt,False,final_wall,final_cpu)
            return receipt

    def read(self, receipt: Receipt) -> tuple[Path, dict]:
        if set(receipt) != {'job_key', 'path', 'manifest_sha256'}:
            raise ValueError('Invalid calibration receipt fields')
        folder = safe_path(self.output_root, receipt['path'])
        try:
            manifest_path = folder / 'manifest.json'
            if sha256_file(manifest_path) != receipt['manifest_sha256']:
                raise ValueError('Study manifest hash mismatch')
            marker = read_json(folder / 'COMPLETED')
            if marker != {'manifest_sha256': receipt['manifest_sha256']}:
                raise ValueError('Study completion marker hash mismatch')
            manifest = read_json(manifest_path)
            if manifest['bindings'] != self.bindings or manifest['protocol_id'] != PROTOCOL_ID:
                raise ValueError('Study binding mismatch')
            identity = self._identity(manifest['stage'], manifest['request'])
            key = job_key(identity)
            if key != receipt['job_key'] or key != manifest['job_key']:
                raise ValueError('Study job identity hash mismatch')
            if folder != self.output_root / 'jobs' / manifest['stage'] / key:
                raise ValueError('Study receipt coordinate path mismatch')
            if read_json(folder / 'request.json') != identity:
                raise ValueError('Study request hash mismatch')
            actual = {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file()
                      and path not in {folder / 'manifest.json', folder / 'COMPLETED'}}
            if actual != set(manifest['files']):
                raise ValueError('Study produced file set changed')
            for relative, digest in manifest['files'].items():
                if sha256_file(safe_path(folder, relative)) != digest:
                    raise ValueError(f'Study artifact hash mismatch: {relative}')
        except (OSError, KeyError, TypeError) as error:
            raise ValueError('Incomplete or malformed study receipt') from error
        return folder, manifest
