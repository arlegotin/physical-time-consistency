"""Immutable receipts for individual training fits and analytic plumbing controls."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import resource
import subprocess
import sys
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path

from .contracts import verify_run
from .io import append_event,atomic_json,canonical_json,complete_run,job_key,job_lock,read_json,safe_path,sha256_file,utc_now

SOURCES = ('poc/__init__.py','poc/__main__.py','poc/io.py','poc/contracts.py','poc/gates.py','poc/execution.py',
           'poc/train.py','poc/training_jobs.py','poc/dataset.py','poc/history.py','poc/analytic.py','poc/clocks.py',
           'poc/research_jobs.py',
           'poc/models/__init__.py','poc/models/common.py','poc/models/power.py','poc/models/mlp.py',
           'poc/models/sequence.py','poc/models/ode.py','poc/models/factory.py','poc/envs.py',
           'poc/oracle_rigidbody.py','poc/sensitivity.py','poc/milestones/__init__.py','poc/milestones/m10.py',
           'poc/milestones/m20.py')


def find_partial(root,key):
    root = Path(root).resolve()
    matches, edges = {}, {}
    for path in sorted((Path(root) / 'evidence/M30').glob('*/request.json')):
        request = read_json(path)
        if request['job_key'] == key and not (path.parent / 'COMPLETED').exists():
            matches[path.parent.name] = path.parent
            if request.get('resume_from'):
                ancestor = safe_path(root, request['resume_from'])
                if ancestor.parent != root / 'evidence/M30':
                    raise ValueError('Resume ancestry must stay inside this training context')
                edges.setdefault(ancestor.name, set()).add(path.parent.name)
            marker = path.parent / 'RESUMED_BY.json'
            if marker.exists():
                link = read_json(marker)
                if link['job_key'] != key:
                    raise ValueError('Resume ancestry has a conflicting semantic identity')
                edges.setdefault(path.parent.name,set()).add(link['run_id'])
    if not matches:
        return None
    # Every request records its ancestor atomically, so a replacement without a
    # checkpoint cannot hide the prior committed state, even before its first epoch.
    parents = {child:parent for parent,children in edges.items() for child in children if child in matches}
    if any(len([p for p,c in edges.items() if child in c]) > 1 for child in matches):
        raise ValueError('Conflicting multiple resume ancestors')
    roots = set()
    depths = {}
    for name in matches:
        current, seen, depth = name, set(), 0
        while current in parents:
            if current in seen:
                raise ValueError('Cyclic partial training ancestry')
            seen.add(current); current = parents[current]; depth += 1
        roots.add(current); depths[name] = depth
    if len(roots) != 1:
        raise ValueError('Conflicting multiple partial jobs for one semantic identity')
    committed = []
    for name, folder in matches.items():
        for stem in ('resume', 'resume.prev'):
            path, marker, rng = folder / (stem+'.pt'), folder / (stem+'.complete.json'), folder / (stem+'.rng.json')
            try:
                commit = read_json(marker)
                if commit['checkpoint_sha256'] == sha256_file(path) and commit['rng_sha256'] == sha256_file(rng):
                    committed.append((depths[name], name)); break
            except (ValueError,OSError,KeyError):
                continue
    candidates = committed or [(depths[name],name) for name in matches]
    deepest = max(depth for depth,name in candidates)
    names = [name for depth,name in candidates if depth == deepest]
    if len(names) != 1:
        raise ValueError('Conflicting multiple committed partial branches')
    return matches[names[0]]


def validate_data_inputs(root,hashes):
    for path,expected in hashes.items():
        if sha256_file(safe_path(root,path)) != expected:
            raise ValueError('Training input hash mismatch: '+path)
    return dict(hashes)


def training_job(root,cell,phase,parameters,expected,parents,producer,resume=False):
    root = Path(root).resolve()
    environment = read_json(root / 'locks/environment.json')
    for name,version in environment['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Installed package differs from pinned training environment: '+name)
    imported = {str(Path(module.__file__).resolve().relative_to(root)) for module in list(sys.modules.values())
          if getattr(module,'__file__',None) and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(root / 'poc')}
    if imported-set(SOURCES):
        raise ValueError('Undeclared local training imports: '+str(sorted(imported-set(SOURCES))))
    dependencies = {filename:sha256_file(root / filename) for filename in SOURCES}
    dependencies.update(validate_data_inputs(root,parameters.get('data_hashes',{})))
    for decision in parents:
        filename = f'decisions/{decision["gate_id"]}/{decision["decision_id"]}.json'
        dependencies[filename] = sha256_file(root / filename)
    fingerprint = {key:environment[key] for key in ('python','architecture','os','kernel','core_count','versions',
                   'torch_configuration','torch_threads','torch_interop_threads','deterministic_algorithms')}
    inputs = {'protocol_version':'PTCO-1.0','milestone_id':'M30','cell_id':cell,'phase':phase,
              'parameters':parameters,'dependencies':dependencies,'environment_fingerprint':fingerprint,
              'wheel_hashes':{wheel['filename']:wheel['sha256'] for wheel in read_json(root / 'locks/wheels.json')['wheels']}}
    key = job_key(inputs)
    with job_lock(key,root):
        for path in sorted((root / 'evidence/M30').glob('*/manifest.json')):
            prior = read_json(path)
            if prior['job_key'] == key and prior['completion_status'] == 'complete':
                verify_run(root,prior)
                return prior
        partial = find_partial(root,key)
        if partial and not resume:
            raise ValueError('Compatible interrupted training exists; use --resume')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
        folder = root / 'evidence/M30' / run_id
        folder.mkdir(parents=True)
        for name in ('stdout.log','stderr.log','events.jsonl'):
            (folder / name).touch()
        atomic_json(folder / 'request.json',{'job_key':key,'inputs':inputs,'cell_id':cell,
                    'resume_from':str(partial.relative_to(root)) if partial else None})
        started,wall,cpu = utc_now(),time.perf_counter_ns(),time.process_time()
        append_event(folder / 'events.jsonl',{'event':'training_job_start','cell_id':cell,'utc':started})
        try:
            checks,summary = producer(folder,partial)
        except (ValueError,RuntimeError,OSError,AssertionError) as error:
            checks = [{'check_id':criterion,'status':'invalid','reason':str(error),'scope':cell,'tolerance':None} for criterion in expected]
            summary = {'missing_cells':list(expected),'failure':str(error)}
            (folder / 'stderr.log').write_text(str(error)+'\n')
        valid = {row['check_id'] for row in checks} == set(expected) and all(row['status'] == 'pass' for row in checks)
        atomic_json(folder / 'validity.json',{'validity_status':'valid' if valid else 'invalid','checks':checks,'blocker':None})
        atomic_json(folder / 'summary.json',summary)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic_json(folder / 'resources.json',{'wall_seconds':(time.perf_counter_ns()-wall)/1e9,
           'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':int(peak*(1 if platform.system() == 'Darwin' else 1024)),
           'rss_convention':'macOS bytes' if platform.system() == 'Darwin' else 'Linux KiB normalized to bytes',
           'disk_bytes':sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()),
           'training_updates':summary.get('training_updates',0),'original_estimate':parameters.get('resource_projection')})
        append_event(folder / 'events.jsonl',{'event':'training_job_end','valid':valid,'utc':utc_now()})
        revision = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
        dirty = subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True)
        record = {'schema_version':1,'protocol_version':'PTCO-1.0','milestone_id':'M30','run_id':run_id,
            'job_key':key,'job_inputs':inputs,'phase':phase,'argv':[sys.executable,*sys.argv],
            'code_revision':revision.stdout.strip(),'dirty':bool(dirty.stdout),
            'code_tree_sha256':hashlib.sha256(canonical_json(dependencies)).hexdigest(),
            'configuration_sha256':hashlib.sha256(canonical_json(parameters)).hexdigest(),
            'environment_sha256':hashlib.sha256(canonical_json(fingerprint)).hexdigest(),
            'data_hashes':parameters.get('data_hashes',{}),'model_hashes':summary.get('model_hashes',{}),
            'parent_decisions':parents,'started_utc':started,'ended_utc':utc_now(),
            'completion_status':'complete' if valid else 'failed','expected_cells':list(expected),
            'produced_files':[{'path':str(path.relative_to(root)),'sha256':sha256_file(path)} for path in sorted(folder.rglob('*')) if path.is_file()]}
        if valid and phase == 'unit':
            previous_ids = []
            for path in sorted((root / 'evidence/M30').glob('*/manifest.json')):
                previous = read_json(path)
                if previous['phase'] == 'unit' and previous['job_inputs'].get('cell_id') == cell:
                    verify_run(root,previous,check_dependencies=False)
                    previous_ids.append(previous['run_id'])
            if previous_ids:
                record['supersedes'] = previous_ids
                record['reason'] = 'This analytic plumbing control was repeated under current source; earlier attempts remain archived'
        complete_run(root,record)
        return record
