"""Explicit E01/E02 receipts using the existing immutable producer envelope."""
from pathlib import Path

from .execution import parent,run_job
from .io import atomic_json,job_key,read_json

SOURCES = ('poc/__init__.py','poc/__main__.py','poc/io.py','poc/contracts.py','poc/gates.py','poc/execution.py',
           'poc/analytic.py','poc/clocks.py','poc/dataset.py','poc/generate.py','poc/history.py','poc/envs.py',
           'poc/oracle_rigidbody.py','poc/sensitivity.py','poc/model_sensitivity.py','poc/metrics.py',
           'poc/criteria.py','poc/statistics.py','poc/criterion_gates.py','poc/probes.py','poc/prediction.py',
           'poc/audit_jobs.py','poc/models/__init__.py','poc/models/common.py','poc/models/power.py',
           'poc/research_jobs.py',
           'poc/models/mlp.py','poc/models/sequence.py','poc/models/ode.py','poc/models/factory.py',
           'poc/milestones/__init__.py','poc/milestones/m10.py','poc/milestones/m20.py')


def audit_job(root,experiment,job,metadata,checks,producer):
    root = Path(root)
    files = list(SOURCES)+list(metadata['input_hashes'])
    parent_decision = parent(root,'G20')
    request = {'experiment_id':experiment,**job['semantic_inputs']}
    path = 'config/audit_requests/'+job_key(request)+'.json'
    atomic_json(root/path,request)
    return run_job(root,'M30',job['phase'],list(dict.fromkeys(files)),checks,[parent_decision],producer,
                   config_keys={path:request,'config/numerical.json':read_json(root/'config/numerical.json')})
