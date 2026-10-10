#!/usr/bin/env bash
set -euo pipefail
PTCO_WORKDIR=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd -- "$PTCO_WORKDIR"
.venv/bin/python - <<'PY'
from pathlib import Path
import time
from poc.io import read_json
path=Path('reports/control_nominal_completion.json')
previous=None
while True:
    record=read_json(path) if path.exists() else {'state':'pending'}
    state=record['state']
    if state != previous:
        print('Original nominal prerequisite:',state,flush=True)
        previous=state
    if state=='complete':
        break
    if state in {'unqualified','interrupted'}:
        raise RuntimeError('Original nominal cohort needs its recorded diagnosis/resumption: '+record.get('reason',''))
    time.sleep(30)
PY
.venv/bin/python -m poc.remaining_development --prepare
.venv/bin/python -m poc.remaining_development --resume
.venv/bin/python -m poc.utility_training --resume
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
from poc.io import read_json
root=Path.cwd()
design=read_json(root/'config/control_confirmation_design.json')
projection=design['resource_projection']['estimated_artifact_bytes']
available=shutil.disk_usage(root).free
print({'phase':'primary_control_batch','measured_proxy_bytes':projection,'actual_available_bytes':available},flush=True)
if available < projection:
    raise RuntimeError('Actual disk space is below the registered primary artifact projection; preserve evidence and provision more disk before resuming control. This is a resource prerequisite, not a scientific result.')
PY
.venv/bin/python -m poc.control_confirmation --resume
.venv/bin/python - <<'PY'
print('Primary paired control episodes complete. Independent numerical/reference audits and signed-cost reporting still required; no overall PoC verdict assigned.')
PY
