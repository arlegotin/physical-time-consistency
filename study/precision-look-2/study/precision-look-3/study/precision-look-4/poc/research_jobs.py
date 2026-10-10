"""One explicitly requested research job per fresh, deterministic CPU process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'


def main(argv=None):
    from .io import read_json,run_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('generate','train','E01','E02','controls'))
    parser.add_argument('request',type=Path)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    request = read_json(args.request)
    from .execution import cpu_defaults
    cpu_defaults(request.get('semantic_inputs',request).get('seed',9000))
    if args.command == 'generate':
        from .generate import generate_corpus
        record = generate_corpus(root,**request)
    elif args.command == 'controls':
        from .train import run_learning_controls
        record = run_learning_controls(request['model_id'],request['actuator'],root,resume=args.resume)
    elif args.command == 'train':
        from .train import fit
        record = fit(request,root,resume=args.resume)
    elif args.command == 'E01':
        from .model_sensitivity import run_e01
        record = run_e01(request,root,resume=args.resume)
    else:
        from .prediction import run_e02
        record = run_e02(request,root,resume=args.resume)
    print(json.dumps({'manifest_path':str((run_directory(root,record)/'manifest.json').relative_to(root)),
                      'run_id':record['run_id'],'completion_status':record['completion_status']},allow_nan=False),flush=True)
    return 0 if record['completion_status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
