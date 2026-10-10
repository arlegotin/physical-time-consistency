#!/usr/bin/env bash
set -euo pipefail
PTCO_PROJECT=$(cd -- "$(dirname -- "$0")/.." && pwd)
PTCO_PYTHON=${PTCO_PYTHON:-"$PTCO_PROJECT/.reproduction-venv/bin/python"}
if [[ ! -x "$PTCO_PYTHON" ]]; then
  python3.11 -m venv "$PTCO_PROJECT/.reproduction-venv"
  PTCO_PYTHON="$PTCO_PROJECT/.reproduction-venv/bin/python"
fi
cd -- "$PTCO_PROJECT"
"$PTCO_PYTHON" -m poc.study_reproduce --project "$PTCO_PROJECT" "$@"
