Run from the same public source revision with native CPython 3.11 and the pinned requirements. Use a fresh ignored output directory. Complete the pilot before explicit development expansion.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.in
CALIBRATION_OUTPUT="study/short-calibration"
.venv/bin/python -m poc.calibration_study prepare --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study collect --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study train --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study predict --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study decisions --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study time --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study report --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study verify --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study status --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
```

To reverify existing evidence without acquisition or fitting:

```sh
.venv/bin/python -m poc.calibration_study verify --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
.venv/bin/python -m poc.calibration_study report --root . --output-root "$CALIBRATION_OUTPUT" --matrix pilot
```

Prediction and candidate rows use deterministic level-1 gzip JSONL (rows.jsonl.gz); decode with Python gzip.open. Compression is lossless and occurs outside scorer timing. Raw arrays, selected weights and optimizer products stay in the selected ignored output store. Immutable receipts bind source, runtime, assets, recipe and selected tensors. Copy summaries together with that store to preserve external artifacts. The output path is bound into receipts; a moved store requires a fresh reproducible run, not edited bindings. Finalization costs use unique original receipt-bound observations from events.jsonl, outside immutable artifact hashes. Observation-record persistence is excluded.
