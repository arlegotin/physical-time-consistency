### Task 1: Tested decomposition and frozen probe inputs

**Files:** Create `poc/shared_state_diagnostic.py`, `tests/test_shared_state_diagnostic.py`; create `config/shared_state_diagnostic_freeze.json`, `data/diagnostic_inputs/<digest>.npz` through prepare only.

**Interfaces:** Consumes existing `make_probe_library`, `load_corpus`, pilot selected identities, `predict` API. Produces `decompose(model_zero, model_command, truth_zero, truth_command, composed)` and `prepare(root)` with hash-bound probe coordinates and selected identities.

- [ ] Write tests that catch sign errors/cancellation, fixed lagged initial activation through zero/nonzero and half-step queries, and short/altered probe inputs.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_shared_state_diagnostic.py`; expected failure because diagnostic module is absent.
- [ ] Implement pure decomposition, model endpoint/composition queries, frozen selection and input validation.
- [ ] Run the same tests; expected pass. Prepare must not evaluate model outputs. Commit the additive code/spec/plan/tests.

