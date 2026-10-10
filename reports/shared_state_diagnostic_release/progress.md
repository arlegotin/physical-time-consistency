# SDD ledger — plan: docs/superpowers/plans/2026-10-08-shared-state-diagnostic.md

Base: 69744dd. Current-branch execution explicitly authorized by the owner; no new worktree or repeated approval. The already documented shared-state diagnostic is the approved continuation. Preserve the five pre-existing runtime changes and all historical evidence.

Pre-flight: Task 1 produces decompose/model_endpoints/frozen probe arrays; Task 2 consumes these exact APIs and uses the same selected checkpoint identities. Task 2 produces complete raw reference/model arrays and reaggregation; Task 3 consumes the immutable manifest. No conflicting interfaces found.

Task 1: RED observed absent diagnostic module (ImportError); GREEN four analytic/branch/selection checks passed. Whole project suite running before commit. Prepare deliberately deferred until all producer code is committed.

Task 2: Full suite 960 passed in 132.76 seconds. Existing signed fixture caught a component slicing mistake in normalization; fixed the q/v slice at the caller and all eight focused tests passed.

Task 3: Initial prepare failed before writing a freeze because root corpus provenance references an earlier CLI hash (`Stale semantic dependency: poc/__main__.py`). Both established historical contexts successfully validate the same development corpus through the unchanged loader. Ruling: load from the pilot's original nominal context, requiring manifest and NPZ byte identity with the original root corpus and binding both copies. This preserves labels and strict historical gate validation rather than weakening the validator; cost if wrong is invalid provenance, guarded by a changed-copy rejection regression. No model outputs were evaluated by the failed prepare.

Task 3: Ruling: bind scientific spec and exact source/data/weights; keep mutable implementation checkbox/ledger progress outside scientific semantic inputs. This avoids changing experiment identity when marking tasks done; scientific design remains hash-bound. Cost if wrong: a plan-only substantive change would require checking the bound spec/source rather than the plan checksum.
Task 1: complete (commits 69744dd..caded11, tests: .venv/bin/python -m pytest -q tests/test_shared_state_diagnostic.py → 4 passed in 0.52s)
Task 2: complete (commits caded11..c736898, tests: .venv/bin/python -m pytest -q tests/test_shared_state_diagnostic.py → 8 passed in 0.57s)
Task 3: complete (commits c736898..7691048, tests: .venv/bin/python -m poc.shared_state_diagnostic verify → {'status': 'complete', 'measured_model_rows': 3072})

Final review: fresh gpt-6-astra read-only review; no critical code issue. Important causal wording corrected; numerical range and explicit split computation regraded as required report correctness and corrected in the same prose-only pass. No production change; 962-test suite remains applicable; independent 1536-statistic arithmetic and verify pass. Exhaustive rulings are in reports/shared_state_diagnostic_review.md.
