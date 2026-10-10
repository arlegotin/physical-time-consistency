### Task 2: References, immutable measurement and report

**Files:** Extend only new module/tests; register `config/revisions/MX-shared-state-0001.json`.

**Interfaces:** Consumes Task 1 freeze and probes; existing Plant, independent integrate, selected model loader and run_job. Produces raw endpoint/error arrays, complete reference convergence/status arrays, summary rows and `reports/shared_state_diagnostic_result.{json,md}` bound to the immutable manifest.

- [ ] Add failing tests for uncertainty propagation, signal resolution, changed artifact hash and missing model coordinates; run focused tests, expected failure.
- [ ] Implement reference sweep, measurement, fail-closed raw aggregation/reaggregation and CLI `prepare/run/report/verify`. All four models use the same frozen arrays; report physical errors and paired comparisons by plant/stratum/horizon.
- [ ] Run focused tests and `.venv/bin/python -m pytest -q`; expected all pass. Commit tested producer before freezing/running it.

