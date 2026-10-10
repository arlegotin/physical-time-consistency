The nominal-selection integrity audit will recompute the recorded validation scores of all 80 frozen direct A/shared checkpoints in the forty-seed cohort. It does not retrain, select a different checkpoint, inspect fine-time outcomes or create another statistical look. Execution waits until every prospective checkpoint freeze exists.

The independent code review identified that the frozen dataset loader writes access events into its source context and does not accept the newer `audit_path` argument. The audit now calls that exact loader while redirecting only its event sink to an external journal, restores the sink in `finally`, and suppresses bytecode writes before frozen imports. The frozen source remains unchanged. Three targeted tests failed before this correction; all eight nominal-audit tests passed afterward. A real-context dry run found 69 of 80 checkpoint freezes and correctly listed the eleven missing cells without loading models or writing an audit result.

The reviewer confirmed the correction and found no remaining blocker in this bounded review. The score/count comparisons, validation-only access, archive/canonical tensor hashes and tensor immutability check were also reviewed. This is software verification; the actual eighty-checkpoint score audit has not run.

After the cohort completes, run from the repository root:

```bash
.venv/bin/python reports/audit_nominal_selections.py --root study/precision-look-2/study/precision-look-3/study/precision-look-4 --output reports/selection_integrity_forty.json
```
