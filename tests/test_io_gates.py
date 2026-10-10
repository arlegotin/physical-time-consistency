"""Behavioral contracts for the standard-library execution envelope."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def api(self, module, name):
        try:
            imported = importlib.import_module(f"poc.{module}")
        except ImportError as exc:
            self.fail(f"Missing execution contract poc.{module}: {exc}")
        self.assertTrue(hasattr(imported, name), f"Missing {module}.{name}")
        return getattr(imported, name)

    def test_real_preflight_without_venv_preserves_blocker_and_does_not_install(self):
        project = Path(__file__).resolve().parents[1]
        for name in ('poc', 'config', 'schemas'):
            shutil.copytree(project / name, self.root / name)
        for name in ('requirements.in',):
            shutil.copyfile(project / name, self.root / name)
        # The public checkout excludes the real owner's allocation. Keep this
        # synthetic record confined to the temporary, non-executing fixture.
        self.api('io', 'atomic_json')(self.root / 'config/authorization.json', {
            'local_compute_authorized': True, 'package_network_authorized': True,
            'paid_services_authorized': False, 'ram_limit_gib': None,
            'disk_limit_gib': None, 'cpu_time_limit_hours': None,
            'authority_note': 'Synthetic unit-test fixture; grants no real execution authority',
            'recorded_utc': '2026-10-10T00:00:00Z',
        })
        result = subprocess.run([sys.executable, '-S', '-m', 'poc', 'run', 'M00'],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        manifests = list((self.root / 'evidence/M00').glob('*/manifest.json'))
        self.assertEqual(len(manifests), 1, result.stderr)
        record = self.api('io', 'read_json')(manifests[0])
        self.assertTrue(record['run_id'].startswith(record['started_utc'][:19].replace('-', '').replace(':', '')))
        self.api('contracts', 'verify_run')(self.root, record)
        self.assertEqual(record['completion_status'], 'blocked')
        validity = self.api('io', 'read_json')(manifests[0].parent / 'validity.json')
        self.assertEqual(validity['blocker']['check_id'], 'virtual_environment')
        self.assertFalse((self.root / 'wheelhouse').exists())
        events = (manifests[0].parent / 'events.jsonl').read_text()
        self.assertNotIn('command_start', events)

    def test_reference_dry_run_lists_cells_and_work_without_scientific_imports(self):
        result = subprocess.run([sys.executable,'-S','-m','poc','run','M20','--dry-run'],
                                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(len(value['expected_cells']),11)
        self.assertEqual(value['resource_projection']['mujoco_endpoint_branches'],55_296)
        self.assertTrue(value['expected_jobs'])
        self.assertFalse(value['execute'])

    def write_text(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def record(self, checks=None, validity="valid", blocker=None):
        """Synthetic fixture confined to a temporary root, never project evidence."""
        sha = self.api("io", "sha256_file")
        atomic = self.api("io", "atomic_json")
        key = self.api("io", "job_key")
        folder = self.root / "evidence/M00/fixture"
        folder.mkdir(parents=True, exist_ok=True)
        defaults = [
            "authorization", "python", "architecture", "virtual_environment",
            "disk", "packages", "determinism", "nstep", "torch_smoke",
            "wheel_lock", "offline_install", "validators",
        ]
        checks = checks or [{"check_id": c, "status": "pass", "reason": "Synthetic fixture",
                             "tolerance": None, "scope": "M00"} for c in defaults]
        for name, value in {
            "validity.json": {"validity_status": validity, "checks": checks, "blocker": blocker},
            "summary.json": {"measurements": {}, "missing_cells": [], "blocker": blocker},
            "resources.json": {"wall_seconds": 0.0, "cpu_seconds": 0.0, "peak_rss_bytes": 0},
        }.items():
            atomic(folder / name, value)
        for name in ("stdout.log", "stderr.log", "events.jsonl"):
            (folder / name).write_text("")
        source = self.write_text("fixture_source.py", "fixture = True\n")
        inputs = {"protocol_version": "PTCO-1.0", "dependencies": {"fixture_source.py": sha(source)}}
        return {
            "schema_version": 1, "protocol_version": "PTCO-1.0",
            "milestone_id": "M00", "run_id": "fixture", "job_key": key(inputs),
            "job_inputs": inputs, "phase": "preflight", "argv": ["poc", "run", "M00"],
            "code_revision": "fixture", "code_tree_sha256": sha(source), "dirty": False,
            "configuration_sha256": sha(source), "data_hashes": {}, "model_hashes": {},
            "environment_sha256": sha(source), "parent_decisions": [],
            "started_utc": "2026-09-30T00:00:00Z", "ended_utc": "2026-09-30T00:00:01Z",
            "completion_status": "complete", "expected_cells": defaults,
            "produced_files": [{"path": str(p.relative_to(self.root)), "sha256": sha(p)}
                               for p in sorted(folder.iterdir())],
        }

    def test_nonfinite_json_rejected(self):
        read = self.api("io", "read_json")
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                read(self.write_text("bad.json", '{"metric":' + token + '}'))
        atomic = self.api("io", "atomic_json")
        with self.assertRaises(ValueError):
            atomic(self.root / "bad-write.json", {"x": float("nan")})
        self.assertFalse((self.root / "bad-write.json").exists())

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            self.api("io", "read_json")(self.write_text("duplicate.json", '{"x":1,"x":2}'))

    def test_atomic_write_preserves_previous_value_on_failure(self):
        atomic = self.api("io", "atomic_json")
        path = self.root / "state.json"
        atomic(path, {"count": 1})
        with patch("poc.io.os.replace", side_effect=OSError("interrupted rename")):
            with self.assertRaises(OSError):
                atomic(path, {"count": 2})
        self.assertEqual(json.loads(path.read_text()), {"count": 1})
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_job_key_ignores_run_directory(self):
        key = self.api("io", "job_key")
        a = {"h_ns": 40_000_000, "output_directory": "/first", "started_utc": "yesterday"}
        b = {"h_ns": 40_000_000, "output_directory": "/second", "started_utc": "today"}
        self.assertEqual(key(a), key(b))
        self.assertNotEqual(key(a), key({**a, "h_ns": 5_000_000}))
        self.assertNotEqual(key({"dependencies": {"a.py": "1"}}),
                            key({"dependencies": {"a.py": "2"}}))

    def test_completion_marker_written_last(self):
        record = self.record()
        atomic = self.api("io", "atomic_json")
        complete = self.api("io", "complete_run")
        def fail_marker(path, value):
            if path.name == "COMPLETED":
                self.assertTrue((path.parent / "manifest.json").is_file())
                raise OSError("power loss before completion")
            return atomic(path, value)
        with patch("poc.io.atomic_json", side_effect=fail_marker):
            with self.assertRaises(OSError):
                complete(self.root, record)
        self.assertFalse((self.root / "evidence/M00/fixture/COMPLETED").exists())
        with self.assertRaises(ValueError):
            self.api("contracts", "verify_run")(self.root, record)

    def test_completed_run_cannot_be_overwritten(self):
        record = self.record()
        complete = self.api("io", "complete_run")
        complete(self.root, record)
        with self.assertRaises(FileExistsError):
            complete(self.root, {**record, "ended_utc": "2026-10-01T00:00:00Z"})

    def test_superseded_preflight_is_archived_without_authorizing_current_work(self):
        old = self.record()
        old['completion_status'] = 'blocked'
        self.api('io', 'complete_run')(self.root, old)
        self.write_text('fixture_source.py', 'fixture = "repaired"\n')
        new = self.record()
        source = self.write_text('fixture_source.py', 'fixture = "repaired"\n')
        new['job_inputs']['dependencies']['fixture_source.py'] = self.api('io', 'sha256_file')(source)
        new['job_key'] = self.api('io', 'job_key')(new['job_inputs'])
        new['run_id'] = 'replacement'
        new['supersedes'] = ['fixture']
        new['produced_files'] = [item for item in new['produced_files'] if Path(item['path']).name not in {'manifest.json', 'COMPLETED'}]
        new_folder = self.root / 'evidence/M00/replacement'
        new_folder.mkdir()
        for artifact in new['produced_files']:
            source = self.root / artifact['path']
            destination = new_folder / source.name
            shutil.copyfile(source, destination)
            artifact['path'] = str(destination.relative_to(self.root))
        # Restore immutable original artifacts accidentally recreated by the fixture helper.
        for artifact in old['produced_files']:
            self.assertEqual(self.api('io', 'sha256_file')(self.root / artifact['path']), artifact['sha256'])
        self.api('io', 'complete_run')(self.root, new)
        records, problems = self.api('contracts', 'scan_runs')(self.root)
        self.assertEqual(problems, [])
        self.assertEqual(len(records), 2)
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.api('contracts', 'verify_run')(self.root, old)

    def test_excluded_wheel_bytes_are_part_of_hash_closure(self):
        record = self.record()
        wheel = self.write_text('wheelhouse/package.whl', 'binary fixture')
        record['job_inputs']['wheel_hashes'] = {wheel.name: self.api('io', 'sha256_file')(wheel)}
        record['job_key'] = self.api('io', 'job_key')(record['job_inputs'])
        self.api('io', 'complete_run')(self.root, record)
        self.write_text('wheelhouse/package.whl', 'tampered')
        with self.assertRaisesRegex(ValueError, 'Wheel'):
            self.api('contracts', 'verify_run')(self.root, record)

    def test_stale_ancestor_invalidates_child_even_when_parent_json_is_unchanged(self):
        old = self.record()
        self.api('io','complete_run')(self.root,old)
        decision = self.api('gates','decide')('G00',[old],self.root)
        child = self.record()
        child['produced_files'] = [entry for entry in child['produced_files'] if Path(entry['path']).name not in {'manifest.json','COMPLETED'}]
        child['milestone_id'],child['run_id'],child['phase'] = 'M10','child','unit'
        child['parent_decisions'] = [decision]
        source = self.write_text('child.py','child = True\n')
        parent_path = f'decisions/G00/{decision["decision_id"]}.json'
        child['job_inputs'] = {'protocol_version':'PTCO-1.0','dependencies':{
            'child.py':self.api('io','sha256_file')(source),parent_path:self.api('io','sha256_file')(self.root / parent_path)}}
        child['job_key'] = self.api('io','job_key')(child['job_inputs'])
        folder = self.root / 'evidence/M10/child'
        folder.mkdir(parents=True)
        for entry in child['produced_files']:
            destination = folder / Path(entry['path']).name
            shutil.copyfile(self.root / entry['path'],destination)
            entry['path'] = str(destination.relative_to(self.root))
        self.api('io','complete_run')(self.root,child)
        self.api('contracts','verify_run')(self.root,child)
        self.write_text('fixture_source.py','parent = "changed"\n')
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.api('contracts','verify_run')(self.root,child)

    def test_live_job_lock_prevents_duplicate_execution(self):
        lock = self.api("io", "job_lock")
        with lock("a" * 64, self.root):
            with self.assertRaises(RuntimeError):
                with lock("a" * 64, self.root):
                    self.fail("A duplicate job acquired a live lock")
        with lock("a" * 64, self.root):
            pass

    def test_foreign_job_lock_is_never_reclaimed_by_age(self):
        lock = self.api("io", "job_lock")
        path = self.root / "locks/active" / ("b" * 64 + ".json")
        self.api("io", "atomic_json")(path, {
            "hostname": "another-host", "pid": 99999999, "heartbeat": "1900-01-01T00:00:00Z",
            "job_key": "b" * 64, "process_start_identifier": "old", "owner_token": "foreign",
        })
        with self.assertRaises(RuntimeError):
            with lock("b" * 64, self.root):
                pass
        self.assertTrue(path.exists())

    def test_schema_example_with_null_provenance_is_invalid(self):
        record = self.record()
        record["code_tree_sha256"] = None
        with self.assertRaises(ValueError):
            self.api("contracts", "validate_record")("manifest", record, self.root)

    def test_missing_evidence_never_go(self):
        decision = self.api("gates", "decide")("G00", [], self.root)
        self.assertEqual(decision["outcome"], "INVALID")
        self.assertIsNone(decision["terminal_verdict"])

    def test_partial_and_tampered_evidence_is_invalid_before_gate(self):
        record = self.record()
        decide = self.api("gates", "decide")
        self.assertEqual(decide("G00", [record], self.root)["outcome"], "INVALID")
        self.api("io", "complete_run")(self.root, record)
        self.write_text("evidence/M00/fixture/summary.json", '{"changed":true}')
        self.assertEqual(decide("G00", [record], self.root)["outcome"], "INVALID")

    def test_semantic_source_change_invalidates_evidence(self):
        record = self.record()
        self.api("io", "complete_run")(self.root, record)
        self.write_text("fixture_source.py", "fixture = False\n")
        self.assertEqual(self.api("gates", "decide")("G00", [record], self.root)["outcome"], "INVALID")

    def test_all_five_gate_outcomes(self):
        for wanted, status in [("GO", "pass"), ("NO-GO", "fail"),
                               ("INCONCLUSIVE", "unresolved"), ("INVALID", "invalid"),
                               ("BLOCKED", "blocked")]:
            with self.subTest(outcome=wanted):
                isolated = self.root / wanted
                isolated.mkdir()
                previous = self.root
                self.root = isolated
                try:
                    record = self.record()
                    path = isolated / "evidence/M00/fixture/validity.json"
                    validity = json.loads(path.read_text())
                    validity["checks"][0]["status"] = status
                    validity["checks"][0]["failure_kind"] = "backend"
                    if status == "blocked":
                        validity["blocker"] = {"required_prerequisite": "authorized allocation",
                                               "reason": "Resource owner has not supplied limits"}
                    self.api("io", "atomic_json")(path, validity)
                    for entry in record["produced_files"]:
                        entry["sha256"] = self.api("io", "sha256_file")(isolated / entry["path"])
                    self.api("io", "complete_run")(isolated, record)
                    decision = self.api("gates", "decide")("G00", [record], isolated)
                    self.assertEqual(decision["outcome"], wanted)
                    self.assertIsNone(decision["terminal_verdict"])
                finally:
                    self.root = previous

    def test_gate_decisions_are_idempotent(self):
        record = self.record()
        self.api("io", "complete_run")(self.root, record)
        decide = self.api("gates", "decide")
        self.assertEqual(decide("G00", [record], self.root), decide("G00", [record], self.root))
        self.assertEqual(len(list((self.root / "decisions/G00").glob("*.json"))), 1)

    def test_path_escape_cannot_supply_evidence(self):
        record = self.record()
        record["produced_files"][0]["path"] = "../outside.json"
        with self.assertRaises(ValueError):
            self.api("contracts", "validate_record")("manifest", record, self.root)

    def test_cli_help_without_scientific_imports(self):
        for argv in (["--help"], ["run", "MF", "--help"]):
            result = subprocess.run([sys.executable, "-S", "-m", "poc", *argv], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_cli_rejects_unknown_milestone(self):
        result = subprocess.run([sys.executable, "-S", "-m", "poc", "run", "M99"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_empty_report_is_nonterminal_without_measured_claims(self):
        result = self.api("report", "write_report")(self.root, [])
        self.assertEqual(result["status"], "NONTERMINAL")
        text = (self.root / "reports/final_report.md").read_text()
        self.assertIn("not run", text)
        self.assertNotIn("POC PASS", text)

    def wheel(self, name="dummy", version="1.0", tag="py3-none-any", metadata_version=None):
        folder = self.root / "wheelhouse"
        folder.mkdir(exist_ok=True)
        path = folder / f"{name}-{version}-{tag}.whl"
        info = f"{name}-{version}.dist-info"
        files = {
            f"{name}/__init__.py": "value = 1\n",
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {metadata_version or version}\nLicense: MIT\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: PTCO-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{info}/licenses/LICENSE": "Synthetic fixture license notice\n",
        }
        files[f"{info}/RECORD"] = "".join(f"{p},,\n" for p in [*files, f"{info}/RECORD"])
        with zipfile.ZipFile(path, "w") as archive:
            for entry, value in files.items():
                archive.writestr(entry, value)
        return path

    def test_wheel_lock_uses_metadata_and_records_license_hashes(self):
        wheel = self.wheel()
        result = self.api("bootstrap", "lock_wheels")(wheel.parent, [("dummy", "1.0")])
        self.assertIn("dummy==1.0 --hash=sha256:" + self.api("io", "sha256_file")(wheel), result["requirements_lock"])
        self.assertEqual(result["wheels"][0]["license"], "MIT")
        self.assertTrue(result["wheels"][0]["license_files"])

    def test_wheel_lock_rejects_filename_metadata_version_mismatch(self):
        wheel = self.wheel(metadata_version="2.0")
        with self.assertRaises(ValueError):
            self.api("bootstrap", "lock_wheels")(wheel.parent, [("dummy", "1.0")])

    def test_wheel_lock_rejects_conflicting_versions(self):
        self.wheel(version="1.0")
        self.wheel(version="2.0")
        with self.assertRaises(ValueError):
            self.api("bootstrap", "lock_wheels")(self.root / "wheelhouse", [("dummy", "1.0")])

    def test_wheel_lock_rejects_omitted_installer_distribution(self):
        wheel = self.wheel()
        with self.assertRaises(ValueError):
            self.api("bootstrap", "lock_wheels")(wheel.parent, [("dummy", "1.0"), ("pip", "25.0")])

    def test_wheel_lock_rejects_incompatible_architecture(self):
        wheel = self.wheel(tag="cp311-cp311-win_amd64")
        with self.assertRaises(ValueError):
            self.api("bootstrap", "lock_wheels")(wheel.parent, [("dummy", "1.0")])

    def test_owner_declared_unlimited_resources_are_not_replaced_by_caps(self):
        authority = {"local_compute_authorized": True, "package_network_authorized": True,
                     "paid_services_authorized": False, "ram_limit_gib": None, "disk_limit_gib": None,
                     "cpu_time_limit_hours": None, "authority_note": "Owner imposed no caps",
                     "recorded_utc": "2026-09-30T00:00:00Z"}
        limits = self.api("bootstrap", "authorization_limits")(authority)
        self.assertIsNone(limits["ram_limit_gib"])
        self.assertIsNone(limits["disk_limit_gib"])
        self.assertIsNone(limits["cpu_time_limit_hours"])

    def test_unauthorized_package_access_is_rejected_before_acquisition(self):
        authority = {"local_compute_authorized": True, "package_network_authorized": False,
                     "paid_services_authorized": False, "ram_limit_gib": None, "disk_limit_gib": None,
                     "cpu_time_limit_hours": None, "authority_note": "Offline only",
                     "recorded_utc": "2026-09-30T00:00:00Z"}
        with self.assertRaises(PermissionError):
            self.api("bootstrap", "authorization_limits")(authority)
        self.assertFalse((self.root / ".venv").exists())

    def test_offline_lock_failure_preserves_existing_environment(self):
        wheel = self.wheel()
        lock = self.api("bootstrap", "lock_wheels")(wheel.parent, [("dummy", "1.0")])
        self.write_text("locks/requirements.lock", lock["requirements_lock"])
        self.write_text(".venv/owner-file", "preserve this environment")
        with wheel.open("ab") as out:
            out.write(b"changed after locking")
        result = self.api("bootstrap", "verify_offline_lock")(self.root)
        self.assertEqual(result["status"], "fail")
        self.assertNotEqual(result["returncode"], 0)
        self.assertEqual((self.root / ".venv/owner-file").read_text(), "preserve this environment")
        self.assertFalse(list(self.root.glob(".offline-venv-*")))

    def test_cached_decision_cannot_change_outcome_without_detection(self):
        record = self.record()
        self.api("io", "complete_run")(self.root, record)
        decide = self.api("gates", "decide")
        original = decide("G00", [record], self.root)
        path = self.root / "decisions/G00" / (original["decision_id"] + ".json")
        self.api("io", "atomic_json")(path, {**original, "outcome": "NO-GO"})
        with self.assertRaises(ValueError):
            decide("G00", [record], self.root)

    def test_irrelevant_config_growth_preserves_evidence_but_relevant_change_does_not(self):
        path = self.write_text("config/criteria.json", '{"statistical_family_size":280,"criteria":[]}')
        record = self.record()
        record["job_inputs"]["configuration_keys"] = {"config/criteria.json": {"statistical_family_size": 280}}
        record["job_key"] = self.api("io", "job_key")(record["job_inputs"])
        self.api("io", "complete_run")(self.root, record)
        path.write_text('{"statistical_family_size":280,"criteria":["new declared criterion"]}')
        self.api("contracts", "verify_run")(self.root, record)
        path.write_text('{"statistical_family_size":279,"criteria":[]}')
        with self.assertRaises(ValueError):
            self.api("contracts", "verify_run")(self.root, record)

    def protocol_fixture(self):
        project = Path(__file__).resolve().parent.parent
        for name in ("base.json", "milestones.json", "criteria.json"):
            self.write_text("config/" + name, (project / "config" / name).read_text())

    def test_protocol_rejects_unversioned_primary_endpoint_change(self):
        self.protocol_fixture()
        path = self.root / "config/base.json"
        base = json.loads(path.read_text())
        base["primary_model_step_s"] = .01
        path.write_text(json.dumps(base))
        with self.assertRaises(ValueError):
            self.api("contracts", "validate_protocol")(self.root)

    def test_protocol_rejects_unknown_configuration_key(self):
        self.protocol_fixture()
        path = self.root / "config/base.json"
        base = json.loads(path.read_text())
        base["unregistered_override"] = True
        path.write_text(json.dumps(base))
        with self.assertRaises(ValueError):
            self.api("contracts", "validate_protocol")(self.root)


if __name__ == "__main__":
    unittest.main()
