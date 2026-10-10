"""Actual CPU preflight, binary-wheel locking and offline recreation."""
from __future__ import annotations

from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from . import PROTOCOL_VERSION
from .contracts import BOOTSTRAP_CHECKS, validate_protocol, verify_run
from .io import (append_event, atomic_bytes, atomic_json, canonical_json, complete_run,
                 job_key, job_lock, read_json, run_directory, sha256_file, utc_now)

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
PINS = {"numpy": "2.2.6", "scipy": "1.15.3", "pytest": "8.4.1", "torch": "2.8.0", "mujoco": "3.14.0"}


def authorization_limits(authorization: dict) -> dict:
    required = {"local_compute_authorized", "package_network_authorized", "paid_services_authorized",
                "ram_limit_gib", "disk_limit_gib", "cpu_time_limit_hours", "authority_note", "recorded_utc"}
    if set(authorization) != required:
        raise ValueError("Authorization must have exactly the required allocation fields")
    if authorization["local_compute_authorized"] is not True:
        raise PermissionError("Local compute permission is absent")
    if authorization["package_network_authorized"] is not True:
        raise PermissionError("Package network permission is absent; an authorized offline wheelhouse may be provided")
    if authorization["paid_services_authorized"] is not False:
        raise ValueError("This protocol does not authorize paid services")
    if not isinstance(authorization["authority_note"], str) or not authorization["authority_note"].strip():
        raise ValueError("Allocation requires an actual authority note")
    datetime.fromisoformat(authorization["recorded_utc"].replace("Z", "+00:00"))
    limits = {key: authorization[key] for key in ("ram_limit_gib", "disk_limit_gib", "cpu_time_limit_hours")}
    for key, value in limits.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Invalid owner limit: {key}")
    return limits


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _compatible_wheel(filename: str) -> bool:
    try:
        _, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return False
    major, minor = sys.version_info[:2]
    python_ok = False
    for tag in python_tag.split("."):
        if tag in {f"py{major}", f"py{major}{minor}", f"cp{major}{minor}"}:
            python_ok = True
        elif abi_tag == "abi3" and tag.startswith(f"cp{major}") and tag[3:].isdigit() and int(tag[3:]) <= minor:
            python_ok = True
    if not python_ok or abi_tag not in {"none", "abi3", f"cp{major}{minor}"}:
        return False
    machine, system = platform.machine(), platform.system()
    for tag in platform_tag.split("."):
        if tag == "any":
            return True
        if system == "Darwin" and tag.startswith("macosx_") and tag.endswith(("_" + machine, "_universal2")):
            match = re.match(r"macosx_(\d+)_(\d+)_", tag)
            os_version = platform.mac_ver()[0]
            if match and os_version and tuple(map(int, match.groups())) <= tuple(map(int, os_version.split(".")[:2])):
                return True
        if system == "Linux" and tag.endswith("_" + machine):
            if tag.startswith("linux_"):
                return True
            if tag.startswith("manylinux"):
                glibc = platform.libc_ver()[1]
                match = re.match(r"manylinux_(\d+)_(\d+)_", tag)
                if match and glibc and tuple(map(int, match.groups())) <= tuple(map(int, glibc.split(".")[:2])):
                    return True
                if tag.startswith(("manylinux1_", "manylinux2010_", "manylinux2014_")) and glibc:
                    floor = (2, 17) if "2014" in tag else (2, 12) if "2010" in tag else (2, 5)
                    if tuple(map(int, glibc.split(".")[:2])) >= floor:
                        return True
    return False


def lock_wheels(wheel_dir: Path, resolved: list[tuple[str, str]]) -> dict:
    expected = {}
    for name, version in resolved:
        key = _name(name)
        if key in expected:
            raise ValueError(f"Duplicate resolved distribution: {name}")
        expected[key] = version
    wheels, seen = [], {}
    for path in sorted(wheel_dir.glob("*.whl")):
        if not _compatible_wheel(path.name):
            raise ValueError(f"Wheel does not match current Python/platform: {path.name}")
        with zipfile.ZipFile(path) as archive:
            metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                raise ValueError(f"Wheel has no unique distribution metadata: {path.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
            name, version = metadata.get("Name"), metadata.get("Version")
            if not name or not version:
                raise ValueError("Wheel METADATA omits Name/Version")
            key = _name(name)
            filename_prefix = path.name[:-4].rsplit("-", 3)[0].split("-")
            if len(filename_prefix) < 2 or _name(filename_prefix[0]) != key or filename_prefix[1] != version:
                raise ValueError(f"Filename/METADATA mismatch: {path.name}")
            if key in seen or expected.get(key) != version:
                raise ValueError(f"Conflicting/unresolved wheel: {name} {version}")
            seen[key] = version
            license_files = []
            notice_root = wheel_dir.parent / "locks/license_notices" / f"{key}-{version}"
            for member in archive.namelist():
                base = Path(member).name.lower()
                if member.endswith("/") or not base.startswith(("license", "licence", "notice", "copying")):
                    continue
                content = archive.read(member)
                notice_path = notice_root / (hashlib.sha256(member.encode()).hexdigest()[:12] + "-" + Path(member).name)
                atomic_bytes(notice_path, content)
                license_files.append({"wheel_path": member, "path": str(notice_path.relative_to(wheel_dir.parent)),
                                      "sha256": hashlib.sha256(content).hexdigest()})
            wheels.append({"name": name, "version": version, "filename": path.name, "sha256": sha256_file(path),
                           "license": metadata.get("License"), "license_expression": metadata.get("License-Expression"),
                           "license_files": license_files, "source_url": None})
    if expected.keys() - seen.keys():
        raise ValueError(f"Resolved distributions have no wheel: {sorted(expected.keys() - seen.keys())}")
    lock = "".join(f"{wheel['name']}=={wheel['version']} --hash=sha256:{wheel['sha256']}\n" for wheel in wheels)
    return {"requirements_lock": lock, "wheels": wheels}


def verify_offline_lock(root: Path) -> dict:
    root = root.resolve()
    logs = []
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(prefix=".offline-venv-", dir=root) as temporary:
        interpreter = Path(temporary) / "bin/python"
        commands = [[sys.executable, "-m", "venv", temporary],
                    [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
                     "--find-links", str(root / "wheelhouse"), "--require-hashes", "-r", str(root / "locks/requirements.lock")],
                    [str(interpreter), "-m", "pip", "check"]]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, cwd=root)
            logs.append({"argv": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                break
    value = {"status": "pass" if logs[-1]["returncode"] == 0 and len(logs) == 3 else "fail",
             "returncode": logs[-1]["returncode"], "commands": logs,
             "wall_seconds": (time.perf_counter_ns() - started) / 1e9}
    atomic_json(root / "locks/offline_install.json", value)
    return value


def _command(argv: list[str], root: Path, folder: Path) -> subprocess.CompletedProcess:
    append_event(folder / "events.jsonl", {"event": "command_start", "argv": argv, "utc": utc_now()})
    with (folder / "stdout.log").open("ab") as stdout, (folder / "stderr.log").open("ab") as stderr:
        stdout.write(canonical_json({"argv": argv}) + b"\n")
        stdout.flush()
        result = subprocess.run(argv, cwd=root, stdout=stdout, stderr=stderr)
    append_event(folder / "events.jsonl", {"event": "command_end", "argv": argv,
                                          "returncode": result.returncode, "utc": utc_now()})
    if result.returncode:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {argv!r}; full output in {folder / 'stderr.log'}")
    return result


def _system_fingerprint() -> dict:
    return {"python": sys.version, "executable": sys.executable, "architecture": platform.machine(),
            "os": platform.system(), "kernel": platform.release(), "platform": platform.platform(),
            "cpu_description": platform.processor() or platform.machine(), "core_count": os.cpu_count(),
            "virtual_environment": sys.prefix != sys.base_prefix,
            "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES}}


def _inputs(root: Path) -> dict:
    declared = {"poc/__init__.py", "poc/__main__.py", "poc/bootstrap.py", "poc/io.py", "poc/contracts.py",
                "poc/gates.py", "poc/report.py"}
    imported = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).resolve().is_relative_to(root / "poc") and name.endswith(".py"):
            relative = str(Path(name).resolve().relative_to(root))
            if relative not in declared:
                raise ValueError(f"Undeclared local bootstrap dependency: {relative}")
            imported[relative] = sha256_file(Path(name))
    dependencies = dict(imported)
    for relative in ("requirements.in", "POC_EXECUTION_PLAN.md", "config/protocol_provenance.json",
                     "schemas/manifest.schema.json", "schemas/decision.schema.json"):
        dependencies[relative] = sha256_file(root / relative)
    wheel_hashes = {}
    manifest = root / "locks/wheels.json"
    if manifest.exists():
        wheel_hashes = {entry["filename"]: entry["sha256"] for entry in read_json(manifest)["wheels"]}
    return {"protocol_version": PROTOCOL_VERSION, "milestone_id": "M00", "phase": "preflight",
            "dependencies": dependencies, "configuration_keys": {
                "config/base.json": read_json(root / "config/base.json"),
                "config/authorization.json": read_json(root / "config/authorization.json"),
                "config/criteria.json": {"protocol_version": PROTOCOL_VERSION, "statistical_family_size": 280},
                "config/milestones.json": {"protocol_version": PROTOCOL_VERSION}},
            "environment_fingerprint": _system_fingerprint(), "wheel_hashes": wheel_hashes,
            "intended_versions": PINS}


def _smoke(folder: Path) -> dict:
    import random
    import numpy as np
    import scipy
    import torch
    import mujoco
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    x = torch.tensor([1., 2.], dtype=torch.float32, requires_grad=True)
    (x * x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([2., 4.]))
    derivative = torch.autograd.functional.jacobian(lambda a: a * a,
                                                   torch.tensor([1., 2.], dtype=torch.float64))
    torch.testing.assert_close(derivative, torch.diag(torch.tensor([2., 4.], dtype=torch.float64)), atol=1e-12, rtol=1e-12)
    linear = torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=.001, weight_decay=.000001, betas=(.9, .999),
                                 eps=1e-8, amsgrad=False, foreach=False, fused=False)
    before = linear.weight.detach().clone()
    linear(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    if torch.equal(before, linear.weight):
        raise ValueError("AdamW smoke update changed no weights")
    checkpoint = folder / "smoke_weights.pt"
    torch.save(linear.state_dict(), checkpoint)
    recovered = torch.load(checkpoint, weights_only=True, map_location="cpu")
    for name, tensor in linear.state_dict().items():
        torch.testing.assert_close(tensor, recovered[name], atol=0, rtol=0)
    world = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    single, repeated = mujoco.MjData(world), mujoco.MjData(world)
    for _ in range(20):
        mujoco.mj_step(world, single)
    mujoco.mj_step(world, repeated, nstep=20)
    if abs(single.time - repeated.time) > 1e-15:
        raise ValueError("Installed mj_step nstep semantics differ")
    readings = [time.perf_counter_ns() for _ in range(10)]
    if readings != sorted(readings) or readings[-1] <= readings[0]:
        raise ValueError("Elapsed timer is not monotonic/resolvable")
    atomic_bytes(folder / "hash_fixture.txt", b"PTCO-1.0")
    if sha256_file(folder / "hash_fixture.txt") != hashlib.sha256(b"PTCO-1.0").hexdigest():
        raise ValueError("File hashing smoke check failed")
    distributions = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: _name(item.metadata["Name"])):
        licenses = []
        for relative in distribution.files or []:
            if Path(str(relative)).name.lower().startswith(("license", "licence", "notice", "copying")):
                path = distribution.locate_file(relative)
                if path.is_file():
                    licenses.append({"path": str(relative), "sha256": sha256_file(path)})
        distributions.append({"name": distribution.metadata["Name"], "version": distribution.version,
                              "license": distribution.metadata.get("License"),
                              "license_expression": distribution.metadata.get("License-Expression"), "license_files": licenses})
    return {**_system_fingerprint(), "pip_version": importlib.metadata.version("pip"),
            "versions": {name: importlib.metadata.version(name) for name in PINS},
            "torch_configuration": torch.__config__.show(), "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "rng_seeds": {"python": 17, "numpy": 18, "torch": 19}, "num_workers": 0,
            "device": "cpu", "train_dtype": "float32", "audit_dtype": "float64",
            "distributions": distributions, "nstep_smoke": {"single_time": single.time, "nstep_time": repeated.time},
            "float64_jacobian": derivative.tolist(), "timer_readings_ns": readings}


def run_m00(root: Path, authorization: dict) -> dict:
    root = root.resolve()
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    initial = _inputs(root)
    reservation = job_key({**initial, 'wheel_hashes': {}})
    with job_lock(reservation, root):
        for path in sorted((root / 'evidence/M00').glob('*/manifest.json')):
            previous = read_json(path)
            if previous['completion_status'] == 'complete' and previous['job_key'] == job_key(initial):
                verify_run(root, previous)
                return previous
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        folder = root / 'evidence/M00' / run_id
        folder.mkdir(parents=True)
        for name in ('stdout.log', 'stderr.log', 'events.jsonl'):
            (folder / name).touch()
        started_utc, started = utc_now(), time.perf_counter_ns()
        cpu_start = time.process_time()
        children_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        checks = {name: {'check_id': name, 'status': 'unresolved', 'reason': 'Not run',
                         'tolerance': None, 'scope': 'M00'} for name in BOOTSTRAP_CHECKS}
        blocker, environment, active = None, _system_fingerprint(), 'authorization'
        def passed(name: str, reason: str):
            checks[name].update(status='pass', reason=reason)
            append_event(folder / 'events.jsonl', {'event': 'check_pass', 'check_id': name, 'utc': utc_now()})
        try:
            limits = authorization_limits(authorization)
            passed(active, f'Actual owner allocation recorded: {limits}')
            active = 'python'
            if sys.version_info[:2] != (3, 11):
                raise RuntimeError('CPython 3.11 is required')
            passed(active, sys.version)
            active = 'architecture'
            if (platform.system(), platform.machine()) not in {('Darwin', 'arm64'), ('Linux', 'x86_64')}:
                raise RuntimeError('Authorized native macOS ARM64 or Linux x86-64 CPU host required')
            passed(active, platform.platform())
            active = 'virtual_environment'
            if sys.prefix == sys.base_prefix or not Path(sys.prefix).resolve().is_relative_to(root):
                raise RuntimeError('Activate the project-created Python 3.11 virtual environment before M00')
            passed(active, str(Path(sys.prefix).relative_to(root)))
            active = 'disk'
            free = shutil.disk_usage(root).free
            if free <= 0:
                raise RuntimeError('No writable disk space available')
            environment['available_disk_bytes_at_start'] = free
            passed(active, f'{free} available bytes; suggested estimates are not resource caps')
            active = 'packages'
            pip = [sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check']
            if platform.system() == 'Linux':
                _command(pip + ['install', '--only-binary=:all:', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.8.0'], root, folder)
                other = [f'{name}=={version}' for name, version in PINS.items() if name != 'torch']
                _command(pip + ['install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', *other], root, folder)
            else:
                _command(pip + ['install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple', '-r', 'requirements.in'], root, folder)
            _command(pip + ['check'], root, folder)
            for name, version in PINS.items():
                installed = importlib.metadata.version(name)
                accepted = {version, version + '+cpu'} if name == 'torch' and platform.system() == 'Linux' else {version}
                if installed not in accepted:
                    raise ValueError(f'Pinned package mismatch: {name} {installed}')
            passed(active, 'All exact pins installed and pip check passed')
            active = 'torch_smoke'
            environment = {**environment, **_smoke(folder)}
            passed(active, 'float32 backward, float64 autograd, AdamW update, safe checkpoint load, timers and hashing passed')
            passed('nstep', 'Installed empty-world nstep=20 matches twenty individual steps')
            passed('determinism', 'CPU, independently seeded RNGs, one thread, deterministic algorithms')
            atomic_json(root / 'locks/environment.json', environment)
            atomic_json(folder / 'environment.json', environment)
            active = 'wheel_lock'
            freeze = subprocess.run(pip + ['freeze', '--all'], cwd=root, capture_output=True, text=True, check=True)
            atomic_bytes(root / 'locks/requirements.resolved.txt', freeze.stdout.encode())
            resolved = []
            for line in freeze.stdout.splitlines():
                if not line or line.startswith('#'):
                    continue
                if line.count('==') != 1:
                    raise ValueError(f'Non-exact resolved distribution: {line}')
                resolved.append(tuple(line.split('==')))
            wheel_dir = root / 'wheelhouse'
            wheel_dir.mkdir(exist_ok=True)
            download_log = folder / 'pip-download.log'
            download = pip + ['--log', str(download_log), 'download', '--only-binary=:all:', '--no-deps', '-d', 'wheelhouse']
            if platform.system() == 'Linux':
                cpu = [f'{name}=={version}' for name, version in resolved if _name(name) == 'torch']
                others = [f'{name}=={version}' for name, version in resolved if _name(name) != 'torch']
                _command(download + ['--index-url', 'https://download.pytorch.org/whl/cpu', *cpu], root, folder)
                _command(download + ['--index-url', 'https://pypi.org/simple', *others], root, folder)
            else:
                _command(download + ['--index-url', 'https://pypi.org/simple', '-r', 'locks/requirements.resolved.txt'], root, folder)
            lock = lock_wheels(wheel_dir, resolved)
            download_text = download_log.read_text()
            from urllib.parse import unquote, urlsplit
            urls = re.findall(r'https://[^\s<>\"\)]+', download_text)
            for wheel in lock['wheels']:
                sources = {url.split('#')[0] for url in urls if Path(unquote(urlsplit(url).path)).name == wheel['filename']}
                trusted = sorted(url for url in sources if urlsplit(url).hostname in {'files.pythonhosted.org', 'download.pytorch.org'})
                if not trusted:
                    raise ValueError(f'No recorded official source URL for {wheel["filename"]}')
                wheel['source_url'] = trusted[0]
            atomic_bytes(root / 'locks/requirements.lock', lock['requirements_lock'].encode())
            atomic_json(root / 'locks/wheels.json', {'schema_version': 1, 'wheels': lock['wheels']})
            atomic_json(folder / 'wheels.json', {'schema_version': 1, 'wheels': lock['wheels']})
            passed(active, f'{len(resolved)} complete resolved distributions, exact compatible wheels, SHA-256 hashes and license notices')
            active = 'offline_install'
            offline = verify_offline_lock(root)
            atomic_json(folder / 'offline_install.json', offline)
            if offline['status'] != 'pass':
                raise RuntimeError('Fresh temporary venv hash-locked offline installation failed; full output preserved')
            passed(active, 'Fresh temporary venv installed solely from locked wheelhouse; pip check passed')
            active = 'validators'
            validate_protocol(root)
            passed(active, 'Protocol constants, milestone registry and strict serialization validate')
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as exc:
            checks[active].update(status='blocked' if active in {'authorization', 'python', 'architecture', 'virtual_environment', 'disk', 'packages'} else 'invalid',
                                  reason=str(exc))
            blocker = {'check_id': active, 'reason': str(exc), 'required_prerequisite':
                       f'Resolve {active} at the specified pins on an authorized CPU host; resume M00'}
            append_event(folder / 'events.jsonl', {'event': 'preflight_failure', 'check_id': active, 'error': str(exc), 'utc': utc_now()})
        valid = all(check['status'] == 'pass' for check in checks.values())
        blocked = blocker is not None and checks[active]['status'] == 'blocked'
        atomic_json(folder / 'environment.json', environment)
        atomic_json(folder / 'validity.json', {'validity_status': 'valid' if valid else 'blocked' if blocked else 'invalid',
                                             'checks': list(checks.values()), 'blocker': blocker})
        atomic_json(folder / 'summary.json', {'measurements': environment, 'scientific_experiments_run': False,
                                            'missing_cells': [key for key, value in checks.items() if value['status'] != 'pass'], 'blocker': blocker})
        usage, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        atomic_json(folder / 'resources.json', {'wall_seconds': (time.perf_counter_ns() - started) / 1e9,
                    'cpu_seconds': time.process_time() - cpu_start + children.ru_utime + children.ru_stime - children_start.ru_utime - children_start.ru_stime,
                    'peak_rss_bytes': int(max(usage.ru_maxrss, children.ru_maxrss) * (1 if platform.system() == 'Darwin' else 1024)),
                    'rss_convention': 'macOS bytes' if platform.system() == 'Darwin' else 'Linux KiB normalized to bytes',
                    'disk_bytes': sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
                    'original_estimate': {'working_ram_gib': 8, 'working_disk_gib': 10, 'status': 'estimate_not_a_cap'},
                    'network_calls': 0, 'candidate_evaluations': 0, 'training_updates': 0})
        inputs = _inputs(root)
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=root, capture_output=True, text=True)
        record = {'schema_version': 1, 'protocol_version': PROTOCOL_VERSION, 'milestone_id': 'M00', 'run_id': run_id,
                  'job_inputs': inputs, 'job_key': job_key(inputs), 'phase': 'preflight', 'argv': ['python', '-m', 'poc', 'run', 'M00'],
                  'code_revision': revision.stdout.strip() or 'uncommitted-no-git-revision',
                  'code_tree_sha256': hashlib.sha256(canonical_json(inputs['dependencies'])).hexdigest(),
                  'dirty': bool(dirty.stdout) or dirty.returncode != 0,
                  'configuration_sha256': hashlib.sha256(canonical_json(inputs['configuration_keys'])).hexdigest(),
                  'data_hashes': {}, 'model_hashes': {}, 'environment_sha256': sha256_file(folder / 'environment.json'),
                  'parent_decisions': [], 'started_utc': started_utc, 'ended_utc': utc_now(),
                  'completion_status': 'complete' if valid else 'blocked' if blocked else 'failed',
                  'expected_cells': list(BOOTSTRAP_CHECKS),
                  'produced_files': [{'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}
                                     for path in sorted(folder.rglob('*')) if path.is_file()]}
        if valid:
            retired = []
            for path in sorted((root / 'evidence/M00').glob('*/manifest.json')):
                previous = read_json(path)
                verify_run(root, previous, check_dependencies=False)
                retired.append(previous['run_id'])
            if retired:
                record['supersedes'] = retired
                record['reason'] = 'Capability checks repeated under current semantic dependencies; earlier attempts remain immutable history'
        complete_run(root, record)
        from .gates import record_inputs
        record_inputs(root, 'G00', [record])
        return record
