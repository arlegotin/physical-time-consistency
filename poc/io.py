"""Strict serialization, semantic hashes and atomic execution artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def atomic_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    atomic_bytes(path, content)


def append_event(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        output.write(canonical_json(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def job_key(inputs: dict) -> str:
    volatile = {"started_utc", "ended_utc", "created_utc", "recorded_utc", "run_id",
                "output_directory", "run_directory"}
    def semantic(value):
        if isinstance(value, dict):
            return {k: semantic(v) for k, v in value.items() if k not in volatile}
        if isinstance(value, list):
            return [semantic(v) for v in value]
        return value
    return hashlib.sha256(canonical_json(semantic(inputs))).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"Expected project-relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes project: {relative}")
    return path


def run_directory(root: Path, record: dict) -> Path:
    for name in (record["milestone_id"], record["run_id"]):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            raise ValueError("Unsafe run identifier")
    return safe_path(root, f"evidence/{record['milestone_id']}/{record['run_id']}")


def _process_start(pid: int) -> str:
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                capture_output=True, text=True)
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


@contextmanager
def job_lock(key: str, root: Path):
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Job key must be SHA-256")
    path = root / "locks/active" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"hostname": socket.gethostname(), "pid": os.getpid(), "job_key": key,
             "process_start_identifier": _process_start(os.getpid()),
             "heartbeat": utc_now(), "owner_token": uuid.uuid4().hex}
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            previous = read_json(path)
            if previous.get("hostname") != socket.gethostname():
                raise RuntimeError("Foreign job lock needs operator-confirmed release")
            try:
                os.kill(previous["pid"], 0)
            except ProcessLookupError:
                if attempt == 0:
                    append_event(root / "evidence/lock_events.jsonl",
                                 {"event": "reclaim_dead_process", "lock": previous, "utc": utc_now()})
                    path.unlink()
                    continue
            except (PermissionError, KeyError, TypeError):
                pass
            raise RuntimeError("Job lock is live or process identity cannot be verified")
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_json(owner) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    try:
        yield owner
    finally:
        if path.exists() and read_json(path).get("owner_token") == owner["owner_token"]:
            path.unlink()


def complete_run(root: Path, record: dict) -> None:
    from .contracts import validate_record
    root = root.resolve()
    folder = run_directory(root, record)
    if (folder / "COMPLETED").exists():
        raise FileExistsError(f"Completed evidence is immutable: {folder}")
    validate_record("manifest", record, root)
    for entry in record["produced_files"]:
        if sha256_file(safe_path(root, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {entry['path']}")
    atomic_json(folder / "manifest.json", record)
    atomic_json(folder / "COMPLETED", {"manifest_sha256": sha256_file(folder / "manifest.json")})
    append_event(root / "evidence/index.jsonl", {
        "run_id": record["run_id"], "milestone_id": record["milestone_id"],
        "job_key": record["job_key"], "manifest": str((folder / "manifest.json").relative_to(root)),
        "sha256": sha256_file(folder / "manifest.json"), "utc": utc_now(),
    })
