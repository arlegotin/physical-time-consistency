"""Create a privacy-screened source snapshot without changing the research archive.

Uses committed Git blobs only. No uploads, history rewriting, scientific
reaggregation, or reads from ignored datasets/environments are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile


PREFIX = 'physical-time-consistency/'
MANIFEST = 'PUBLICATION_MANIFEST.json'
LICENSE_SHA256 = 'd8a6cc31abc16b6748c7a21f21611f5a1ec33f67d22ca23d7da1c19b95496bee'
REQUIRED = {'README.md', 'LICENSE', 'CITATION.cff'}
SECRET_PATTERNS = {
    'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'openai_token': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'slack_token': rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
    'google_api_key': rb'\bAIza[A-Za-z0-9_-]{35}\b',
    'credential_in_url': rb'https?://[^\s/<>"\x27:]{1,100}:[^\s/<>"\x27@]{1,200}@[A-Za-z0-9.-]+',
}
SECRET_REGEXES = {name: re.compile(pattern) for name, pattern in SECRET_PATTERNS.items()}
ASSIGNMENT = re.compile(rb'(?i)(?<![A-Za-z0-9_])["\x27]?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|client[_-]?secret)["\x27]?\s*[:=]\s*(?:["\x27]([^"\x27\r\n]*)["\x27]|([A-Za-z0-9_+/=.-]+))')
# Exact example values only: a real credential may contain any of these words.
PLACEHOLDERS = {b'', b'example', b'placeholder', b'your_api_key', b'your_token',
                b'dummy', b'none', b'null', b'fake', b'not-a-secret'}
PRIVACY = (
    ('local_home_path', re.compile(rb'/(?:Users|home)/[A-Za-z0-9][A-Za-z0-9_.-]*'), b'/redacted-home'),
    ('local_checkout_path', re.compile(rb'/Volumes/git/[A-Za-z0-9][A-Za-z0-9_.-]*/'), b'/workspace/'),
    ('local_temporary_path', re.compile(rb'/(?:private/)?var/folders/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?'), b'/tmp/redacted'),
)
MACHINE_NAME = re.compile(rb'(?i)("(?:hostname|host_name|computer_name)"\s*:\s*)"(?:[^"\\]|\\.)*"')
PRIVATE_FILES = {
    'config/authorization.json', 'decisions/next_action.json', 'decisions/gate_inputs.json',
    'evidence/index.jsonl', 'evidence/data_access.jsonl', 'evidence/lock_events.jsonl',
    'reports/reference_serial_handoff.json', 'reports/control_live_monitor.json',
    'docs/execution-history.md',
    'reports/efficiency_pilot_release_evidence/check_local_artifacts.py',
    'reports/provenance/finish_venv_relocation.py',
    'reports/provenance/relocate_fresh_workspace.py',
    'reports/provenance/relocate_venv.py',
    'reports/project_completion_status.json',
    'reports/control_finer_full_review_2026-10-06.json',
    'reports/development_solver_serial_audit.json',
    'reports/nominal_selection_serial_audit.json',
    'locks/reproduction_cache_cleanup.json',
}
# A declared source adaptation changes only the workstation-specific default
# directory. Full-file hashes guard both sides; frozen science bytes stay local.
PORTABLE_SOURCE = 'poc/compute_study.py'
PORTABLE_ORIGINAL_SHA256 = '9b41f0b0b83952ac828d1eb8355897926f33d1b1e2c74aae70528c3199e329d7'
PORTABLE_EXPORTED_SHA256 = 'd9320da97a36dc2b1abe095b0159bf2bc941138231c325377e8c0140ea486d3e'
PORTABLE_STORE_LINE = b"STORE=Path.home() / 'physical-time-consistency-artifacts/study/compute-matched-node'\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_no_secrets(data: bytes, path: str) -> None:
    labels = [name for name, pattern in SECRET_REGEXES.items() if pattern.search(data)]
    if any((match.group(1) if match.group(1) is not None else match.group(2)).strip().lower()
           not in PLACEHOLDERS for match in ASSIGNMENT.finditer(data)):
        labels.append('literal_secret_assignment')
    if labels:
        # Values are deliberately never included in diagnostics.
        raise ValueError(f'Credential pattern {", ".join(labels)} in {path}; publication refused')


def safe_name(path: str) -> None:
    parts = PurePosixPath(path).parts
    if not path or path.startswith('/') or '\\' in path or any(p in {'.', '..', '.git'} for p in parts):
        raise ValueError('Unsafe publication member path')
    if PurePosixPath(path).as_posix() != path:
        raise ValueError('Noncanonical publication member path')
    assert_no_secrets(path.encode(), '(member filename)')


def omission(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if parts[-1] == 'POC_EXECUTION_PLAN.md':
        return 'superseded supplied planning document; retained in private history'
    if any(p in {'.superpowers', '.venv', '__pycache__', '.pytest_cache', 'wheelhouse',
                 'publication', '.reproduction-venv'} or p.startswith('.partial-') for p in parts):
        return 'local runtime or packaging directory'
    if (path in PRIVATE_FILES or any(path.endswith('/' + name) for name in PRIVATE_FILES)
            or path.startswith('locks/active/') or '/locks/active/' in path):
        return 'executor authorization, ledger or mutable pointer'
    if 'reports' in parts and re.search(
            r'(?:^|_)(?:execution|handoff|launch|daemon|watcher|operational_progress|workstation_sleep|user_pause|user_resume|timing_isolation)(?:_|\.)',
            parts[-1]):
        return 'executor launch, process or handoff record'
    if 'config' in parts and re.search(r'(?:^|_)allocation(?:_|\.)', parts[-1]):
        return 'local executor resource allocation'
    if path.endswith(('.log', '.jsonl', '.diff')):
        return 'executor log, event stream or raw review diff'
    if path.endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.7z', '.pyc', '.pt', '.npz', '.npy', '.whl')):
        return 'private execution archive, binary dependency or raw experiment artifact'
    if path.endswith('.png'):
        return 'browser screenshot; presentation is available in HTML/SVG'
    if 'reports' in parts and any('storage_' in part for part in parts):
        return 'workstation storage maintenance record'
    return None


def protected(path: str) -> bool:
    return path.endswith('.py') or path == 'LICENSE' or 'license_notices' in PurePosixPath(path).parts


def public_copy(path: str, data: bytes) -> tuple[bytes, list[str]]:
    """Screen a file and redact only metadata, preserving code/notices exactly."""
    safe_name(path)
    assert_no_secrets(data, path)
    try:
        data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError(f'Unsupported binary file: {path}') from error
    if re.search(rb'[\x00-\x08\x0b\x0e-\x1f\x7f]', data):
        raise ValueError(f'Unsupported binary/control payload: {path}')
    result, changes = data, []
    if path == PORTABLE_SOURCE and PRIVACY[0][1].search(data):
        if digest(data) != PORTABLE_ORIGINAL_SHA256:
            raise ValueError('Unexpected source adaptation input; original hash differs')
        old = next(line for line in data.splitlines(keepends=True) if line.startswith(b'STORE=Path('))
        result = data.replace(old, PORTABLE_STORE_LINE, 1)
        if digest(result) != PORTABLE_EXPORTED_SHA256:
            raise ValueError('Unexpected source adaptation output')
        changes.append('portable_default_artifact_path')
    for label, pattern, replacement in PRIVACY:
        result, count = pattern.subn(replacement, result)
        if count:
            changes.append(label)
    if path.endswith('.json'):
        updated = MACHINE_NAME.sub(rb'\1"redacted-host"', result)
        if updated != result:
            changes.append('machine_name')
        result = updated
    if changes and protected(path) and not (
            path == PORTABLE_SOURCE and changes == ['portable_default_artifact_path']):
        raise ValueError(f'Privacy redaction required in protected file: {path}')
    return result, changes


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def write_member(archive: zipfile.ZipFile, path: str, data: bytes, executable: bool = False) -> None:
    info = zipfile.ZipInfo(PREFIX + path, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (0o100755 if executable else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data, compresslevel=6)


def export_repository(root: Path, output: Path, revision: str = 'HEAD') -> dict:
    """Atomically export a committed snapshot; never overwrite an existing file."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; verify it or choose a new filename')
    if output.is_relative_to(root) and not output.is_relative_to(root / 'publication'):
        raise ValueError('In-repository output must be under the ignored publication/ directory')
    if revision.startswith('-'):
        raise ValueError('Invalid revision')
    commit = git(root, 'rev-parse', '--verify', revision + '^{commit}').decode().strip()
    files = []
    for row in git(root, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0'):
        if not row:
            continue
        meta, name = row.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        path = name.decode('utf-8')
        safe_name(path)
        if mode == '120000':
            raise ValueError(f'Committed symlink is not publishable: {path}')
        if kind != 'blob' or mode not in {'100644', '100755'}:
            raise ValueError(f'Unsupported tracked object: {path}')
        if path == MANIFEST:
            raise ValueError('Reserved publication manifest is already tracked')
        files.append((path, mode, oid))
    if not REQUIRED <= {p for p, _, _ in files}:
        raise ValueError('Commit does not contain README.md, LICENSE and CITATION.cff')
    record = dict(schema_version=1, source_commit=commit,
                  contains_git_history=False, raw_experiment_store_included=False,
                  scope='Derived public source snapshot; original frozen evidence remains external and unmodified',
                  files=[], omitted=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.publication-', suffix='.zip', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    reader = subprocess.Popen(['git', '-C', str(root), 'cat-file', '--batch'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with zipfile.ZipFile(temporary, 'w') as archive:
            for path, mode, oid in sorted(files):
                reader.stdin.write((oid + '\n').encode())
                reader.stdin.flush()
                header = reader.stdout.readline().decode().split()
                if len(header) != 3 or header[0] != oid or header[1] != 'blob':
                    raise ValueError('Incomplete committed blob header')
                data = reader.stdout.read(int(header[2]))
                if len(data) != int(header[2]) or reader.stdout.read(1) != b'\n':
                    raise ValueError('Incomplete committed blob read')
                source_hash = digest(data)
                reason = omission(path)
                if reason:
                    record['omitted'].append(dict(path=path, source_sha256=source_hash, reason=reason))
                    continue
                published, transformations = public_copy(path, data)
                if path == 'LICENSE' and digest(published) != LICENSE_SHA256:
                    raise ValueError('AGPL license text does not match the verified SPDX text')
                write_member(archive, path, published, mode == '100755')
                record['files'].append(dict(path=path, source_sha256=source_hash,
                                           exported_sha256=digest(published),
                                           bytes=len(published), transformations=transformations))
            write_member(archive, MANIFEST, (json.dumps(record, indent=2, sort_keys=True,
                                                      allow_nan=False) + '\n').encode())
        verify_archive(temporary)
        # Exclusive linking publishes the completed file without a race-prone overwrite.
        os.link(temporary, output)
        return record
    finally:
        reader.stdin.close()
        reader.stdout.close()
        reader.stderr.close()
        reader.wait()
        temporary.unlink(missing_ok=True)


def verify_archive(path: Path) -> dict:
    """Check exact membership, hashes and privacy; never extract arbitrary paths."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate publication members')
        for info in archive.infolist():
            name = info.filename
            if (info.create_system != 3 or info.external_attr >> 16 not in {0o100644, 0o100755}
                    or info.flag_bits & 1):
                raise ValueError('Unsafe publication member type, permissions or encryption')
            if not name.startswith(PREFIX):
                raise ValueError('Unexpected publication archive root')
            safe_name(name[len(PREFIX):])
        record = json.loads(archive.read(PREFIX + MANIFEST),
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite manifest')))
        if (record.get('schema_version') != 1 or record.get('contains_git_history') is not False
                or record.get('raw_experiment_store_included') is not False):
            raise ValueError('Wrong publication manifest scope/version')
        expected = {PREFIX + MANIFEST}
        for row in record['files']:
            name = PREFIX + row['path']
            safe_name(row['path'])
            if name in expected or omission(row['path']) is not None:
                raise ValueError('Duplicate or prohibited manifest entry')
            expected.add(name)
            data = archive.read(name)
            if len(data) != row['bytes'] or digest(data) != row['exported_sha256']:
                raise ValueError(f'Publication hash/size mismatch: {row["path"]}')
            if not re.fullmatch('[0-9a-f]{64}', row['source_sha256']):
                raise ValueError('Invalid original SHA-256')
            screened, changes = public_copy(row['path'], data)
            if changes or screened != data:
                raise ValueError(f'Unredacted local metadata: {row["path"]}')
            portable = (row['path'] == PORTABLE_SOURCE
                        and row['transformations'] == ['portable_default_artifact_path']
                        and row['source_sha256'] == PORTABLE_ORIGINAL_SHA256
                        and row['exported_sha256'] == PORTABLE_EXPORTED_SHA256)
            if protected(row['path']) and not portable and (
                    row['transformations'] or row['source_sha256'] != row['exported_sha256']):
                raise ValueError('Protected code or notice changed in public snapshot')
        if set(names) != expected or not REQUIRED <= {r['path'] for r in record['files']}:
            raise ValueError('Missing or unlisted publication files')
        if digest(archive.read(PREFIX + 'LICENSE')) != LICENSE_SHA256:
            raise ValueError('Wrong AGPL text')
        assert_no_secrets(archive.read(PREFIX + MANIFEST), MANIFEST)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--output', type=Path, help='Create a ZIP from committed HEAD bytes')
    choices.add_argument('--verify', type=Path, help='Verify an existing ZIP; no extraction or experiments')
    parser.add_argument('--revision', default='HEAD')
    args = parser.parse_args()
    try:
        record = (verify_archive(args.verify) if args.verify else
                  export_repository(Path(__file__).resolve().parents[1], args.output, args.revision))
    except (ValueError, FileExistsError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Publication refused: {error}\n')
    print(json.dumps(dict(source_commit=record['source_commit'], included_files=len(record['files']),
                          omitted_files=len(record['omitted']),
                          redacted_files=sum(bool(r['transformations']) for r in record['files']),
                          archive_sha256=digest((args.verify or args.output).read_bytes()))))


if __name__ == '__main__':
    main()
