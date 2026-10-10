"""Publication must preserve scientific bytes and refuse unsafe disclosure."""
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def exporter():
    path = ROOT / 'scripts/export_publication.py'
    assert path.exists(), 'Publication exporter has not been implemented'
    spec = importlib.util.spec_from_file_location('publication_exporter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private_home():
    return '/' + '/'.join(('Users', 'fixture-person'))


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


@pytest.fixture
def committed_repo(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    git(root, 'init', '-q')
    files = {
        'LICENSE': (ROOT / 'LICENSE').read_bytes(),
        'CITATION.cff': (ROOT / 'CITATION.cff').read_bytes(),
        'README.md': b'# Published result\n',
        '.gitignore': b'ignored.txt\npublication/\n',
        'poc/model.py': b'def model(): return 42\n',
        'poc/compute_study.py': (ROOT / 'poc/compute_study.py').read_bytes(),
        'reports/metric.json': json.dumps({'cost': .28296467869929587,
            'artifact_root': private_home() + '/artifacts'}).encode(),
        'evidence/stdout.log': b'Private executor log\n',
        'evidence/index.jsonl': b'Private ledger\n',
        'config/authorization.json': b'Private owner allocation\n',
        'reports/provenance/execution.zip': b'Not a distributable execution archive\n',
        'locks/license_notices/library/LICENSE': b'Original dependency notice\n',
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Publication Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'Fixture')
    return root


def test_metadata_redaction_preserves_metrics_and_original_bytes():
    module = exporter()
    original = json.dumps({'cost': .28296467869929587,
        'artifact_root': private_home() + '/artifacts', 'hostname': 'private-machine'}).encode()
    published, changes = module.public_copy('reports/result.json', original)
    assert json.loads(published)['cost'] == json.loads(original)['cost']
    assert private_home().encode() not in published
    assert b'private-machine' not in published
    assert 'local_home_path' in changes and 'machine_name' in changes
    assert json.loads(original)['hostname'] == 'private-machine'
    assert module.public_copy('reports/result.json', published) == (published, [])


def test_source_and_dependency_notices_are_exact_or_refused():
    module = exporter()
    data = b'def prediction(): return 1\n'
    assert module.public_copy('poc/model.py', data) == (data, [])
    notice = b'Copyright a library author. Original notices.\n'
    assert module.public_copy('locks/license_notices/library/LICENSE', notice) == (notice, [])
    for path in ('poc/model.py', 'locks/license_notices/library/LICENSE'):
        with pytest.raises(ValueError, match='protected file'):
            module.public_copy(path, (private_home() + '/private').encode())


def test_archived_contexts_do_not_reintroduce_owner_authorization_or_runtime_pointers():
    module = exporter()
    for name in ('config/authorization.json', 'decisions/next_action.json',
                 'reports/reference_serial_handoff.json', 'locks/active/job.json'):
        assert module.omission('study/old/context/' + name) is not None


@pytest.mark.parametrize('token, label', [
    ('ghp_' + 'A' * 36, 'github_token'),
    ('sk-proj-' + 'B' * 50, 'openai_token'),
    ('AKIA' + 'A' * 16, 'aws_access_key'),
    ('-----BEGIN ' + 'PRIVATE KEY-----', 'private_key'),
])
def test_credentials_fail_without_echoing_the_value(token, label):
    with pytest.raises(ValueError) as error:
        exporter().public_copy('reports/result.json', token.encode())
    assert label in str(error.value)
    assert token not in str(error.value)


@pytest.mark.parametrize('value', ['actual-private-value-123456', 'realprivateNone123456'])
@pytest.mark.parametrize('quoted', [False, True])
def test_literal_credentials_cannot_hide_in_env_assignments_or_placeholder_substrings(value, quoted):
    key = 'PASS' + 'WORD'
    data = json.dumps({key: value}) if quoted else key + '=' + value
    with pytest.raises(ValueError, match='literal_secret_assignment') as error:
        exporter().public_copy('config/example.txt', data.encode())
    assert value not in str(error.value)


def test_utf8_binary_payload_is_refused():
    with pytest.raises(ValueError, match='binary'):
        exporter().public_copy('assets/payload.txt', b'MZ' + bytes([0]) * 32)


def test_operational_records_and_private_journal_are_omitted():
    module = exporter()
    for name in ('reports/compute_study_episode_execution.json',
                 'reports/control_precision_execution_launch_2026-10-06.json',
                 'reports/compute_study_handoff.json', 'docs/execution-history.md',
                 'reports/provenance/relocate_venv.py',
                 'reports/affine_volume_continuation_launch.json',
                 'reports/provenance/compute_study_daemon_recovery_004.json',
                 'reports/affine_precision_001_operational_progress.json',
                 'reports/project_completion_status.json',
                 'reports/control_workstation_sleep_review_2026-10-03.json',
                 'reports/daemon_recovery_2026-10-05.json',
                 'reports/compute_study_timing_isolation.json',
                 'reports/control_finer_full_review_2026-10-06.json',
                 'reports/reproduction_watcher_recovery_2026-10-05.json',
                 'reports/user_pause_2026-10-05.json',
                 'reports/user_resume_2026-10-05.json',
                 'reports/development_solver_serial_audit.json',
                 'reports/nominal_selection_serial_audit.json',
                 'reports/control_storage_clone_attempts/attempt/receipt.json',
                 'locks/reproduction_cache_cleanup.json'):
        assert module.omission(name) is not None
    assert module.omission('reports/compute_study_result.json') is None
    # These contain measured completion/coverage, despite an operational flag.
    assert module.omission('reports/control_precision_primary_completion_2026-10-07.json') is None


def test_superseded_protocol_is_excluded_from_publication_even_in_old_contexts():
    module = exporter()
    for path in ('POC_EXECUTION_PLAN.md', 'study/old/context/POC_EXECUTION_PLAN.md'):
        assert module.omission(path) is not None


def test_portable_artifact_default_is_the_only_derived_source_exception():
    module = exporter()
    original = (ROOT / 'poc/compute_study.py').read_bytes()
    public, changes = module.public_copy('poc/compute_study.py', original)
    if module.digest(original) == module.PORTABLE_EXPORTED_SHA256:
        # A public clone already contains the disclosed adaptation.
        assert public == original and changes == []
    else:
        assert module.digest(original) == module.PORTABLE_ORIGINAL_SHA256
        assert changes == ['portable_default_artifact_path']
        old = next(line for line in original.splitlines(keepends=True) if line.startswith(b'STORE=Path('))
        assert public == original.replace(old, module.PORTABLE_STORE_LINE)
    assert module.public_copy('poc/compute_study.py', public) == (public, [])
    with pytest.raises(ValueError, match='source adaptation'):
        module.public_copy('poc/compute_study.py', original + b'\n# ' + private_home().encode() + b'\n')


def test_export_reads_committed_bytes_omits_private_files_and_is_deterministic(committed_repo, tmp_path):
    module = exporter()
    root = committed_repo
    (root / 'README.md').write_text('Uncommitted text must never leak')
    (root / 'ignored.txt').write_text('ghp_' + 'A' * 36)
    archive = tmp_path / 'public.zip'
    record = module.export_repository(root, archive)
    verified = module.verify_archive(archive)
    assert verified == record
    assert record['contains_git_history'] is False
    adapted = next(row for row in record['files'] if row['path'] == 'poc/compute_study.py')
    source_hash = module.digest((ROOT / 'poc/compute_study.py').read_bytes())
    assert adapted['source_sha256'] == source_hash
    assert adapted['transformations'] == ([] if source_hash == module.PORTABLE_EXPORTED_SHA256
                                         else ['portable_default_artifact_path'])
    assert adapted['exported_sha256'] == module.PORTABLE_EXPORTED_SHA256
    names = {row['path'] for row in record['files']}
    assert 'poc/model.py' in names and 'ignored.txt' not in names
    assert 'evidence/stdout.log' not in names
    assert 'config/authorization.json' not in names
    assert 'reports/provenance/execution.zip' not in names
    omitted = {row['path'] for row in record['omitted']}
    assert {'evidence/stdout.log', 'evidence/index.jsonl', 'config/authorization.json'} <= omitted
    with zipfile.ZipFile(archive) as z:
        assert z.read('physical-time-consistency/README.md') == b'# Published result\n'
        assert z.read('physical-time-consistency/poc/model.py') == (root / 'poc/model.py').read_bytes()
        data = json.loads(z.read('physical-time-consistency/reports/metric.json'))
        assert data['cost'] == .28296467869929587 and private_home() not in data['artifact_root']
    second = tmp_path / 'second.zip'
    module.export_repository(root, second)
    assert second.read_bytes() == archive.read_bytes()
    with pytest.raises(FileExistsError):
        module.export_repository(root, archive)
    assert (root / 'README.md').read_text() == 'Uncommitted text must never leak'


def test_manifest_staging_preserves_ignored_study_files_without_extra_private_files(committed_repo, tmp_path):
    module = exporter()
    (committed_repo / '.gitignore').write_text('ignored.txt\nstudy/\npublication/\n')
    tracked = committed_repo / 'study/summary.json'
    tracked.parent.mkdir()
    tracked.write_text('{"cost": 1}')
    git(committed_repo, 'add', '.gitignore')
    git(committed_repo, 'add', '-f', 'study/summary.json')
    git(committed_repo, '-c', 'user.name=Publication Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'Tracked study fixture')
    output = tmp_path / 'stage.zip'
    manifest = module.export_repository(committed_repo, output)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(tmp_path / 'extracted')
    public = tmp_path / 'extracted/physical-time-consistency'
    (public / 'ignored.txt').write_text('Must not be staged')
    git(public, 'init', '-q')
    paths = [row['path'] for row in manifest['files']] + [module.MANIFEST]
    subprocess.run(['git', '-C', str(public), '--literal-pathspecs', 'add', '-f',
                    '--pathspec-from-file=-', '--pathspec-file-nul'],
                   input=b''.join(path.encode() + b'\0' for path in paths), check=True)
    staged = set(git(public, 'ls-files', '-z').decode().rstrip('\0').split('\0'))
    assert staged == set(paths)
    assert 'study/summary.json' in staged and 'ignored.txt' not in staged


def test_secret_in_commit_never_leaves_a_completed_or_partial_archive(committed_repo, tmp_path):
    module = exporter()
    root = committed_repo
    (root / 'reports/credential.txt').write_text('ghp_' + 'A' * 36)
    git(root, 'add', 'reports/credential.txt')
    git(root, '-c', 'user.name=Publication Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'Bad fixture')
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError, match='github_token'):
        module.export_repository(root, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.publication-*'))


def test_symlink_cannot_read_an_untracked_target(committed_repo, tmp_path):
    module = exporter()
    outside = tmp_path / 'outside'
    outside.write_text('Private content')
    (committed_repo / 'poc/link.py').symlink_to(outside)
    git(committed_repo, 'add', 'poc/link.py')
    git(committed_repo, '-c', 'user.name=Publication Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'Symlink fixture')
    with pytest.raises(ValueError, match='symlink'):
        module.export_repository(committed_repo, tmp_path / 'bad.zip')


@pytest.mark.parametrize('mutation', ['changed_file', 'extra_file', 'traversal', 'symlink', 'special_file'])
def test_verification_rejects_tampering_and_unlisted_or_unsafe_members(committed_repo, tmp_path, mutation):
    module = exporter()
    source = tmp_path / 'public.zip'
    module.export_repository(committed_repo, source)
    changed = tmp_path / 'changed.zip'
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(changed, 'w') as output:
        for info in original.infolist():
            data = original.read(info)
            if mutation == 'changed_file' and info.filename.endswith('/README.md'):
                data = b'Altered result'
            if mutation in {'symlink', 'special_file'} and info.filename.endswith('/README.md'):
                info.external_attr = (0o120777 if mutation == 'symlink' else 0o020666) << 16
            output.writestr(info, data)
        if mutation in {'extra_file', 'traversal'}:
            output.writestr('../private' if mutation == 'traversal' else 'physical-time-consistency/extra.txt', b'Extra')
    with pytest.raises(ValueError):
        module.verify_archive(changed)
