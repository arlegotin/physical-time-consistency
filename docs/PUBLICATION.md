# Publication preparation

The owner approved AGPL-3.0-only and citation as Artem Legotin. The [README](../README.md)
presents the scoped result, [CITATION.cff](../CITATION.cff) supplies citation
metadata, and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) defines license scope.

## Publish the clean snapshot

Development history contains local paths, machine metadata and executor records.
A new redaction commit does not remove them from old commits. Preserve the
complete original
working tree, Git history and external experiment store in a verified private
archive. Publish the screened snapshot with clean history. This can use the
**same GitHub repository name and URL**; creating another repository is optional.
Before changing visibility, inspect existing remote branches, pull requests,
Actions logs and release assets. An empty remote needs no history rewrite or
force push. A populated remote requires a separate history-cleanup review.

From the original development archive, after committing publication changes,
build and verify from committed bytes:

```bash
python3 scripts/export_publication.py --output publication/physical-time-consistency-public.zip
python3 scripts/export_publication.py --verify publication/physical-time-consistency-public.zip
```

The exporter has no push, upload or visibility-changing operation. It includes
tracked source, tests, specifications, dependency notices and results. It omits
Git history, executor logs/ledgers, execution ZIPs, browser screenshots, the
private journal, workstation maintenance scripts, local authorization and
runtime launch/handoff records. Ignored datasets, tensors, wheels,
environments and browser profiles are never read. Suspicious credential
patterns and unsupported binary files prevent completion.

Local paths are redacted only in exported copies. Model/controller source and
third-party notices remain byte-identical. One explicit source adaptation in
`poc/compute_study.py` replaces only the literal default artifact directory with
`Path.home()` and the same relative path. Both full-file hashes are guarded;
any other source redaction is refused. `PUBLICATION_MANIFEST.json` records the source
commit, included/exported hashes, privacy transformations and every omission.
Derived metadata does not replace the original frozen scientific evidence.
The public package clearly states that full reaggregation needs external inputs.

Extract the verified ZIP into an empty staging folder and initialize **clean
Git history** there. Keep the original private backup separate from this
checkout; do not create public backup branches or tags pointing to old commits.
The public package contains derived historical metadata; links from old
reports to deliberately omitted executor records may be unavailable. The README
and current documentation provide the supported public navigation. The private
journal preserves the development chronology in the private archive; it is
omitted from the public package. The superseded supplied execution plan is also
omitted. Its original bytes and provenance remain in earlier private Git revisions
and scientific snapshots, rather than being rewritten in historical evidence.

From the extracted `physical-time-consistency/` folder, the following stages
only the manifest's verified files, including tracked study metadata that the
development `.gitignore` excludes from ordinary discovery:

```bash
python3 - <<'PY'
import json
import subprocess
from pathlib import Path

manifest = json.loads(Path('PUBLICATION_MANIFEST.json').read_text())
paths = [row['path'] for row in manifest['files']] + ['PUBLICATION_MANIFEST.json']
subprocess.run(['git', 'init'], check=True)
subprocess.run(['git', '--literal-pathspecs', 'add', '-f',
                '--pathspec-from-file=-', '--pathspec-file-nul'],
               input=b''.join(path.encode() + b'\0' for path in paths), check=True)
PY
```

Review the staged tree and make an initial commit with an explicit Git identity.
Push the reviewed clean history to the intended repository only after verifying
the private backup and remote state. Do not copy local credentials or the old
`.git` directory into the public checkout. If the final public
URL changes, update both citation URL fields and regenerate the package. The
current citation URL is the intended location, not evidence of publication.

GitHub-hosted Actions logs, release assets and other remote attachments are
outside the local scan. Review them separately if reusing a remote. Copies
made after publication cannot be recalled by making a repository private.

## Final checks

- The README explains the problem, method, measured effect and scope on its first screen.
- Negative/unresolved results, accuracy/speed trade-offs and external-artifact limits remain visible.
- Author, license identifier, citation fields and unchanged license text are verified.
- Package hashes, omissions, privacy checks and first-level documentation links are verified.
- Focused model/controller tests run from an extracted package; they are software checks, not new experiments.
- Uploading and changing visibility require the owner's authorization. The owner
  approved publication of this cleaned repository on 2026-10-10.

Publication does not imply peer review, independent laboratory replication,
novelty verification or physical-robot results.

## Local storage

The public checkout needs source, tests, reports, demos and publication metadata.
Large raw datasets, candidate banks, selected tensors, execution environments
and original frozen workspaces belong in the separate private archive. They
remain necessary for full scientific verification and reruns, but do not need
to occupy the public working folder. Keep original byte hashes and dependency
paths recorded; reducing raw evidence to summary numbers is not equivalent to
preserving it. Restore instructions and local archive locations stay private.
