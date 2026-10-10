# License scope and third-party notices

Copyright © 2026 Artem Legotin.

Original project source code, documentation and presentation code are licensed
under GNU Affero General Public License version 3 only (`AGPL-3.0-only`), as
reproduced in [LICENSE](LICENSE). The project is provided without warranty under
that license. Citation metadata is provided in [CITATION.cff](CITATION.cff).

This grant covers rights held by the project owner. It does not assert exclusive
ownership of mathematical methods, scientific facts, previously published work,
or independently authored third-party material. It does not relicense dependencies
or their notices. The [archived design specification](docs/superpowers/specs/2026-09-30-physical-time-consistency-design.md)
derives from supplied research material and cites external references; attribution
and any rights of the original authors remain with them. No rights beyond those
the project owner holds are granted for such material.

## Dependencies

The implementation uses NumPy, SciPy, PyTorch, MuJoCo and pytest. Their transitive
dependencies have their own licenses. Installed distribution metadata and preserved
license/notice files are recorded under [locks/license_notices](locks/license_notices)
and in `locks/environment.json`; these notices must accompany any redistribution
that requires them. Some scientific wheels include additional libraries and their
notices, so the top-level package license is not the entire dependency license record.

Third-party notices are preserved verbatim in the publication snapshot. The snapshot
does not redistribute a virtual environment, package wheels or dependency binaries.
Installing a dependency does not replace its license with the project license.

## License text provenance

The unchanged AGPL-3.0-only text was obtained from the SPDX license-list repository:

<https://raw.githubusercontent.com/spdx/license-list-data/main/text/AGPL-3.0-only.txt>

Its SHA-256 is `d8a6cc31abc16b6748c7a21f21611f5a1ec33f67d22ca23d7da1c19b95496bee`.
The copyright statement in that license text refers to the Free Software Foundation's
license document; the project's copyright statement is given above.
