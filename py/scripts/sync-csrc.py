#!/usr/bin/env python3
"""Mirror the Makefile's library sources and headers into the Python sdist."""

from pathlib import Path
import shutil
import subprocess


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
CSRC = PACKAGE / 'csrc'
if (ROOT / 'Makefile').is_file() and (ROOT / 'src/core.h').is_file():
    # Ship every native backend, independently of the build host. Their C
    # feature guards still decide what the installed extension enables.
    SOURCES = subprocess.check_output(
        ['make', '-s', '--no-print-directory', 'package-c-sources',
         'HAS_CUDA=1', 'HAS_HIP=1', 'HAS_X86=1'], cwd=ROOT, text=True,
    ).split()
    HEADERS = sorted(str(p.relative_to(ROOT)) for directory in ('src', 'vendor/cjson')
                     for p in (ROOT / directory).rglob('*.h'))
else:
    # An extracted sdist has no root Makefile and must not require make to
    # discover its already mirrored sources during installation.
    SOURCES = sorted(str(p.relative_to(CSRC)) for p in CSRC.rglob('*.c'))
    HEADERS = sorted(str(p.relative_to(CSRC)) for p in CSRC.rglob('*.h'))


def main():
    files = SOURCES + HEADERS
    if not files or any(not (ROOT / f).is_file() for f in files):
        raise SystemExit('sync-csrc requires the complete repository source tree')
    if CSRC.exists():
        shutil.rmtree(CSRC)
    for relative in files:
        target = CSRC / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Fresh mtimes ensure editable builds recompile changed sources.
        shutil.copy(ROOT / relative, target)
    print(f'sync-csrc: copied {len(files)} files to {CSRC}')


if __name__ == '__main__':
    main()
