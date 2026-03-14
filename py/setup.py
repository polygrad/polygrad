import os
import re
import runpy
import subprocess
import sys
from setuptools import setup, Extension

if sys.platform == 'win32':
    sys.exit(
        'polygrad requires Linux. Windows is not supported.\n'
        'runtime_cpu.c uses fork() and dlopen() which are not available on Windows.'
    )

# Auto-sync csrc/ in repo checkouts so editable installs pick up core changes.
here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')
repo_root = os.path.dirname(here)


def _read_version():
    pyproject = os.path.join(here, 'pyproject.toml')
    with open(pyproject, encoding='utf-8') as f:
        text = f.read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError('Could not read version from pyproject.toml')
    return m.group(1)

def _needs_sync(sync_script):
    if not os.path.isfile(sync_script):
        return False

    manifest = runpy.run_path(sync_script)
    tracked = manifest.get('SOURCES', []) + manifest.get('HEADERS', [])
    if not tracked:
        return False

    repo_files = [os.path.join(repo_root, rel) for rel in tracked]
    if not all(os.path.isfile(path) for path in repo_files):
        # sdist installs ship csrc/ already; they do not have the repo source tree.
        return not os.path.isdir(csrc)

    if not os.path.isdir(csrc):
        return True

    for rel in tracked:
        src = os.path.join(repo_root, rel)
        dst = os.path.join(csrc, rel)
        if not os.path.isfile(dst):
            return True
        if os.path.getmtime(src) > os.path.getmtime(dst):
            return True
    return False


sync_script = os.path.join(here, 'scripts', 'sync-csrc.py')
if _needs_sync(sync_script):
    print('setup.py: syncing csrc/ from repo sources...')
    subprocess.check_call([sys.executable, sync_script])
elif not os.path.isdir(csrc):
    sys.exit(
        'csrc/ directory not found and scripts/sync-csrc.py could not populate it.\n'
        'Run from the repo: python py/scripts/sync-csrc.py'
    )

# Collect all .c files from csrc/ using relative paths (setuptools requires this)
sources = []
for root, dirs, files in os.walk('csrc'):
    for f in sorted(files):
        if f.endswith('.c'):
            sources.append(os.path.join(root, f))
sources.append(os.path.join('polygrad', '_native.c'))

# Platform-specific libraries
libraries = ['m']
if sys.platform.startswith('linux'):
    libraries.append('dl')

setup(
    name='polygrad',
    version=_read_version(),
    ext_modules=[
        Extension(
            'polygrad._native',
            sources=sources,
            include_dirs=[
                os.path.join('csrc', 'src'),
                os.path.join('csrc', 'vendor', 'cjson'),
            ],
            extra_compile_args=['-std=c11', '-O2', '-D_GNU_SOURCE', '-DPOLY_HAS_CUDA=1'],
            libraries=libraries,
        )
    ]
)
