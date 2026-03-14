import os
import subprocess
import sys
from setuptools import setup, Extension

if sys.platform == 'win32':
    sys.exit(
        'polygrad requires Linux. Windows is not supported.\n'
        'runtime_cpu.c uses fork() and dlopen() which are not available on Windows.'
    )

# Auto-sync csrc/ if missing (self-healing for editable installs)
here = os.path.dirname(os.path.abspath(__file__))
csrc = os.path.join(here, 'csrc')

if not os.path.isdir(csrc):
    sync_script = os.path.join(here, 'scripts', 'sync-csrc.py')
    if os.path.isfile(sync_script):
        print('setup.py: csrc/ not found, running sync-csrc.py...')
        subprocess.check_call([sys.executable, sync_script])
    else:
        sys.exit(
            'csrc/ directory not found and scripts/sync-csrc.py is missing.\n'
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
    ext_modules=[
        Extension(
            'polygrad._native',
            sources=sources,
            include_dirs=[
                os.path.join('csrc', 'src'),
                os.path.join('csrc', 'vendor', 'cjson'),
            ],
            extra_compile_args=['-std=c11', '-O2', '-D_GNU_SOURCE'],
            libraries=libraries,
        )
    ]
)
