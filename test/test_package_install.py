#!/usr/bin/env python3
"""Strict isolated installs of locally built release artifacts; retain raw logs."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def clean_environment():
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME', 'POLYGRAD_LIB', 'POLY_CORE',
                'POLYGRAD_SKIP_NATIVE', 'NODE_PATH', 'NODE_OPTIONS',
                'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER',
                'npm_config_ignore_scripts', 'NPM_CONFIG_IGNORE_SCRIPTS'):
        env.pop(key, None)
    env['POLY_DEVICE'] = 'cpu'
    env['DEV'] = 'CPU'
    return env


def run(command, cwd, env, log):
    print('RUN', ' '.join(map(str, command)), '->', log, flush=True)
    with log.open('w') as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}: {command}; see {log}')


def only_artifact(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one {pattern} in {directory}, found {matches}')
    return matches[0].resolve()


def python_install(work, env):
    artifact = only_artifact(ROOT / 'py/dist', 'polygrad-*.tar.gz')
    venv = work / 'venv'
    run([sys.executable, '-m', 'venv', str(venv)], work, env, work / 'venv.log')
    interpreter = venv / 'bin/python'
    run([str(interpreter), '-I', '-m', 'pip', 'install', '--no-cache-dir', str(artifact)],
        work, env, work / 'install.log')
    run([str(interpreter), '-I', str(ROOT / 'test/package_install_python.py')],
        work, env, work / 'runtime.log')


def node_install(work, env, npm, node):
    run([npm, 'pack', '--ignore-scripts=false', '--pack-destination', str(work)],
        ROOT / 'js', env, work / 'pack.log')
    artifact = only_artifact(work, 'polygrad-*.tgz')
    for lane, expected in (('native', 'native'), ('fallback', 'wasm')):
        prefix = work / lane
        prefix.mkdir()
        install_env = dict(env, CC='false', CXX='false') if lane == 'fallback' else env
        log = work / f'{lane}-install.log'
        run([npm, 'install', '--prefix', str(prefix), '--foreground-scripts',
             '--ignore-scripts=false', '--no-audit', '--no-fund', str(artifact)],
            prefix, install_env, log)
        marker = ('native addon built successfully' if lane == 'native'
                  else 'native addon build failed (will use WASM fallback)')
        install_output = log.read_text()
        if marker not in install_output:
            raise RuntimeError(f'{lane}: install lifecycle did not report {marker!r}; see {log}')
        if lane == 'fallback' and not re.search(r'\bfalse\b', install_output):
            raise RuntimeError(f'forced compiler failure was not observed; see {log}')
        addon = prefix / 'node_modules/polygrad/build/Release/polygrad_napi.node'
        if addon.exists() != (lane == 'native'):
            raise RuntimeError(f'{lane}: unexpected native addon presence at {addon}')
        run([node, str(ROOT / 'test/package_install_node.cjs'), expected],
            prefix, env, work / f'{lane}-runtime.log')
        # Resolve from the installed consumer directory. Browser conditions in
        # Node verify exports only; the real WebGPU gate remains test-browser.
        for subpath, condition in (('', False), ('', True), ('/async', False),
                                   ('/async', True), ('/browser', True), ('/browser/async', True)):
            code = f'''
import assert from 'node:assert/strict';
import * as pg from 'polygrad{subpath}';
const rt = {'await pg.createAsync' if subpath.endswith('/async') else 'pg.create'}({{core:'wasm'}});
assert.deepEqual(Array.from(new rt.Tensor([1,2,3]).mul(2).toArray()), [2,4,6]);
rt.dispose();
'''
            args = [node, '--input-type=module']
            if condition:
                args.append('--conditions=browser')
            label = (subpath.replace('/', '-') or '-root') + ('-browser' if condition else '-node')
            run(args + ['-e', code], prefix, env, work / f'{lane}{label}.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['python', 'node'])
    parser.add_argument('--npm', default='npm')
    parser.add_argument('--node', default='node')
    args = parser.parse_args()
    (ROOT / 'temp').mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'package-{args.kind}-', dir=ROOT / 'temp'))
    print('Evidence:', work, flush=True)
    env = clean_environment()
    if args.kind == 'python':
        python_install(work, env)
    else:
        node_install(work, env, args.npm, args.node)
    print(f'PASS: isolated {args.kind} install; logs: {work}', flush=True)


if __name__ == '__main__':
    main()
