"""Generated native imports must bind to the selected core, not global symbols."""
import os
import json
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_import_discovery_follows_macro_and_helper_calls(tmp_path):
    subprocess.run([sys.executable, 'scripts/extension_bindings.py', '--source',
                    'test/extension_macro_fixture.c', '--entry', 'macro_build',
                    '--inputs', '1', '--outputs', '1', '--output', str(tmp_path)],
                   cwd=ROOT, check=True)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert 'poly_tensor_exp' in manifest['imports']
    assert 'poly_tensor_log' not in manifest['imports']


def test_native_thunks_are_hidden_without_consumer_flags(tmp_path):
    library = tmp_path / 'author.so'
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-shared', '-fPIC', '-Isrc',
                    'build/extension/extension.c', 'test/extension_fixture.c', '-lm',
                    '-o', str(library)], cwd=ROOT, check=True)
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    assert 'poly_extension_build' in symbols
    assert 'poly_tensor_' not in symbols
