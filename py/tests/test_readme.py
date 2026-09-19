"""Execute README examples verbatim, with explicit setup/exclusions beside fences."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from polygrad import _ffi


ROOT = Path(__file__).resolve().parents[2]
READMES = ('README.md', 'py/README.md', 'js/README.md')


def blocks():
    for path in READMES:
        text = (ROOT / path).read_text(encoding='utf-8')
        for i, match in enumerate(re.finditer(r'```(\w+)\n(.*?)```', text, re.S)):
            annotation = re.search(r'<!-- readme-test: ([\w-]+) -->\s*$', text[:match.start()])
            yield path, i, match[1], match[2], annotation[1] if annotation else 'run'


BLOCKS = list(blocks())


def run_example(lang, code, work):
    env = dict(os.environ, POLY_LIB=str(_ffi._lib._name), PYTHONPATH=str(ROOT / 'py'),
               POLY_DEV='CPU', DEV='CPU', POLY_DEBUG='0')
    if lang == 'python':
        command = [sys.executable, '-c', code]
    else:
        node = shutil.which('node')
        assert node, 'Node is required for README example checks'
        # Resolve the checkout from the isolated example directory, including /async.
        code = code.replace("require('polygrad')", f'require({json.dumps(str(ROOT / "js"))})')
        code = code.replace("require('polygrad/async')", f'require({json.dumps(str(ROOT / "js/src/index.async.js"))})')
        command = [node, '-e', code]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope='module')
def model_bundle(tmp_path_factory):
    work = tmp_path_factory.mktemp('readme-model')
    code = next(code for path, _, lang, code, _ in BLOCKS
                if path == 'py/README.md' and lang == 'python' and 'class Linear:' in code)
    run_example('python', code, work)
    return (work / 'linear.pgb').read_bytes()


@pytest.mark.parametrize('path,index,lang,code,mode', BLOCKS,
                         ids=[f'{p}:{i}:{lang}' for p, i, lang, _, _ in BLOCKS])
def test_readme_block(path, index, lang, code, mode, tmp_path, request):
    assert mode in {'run', 'browser', 'network', 'config', 'export', 'package'}, mode
    if mode == 'browser':
        pytest.skip('Browser-only example: requires bundling/HTML and a browser; not certified by Node')
    if mode == 'network' and os.environ.get('README_NETWORK') != '1':
        pytest.skip('HF download example: opt in with README_NETWORK=1')
    if lang == 'text':
        return  # Diagrams, not executable examples.
    if lang == 'bash':
        result = subprocess.run(['bash', '-n'], input=code, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return  # Validate syntax without installing packages or starting release gates.
    if lang == 'json':
        json.loads(code)
        return
    assert lang in {'python', 'js', 'javascript'}, f'Unclassified example: {path}:{index} ({lang})'

    if "linear.pgb" in code or mode == 'export':
        (tmp_path / 'linear.pgb').write_bytes(request.getfixturevalue('model_bundle'))
    if mode == 'config':
        config = next(c for _, _, language, c, _ in BLOCKS if language == 'json')
        (tmp_path / 'network.json').write_text(config, encoding='utf-8')
        if lang != 'python':
            code = "const pg = require('polygrad');\nconst config = " + config + ';\n' + code
    if mode == 'export':
        if lang == 'python':
            code = ('from polygrad import Model\nmodel = Model.load("linear.pgb")\n'
                    'input_array = [3., 4., 5., 6., 7.]\nmodel.forward(x=input_array)\n' + code +
                    '\nassert abs(float(result["prediction"][0]) - 11) < 0.1\n')
        else:
            code = ("const { Model } = require('polygrad');\nconst model = Model.load('linear.pgb');\n"
                    'const inputArray = new Float32Array([3,4,5,6,7]);\nmodel.forward({x:inputArray});\n' + code +
                    "\nif (Math.abs(result.prediction[0] - 11) > 0.1) throw Error('prediction mismatch');\n")
    if mode == 'package':
        parts = [c for p, _, _, c, m in BLOCKS if p == path and m == 'package']
        assert len(parts) == 2
        consumer = next(c for c in parts if 'SomePackage.create' in c)
        implementation = next(c for c in parts if c.startswith('function create'))
        code = implementation + '\nconst SomePackage = {create};\n' + consumer

    if lang == 'python' and 'normalize(x: Tensor)' in code:
        code += '\nassert abs(float(normalize(Tensor([1., 2., 3.])).mean().item())) < 1e-5\n'
    if 'custom_add_4' in code:
        code += ('\nassert y.numpy().tolist() == [11., 22., 33., 44.]\n' if lang == 'python' else
                 "\nif (JSON.stringify(Array.from(y.toArray())) !== '[11,22,33,44]') throw Error('kernel values');\n")
    output = run_example(lang, code, tmp_path)
    if "Model.load('linear.pgb')" in code and mode != 'export':
        values = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', output)]
        assert values == pytest.approx([11, 14, 17, 20, 23], abs=0.1), output
