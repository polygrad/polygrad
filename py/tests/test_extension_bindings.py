"""Generated native imports must bind to the selected core, not global symbols."""
import os
import json
import sys
from pathlib import Path
import subprocess
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def generator():
    spec = importlib.util.spec_from_file_location('extension_bindings', ROOT / 'scripts/extension_bindings.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binding_rules_require_explicit_exposure(generator):
    decls = generator.declarations('typedef struct PolyTensor PolyTensor; PolyTensor *poly_tensor_new_op(PolyTensor *x);')
    with pytest.raises(ValueError, match='poly_tensor_new_op.*decision'):
        generator.validate_rules(decls, {})


def test_binding_rules_reject_stale_declarations(generator):
    with pytest.raises(ValueError, match='poly_removed.*declaration'):
        generator.validate_rules({}, {'poly_removed': {'exclude': 'C-only'}})


def test_binding_rules_validate_named_arrays(generator):
    decls = generator.declarations('typedef struct PolyTensor PolyTensor; '
                                   'PolyTensor *poly_tensor_probe(PolyTensor *x, const long *axes, int count);')
    rule = {'frontend': 'direct', 'arrays': {'axes': {'length': 'old_count', 'width': 8, 'direction': 'in'}}}
    with pytest.raises(ValueError, match='old_count'):
        generator.validate_rules(decls, {'poly_tensor_probe': rule})


def test_binding_rules_reject_unknown_properties(generator):
    decls = generator.declarations('int poly_probe(void);')
    with pytest.raises(ValueError, match='unknown.*expos'):
        generator.validate_rules(decls, {'poly_probe': {'expos': True}})


def test_binding_rules_reject_unsupported_exposed_signature(generator):
    decls = generator.declarations('struct Strange; struct Strange *poly_probe(void);')
    with pytest.raises(ValueError, match='poly_probe.*unsupported'):
        generator.validate_rules(decls, {'poly_probe': {'frontend': 'direct'}})


@pytest.mark.parametrize('contract,match', [
    ({'frontend': 'pair'}, 'owned-output'),
    ({'frontend': 'optional'}, 'owned-output'),
    ({'frontend': 'direct', 'manual': ['python']}, 'duplicate handwritten'),
    ({'frontend': 'direct', 'exclude': 'C-only'}, 'exclusion'),
    ({'manual': ['browser']}, 'handwritten'),
    ({'arrays': {}}, 'decision'),
])
def test_binding_rules_reject_contradictory_contracts(generator, contract, match):
    decls = generator.declarations('typedef struct PolyTensor PolyTensor; PolyTensor *poly_probe(void);')
    with pytest.raises(ValueError, match=match):
        generator.validate_rules(decls, {'poly_probe': contract})


def test_binding_rules_ignore_static_header_helpers(generator):
    decls = generator.declarations('static int poly_helper(void) { return 1; }')
    generator.validate_rules(decls, {})


def test_device_calls_require_reviewed_wait_contract(generator):
    decls = generator.declarations('int poly_probe(int value);')
    with pytest.raises(ValueError, match='may_wait'):
        generator.validate_rules(decls, {'poly_probe': {'manual': ['wasm'], 'wasm_call': {}}})


def test_device_calls_check_string_parameter_names(generator):
    decls = generator.declarations('int poly_probe(int value);')
    with pytest.raises(ValueError, match='string.*value'):
        generator.validate_rules(decls, {'poly_probe': {
            'manual': ['wasm'], 'may_wait': True, 'wasm_call': {'strings': ['value']}}})


def test_device_calls_do_not_narrow_i64_to_number(generator):
    decls = generator.declarations('#include <stdint.h>\nint poly_probe(int64_t value);')
    with pytest.raises(ValueError, match='unsupported Wasm call type int64_t'):
        generator.validate_rules(decls, {'poly_probe': {
            'manual': ['wasm'], 'may_wait': True, 'wasm_call': {}}})


def test_device_calls_keep_signatures_out_of_handwritten_wasm():
    import re
    source = (ROOT / 'js/src/core/wasm_common.js').read_text(encoding='utf-8')
    assert not re.search(r'Module\.ccall\s*\(', source)


def test_device_calls_remain_exported_and_instrumented(generator):
    import re
    rules = generator.load_rules(generator.core_declarations())
    selected = {n for n, r in rules.items() if 'wasm_call' in r}
    exports = subprocess.check_output([sys.executable, 'scripts/wasm_exports.py', 'js/src'],
                                      cwd=ROOT, text=True).strip().split(',')
    assert {'_' + n for n in selected} <= set(exports)
    # The public contract is not a substitute for instrumenting internal C frames.
    make = (ROOT / 'Makefile').read_text(encoding='utf-8')
    only = re.search(r'^WASM_ASYNCIFY_ONLY = (.*)$', make, re.M).group(1)
    assert selected <= set(re.findall(r"'([^']+)'", only))


def test_device_calls_derive_signatures_and_preserve_promise_errors(generator, tmp_path):
    decls = generator.declarations('int poly_probe(void *ctx, const char *device);')
    rules = {'poly_probe': {'manual': ['wasm'], 'may_wait': True,
                            'wasm_call': {'strings': ['device']}}}
    generator.validate_rules(decls, rules)
    target = tmp_path / 'calls.cjs'
    target.write_text(generator.device_call_bindings(decls, rules))
    subprocess.run(['node', '-e', '''
      const assert = require('node:assert/strict')
      const create = require(process.argv[1])
      const seen = []
      let failure = false
      const call = create({ccall(...args) {
        seen.push(args)
        if (!args[4]) return 9
        return failure ? Promise.reject(new Error('device failed')) : Promise.resolve(7)
      }})
      async function main() {
        assert.equal(await call('poly_probe', [123, 'WEBGPU']), 7)
        assert.deepEqual(seen[0], ['poly_probe', 'number', ['number','string'],
                                  [123,'WEBGPU'], {async:true}])
        assert.equal(call('poly_probe', [123, 'CPU'], false), 9)
        assert.throws(() => call('poly_unknown', []), /unreviewed/)
        assert.throws(() => call('poly_probe', [123]), /argument count/)
        assert.equal(seen.length, 2)
        failure = true
        await assert.rejects(call('poly_probe', [123, 'WEBGPU']), /device failed/)
      }
      main().catch(error => { console.error(error); process.exitCode = 1 })
    ''', str(target)], check=True)


def test_binding_rules_host_helpers_cannot_hide_c_declarations(generator):
    rule = {'poly_host': {'host': ['native']}}
    generator.validate_rules({}, rule)
    decls = generator.declarations('int poly_host(void);')
    with pytest.raises(ValueError, match='host-only'):
        generator.validate_rules(decls, rule)


def test_binding_rules_exposure_is_not_selected_by_prefix(generator):
    decls = generator.declarations('typedef struct PolyTensor PolyTensor; PolyTensor *poly_probe(PolyTensor *x);')
    rules = {'poly_probe': {'frontend': 'direct'}}
    generator.validate_rules(decls, rules)
    files = generator.frontend_bindings(decls, rules)
    assert 'lib.poly_probe.argtypes = [ctypes.c_void_p]' in files['py/polygrad/_generated_ffi.py']
    assert 'napi_poly_probe' in files['js/napi_generated.h']
    assert 'poly_probe:' in files['js/src/core/generated.js']


@pytest.mark.parametrize('array,match', [
    ({'length': 'count', 'width': 4, 'direction': 'in'}, 'unsupported array'),
    ({'length': 'count', 'width': 8, 'direction': 'out'}, 'unsupported array'),
    ({'length': 'count', 'width': 8, 'direction': 'in', 'limit': 0}, 'unsupported array'),
    ({'length': 1, 'width': 8, 'direction': 'in'}, 'negative fixed count'),
    ({'length': 'count', 'width': 8, 'direction': 'in', 'nullablee': True}, 'invalid array'),
])
def test_binding_rules_reject_invalid_array_layout(generator, array, match):
    decls = generator.declarations('#include <stdint.h>\ntypedef struct PolyTensor PolyTensor; '
                                   'PolyTensor *poly_probe(const int64_t *axes, int count);')
    with pytest.raises(ValueError, match=match):
        generator.validate_rules(decls, {'poly_probe': {'frontend': 'direct', 'arrays': {'axes': array}}})


def test_binding_rules_reject_duplicate_json_keys(generator, tmp_path, monkeypatch):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/binding_rules.json').write_text('{"poly_f":{},"poly_f":{}}')
    monkeypatch.setattr(generator, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='duplicate binding key: poly_f'):
        generator.load_rules({})


def test_binding_rules_handwritten_consumers_are_explicit(generator):
    import re
    from polygrad import _ffi

    class Function:
        pass

    class Library:
        def __getattr__(self, name):
            fn = Function()
            setattr(self, name, fn)
            return fn

    lib = Library()
    _ffi._declare_signatures(lib)
    decls = generator.core_declarations()
    rules = generator.load_rules(decls)
    generated = {n for n, r in rules.items() if 'frontend' in r}
    native = (ROOT / 'js/napi_api.c').read_text(encoding='utf-8')
    wasm = (ROOT / 'js/src/core/wasm_common.js').read_text(encoding='utf-8')
    python = (ROOT / 'py/polygrad/_ffi.py').read_text(encoding='utf-8')
    consumers = {
        'python': (set(vars(lib)) | set(re.findall(r'lib\.(poly_\w+)\.(?:argtypes|restype)', python))) - generated,
        'native': set(re.findall(r'DECLARE_NAPI_METHOD\("(poly_\w+)"', native)),
        'wasm': set(re.findall(r'^    (poly_\w+):', wasm, re.M)),
    }
    for consumer, names in consumers.items():
        assert names == {n for n, r in rules.items()
                         if consumer in r.get('manual', []) + r.get('host', [])}


def test_binding_rules_clean_generation_matches_shipped_glue(generator, tmp_path):
    generator.core(output=tmp_path)
    generated = [p for p in tmp_path.rglob('*') if p.is_file()]
    assert len(generated) == 8
    for path in generated:
        assert path.read_bytes() == (ROOT / path.relative_to(tmp_path)).read_bytes()
    generator.core(check=True, output=tmp_path)


def test_binding_regeneration_does_not_invalidate_unchanged_build_inputs(generator, tmp_path):
    generator.core(output=tmp_path)
    files = [p for p in tmp_path.rglob('*') if p.is_file()]
    # Use a past timestamp so coarse filesystem precision cannot hide a rewrite.
    for path in files:
        os.utime(path, ns=(1000000000, 1000000000))
    generator.core(output=tmp_path)
    assert all(p.stat().st_mtime_ns == 1000000000 for p in files)
    changed = tmp_path / 'js/src/core/generated_device_calls.js'
    changed.write_text('stale')
    generator.core(output=tmp_path)
    assert changed.read_bytes() == (ROOT / 'js/src/core/generated_device_calls.js').read_bytes()


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
