"""Generated native imports must bind to the selected core, not global symbols."""
import os
import json
import sys
from pathlib import Path
import subprocess
import importlib.util
import ctypes

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_consumer_methods_are_import_roots(tmp_path):
    source = tmp_path / 'author.c'
    source.write_text('#include "polygrad.h"\n'
        'int build(PolyCtx*c,PolyTensor**x,PolyTensor**y){y[0]=poly_tensor_retain(x[0]);return 1;}\n'
        'PolyTensor *extra(PolyCtx*c,PolyTensor*x){return poly_tensor_exp(c,x);}\n')
    rules = tmp_path / 'methods.json'
    rules.write_text(json.dumps({'extra': {'args': {'x': {'tensor': 'borrowed'}},
                                         'return': {'tensor': 'owned'}}}))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/extension_bindings.py'),
        '--source', str(source), '--entry', 'build', '--inputs', '1', '--outputs', '1',
        '--methods', str(rules), '--output', str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert manifest['construction_abi'] == 2
    assert 'poly_tensor_exp' in manifest['imports']
    assert manifest['methods'][0]['name'] == 'extra'
    lib = tmp_path / 'author.so'
    subprocess.run(['cc', '-std=c11', '-fPIC', '-fvisibility=hidden', '-shared',
                    '-I'+str(ROOT/'src'), str(tmp_path/'out/extension.c'),
                    str(tmp_path/'out/methods.c'), '-o', str(lib), '-lm'], check=True)
    loaded = ctypes.CDLL(str(lib))
    for name in ('poly_extension_invoke', 'poly_extension_release', 'poly_extension_free'):
        assert getattr(loaded, name)


def test_consumer_manifest_size_limit(tmp_path):
    source = tmp_path / 'author.c'
    source.write_text('#include "polygrad.h"\n'
        'int build(PolyCtx*c,PolyTensor**x,PolyTensor**y){y[0]=poly_tensor_retain(x[0]);return 1;}\n' +
        '\n'.join(f'int method{i}('+','.join(f'int arg{j}' for j in range(64))+'){return 0;}' for i in range(32)))
    rules = tmp_path/'methods.json'
    rules.write_text(json.dumps({f'method{i}': {'return':'value'} for i in range(32)}))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/extension_bindings.py'),
        '--source',str(source),'--entry','build','--inputs','1','--outputs','1',
        '--methods',str(rules),'--output',str(tmp_path/'out')], capture_output=True,text=True)
    assert result.returncode != 0
    assert '65536' in result.stderr


@pytest.mark.parametrize('rules', [
    {'missing': {'return': 'void'}},
    {'extra': {'return': {'tensor': 'owned'}}},
    {'extra': {'args': {'x': {'tensor': 'owned'}}, 'return': {'tensor': 'owned'}}},
    {'extra': {'args': {'x': {'tensor': 'borrowed'}}, 'return': {'tensor': 'borrowed'}}},
    {'extra': {'args': {'absent': {'tensor': 'borrowed'}}, 'return': {'tensor': 'owned'}}},
])
def test_consumer_method_rules_reject_missing_ownership(generator, rules):
    spec = importlib.util.spec_from_file_location('extension_methods', ROOT / 'scripts/extension_methods.py')
    methods = importlib.util.module_from_spec(spec); spec.loader.exec_module(methods)
    decls = generator.declarations('#include "polygrad.h"\nPolyTensor *extra(PolyCtx *ctx, PolyTensor *x);')
    with pytest.raises(ValueError):
        methods.describe(rules, decls, generator.signature)


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


def test_extension_borrowed_string_contract(generator):
    decls = generator.declarations('int poly_probe(const char *name);')
    extension = {'failure': -1, 'ownership': 'value', 'cleanup': False, 'strings': ['name']}
    generator.validate_rules(decls, {'poly_probe': {'extension': extension}})
    for strings in (['absent'], ['name', 'name'], 'name', [0]):
        with pytest.raises(ValueError, match='string contract'):
            generator.validate_rules(decls, {'poly_probe': {'extension': {**extension, 'strings': strings}}})
    with pytest.raises(ValueError, match='string contract'):
        generator.validate_rules(decls, {'poly_probe': {'extension': {**extension, 'native': True}}})
    with pytest.raises(ValueError, match='unsupported extension argument'):
        generator.validate_rules(decls, {'poly_probe': {'extension': {k:v for k,v in extension.items() if k!='strings'}}})


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
    native_exports = (ROOT / 'js/napi_special_exports.h').read_text(encoding='utf-8')
    assert not re.search(r'DECLARE_NAPI_METHOD\(\s*"poly_', native)
    for name in re.findall(r'DECLARE_NAPI_METHOD\(\s*"(poly_\w+)"', native_exports):
        assert f'static napi_value napi_{name}(' in native
    wasm = (ROOT / 'js/src/core/wasm_common.js').read_text(encoding='utf-8')
    python = (ROOT / 'py/polygrad/_ffi.py').read_text(encoding='utf-8')
    consumers = {
        'python': (set(vars(lib)) | set(re.findall(r'lib\.(poly_\w+)\.(?:argtypes|restype)', python))) - generated,
        'native': set(re.findall(r'DECLARE_NAPI_METHOD\(\s*"(poly_\w+)"', native_exports)),
        'wasm': set(re.findall(r'^    (poly_\w+):', wasm, re.M)),
    }
    # High-level adapters may use JS names (hfImportNew, etc.), rather than
    # expose a same-named ffi property. Require an actual reviewed C callsite.
    wasm_calls = set(re.findall(r'Module\._(poly_\w+)\b', wasm))
    wasm_calls.update(re.findall(r"callDevice\('(poly_\w+)'", wasm))
    consumers['wasm'].update(n for n, r in rules.items()
                             if 'wasm' in r.get('manual', []) and n in wasm_calls)
    for consumer, names in consumers.items():
        assert names == {n for n, r in rules.items()
                         if consumer in r.get('manual', []) + r.get('host', [])}


def test_binding_rules_clean_generation_matches_shipped_glue(generator, tmp_path):
    generator.core(output=tmp_path)
    generated = [p for p in tmp_path.rglob('*') if p.is_file()]
    assert len(generated) == 10
    for path in generated:
        assert path.read_bytes() == (ROOT / path.relative_to(tmp_path)).read_bytes()
    generator.core(check=True, output=tmp_path)


def test_python_special_signatures_come_from_headers(generator):
    source = '''#include <stdint.h>
      typedef struct PolyModel PolyModel;
      typedef struct { int code; } PolyError;
      uint8_t *poly_probe(PolyModel *model, const PolyError *error, int64_t *length);
    '''
    rules = {'poly_probe': {'manual': ['python'],
                           'python': {'pointers': {'error': 'record'}}}}
    output = generator.python_special_bindings(generator.declarations(source), rules, set())
    import ctypes
    from types import SimpleNamespace
    class PolyError(ctypes.Structure):
        _fields_ = [('code', ctypes.c_int)]
    lib = SimpleNamespace(poly_probe=SimpleNamespace())
    namespace = {}
    exec(output, namespace)
    namespace['bind'](lib, {'PolyError': PolyError})
    assert lib.poly_probe.restype == ctypes.POINTER(ctypes.c_uint8)
    assert lib.poly_probe.argtypes == [ctypes.c_void_p, ctypes.POINTER(PolyError),
                                       ctypes.POINTER(ctypes.c_int64)]


def test_python_special_rules_reject_unknown_pointer_names(generator):
    decls = generator.declarations('int poly_probe(int *out);')
    with pytest.raises(ValueError, match='unknown pointer'):
        generator.python_special_bindings(decls, {'poly_probe': {
            'manual': ['python'], 'python': {'pointers': {'typo': 'address'}}}}, set())


@pytest.mark.parametrize('ctype,mode', [('int', 'address'), ('int *', 'bytes'), ('int *', 'record')])
def test_python_pointer_rules_cannot_change_scalar_types(generator, ctype, mode):
    decls = generator.declarations(f'int poly_probe({ctype} out);')
    with pytest.raises(ValueError, match='representation'):
        generator.python_special_bindings(decls, {'poly_probe': {
            'manual': ['python'], 'python': {'pointers': {'out': mode}}}}, set())


def test_python_optional_symbols_and_abi_bootstrap(generator):
    from types import SimpleNamespace
    decls = generator.declarations('int poly_abi_version(void); int poly_optional(void);')
    rules = {'poly_abi_version': {'manual': ['python']},
             'poly_optional': {'manual': ['python'], 'python': {'optional': True}}}
    namespace = {}
    exec(generator.python_special_bindings(decls, rules, set()), namespace)
    lib = SimpleNamespace(poly_abi_version=SimpleNamespace())
    namespace['bind'](lib, {}, only='poly_abi_version')
    assert lib.poly_abi_version.argtypes == []
    namespace['bind'](lib, {})
    with pytest.raises(AttributeError):
        namespace['bind'](SimpleNamespace(), {})


def test_python_ffi_does_not_repeat_c_signatures():
    source = (ROOT / 'py/polygrad/_ffi.py').read_text()
    assert '.argtypes =' not in source
    assert '.restype =' not in source


def test_native_special_registration_comes_from_rules(generator):
    rules = {'poly_probe': {'manual': ['native']},
             'poly_host': {'host': ['native']},
             'poly_hidden': {'exclude': 'C-only'}}
    output = generator.native_special_exports(rules)
    assert 'DECLARE_NAPI_METHOD("poly_probe", napi_poly_probe)' in output
    assert 'DECLARE_NAPI_METHOD("poly_host", napi_poly_host)' in output
    assert 'poly_hidden' not in output


@pytest.mark.parametrize('generation_exit', [0, 17])
def test_npm_prepack_generates_before_mirroring_and_building(generation_exit):
    subprocess.run(['node', '-e', '''
      const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path')
      const assert = require('node:assert/strict'), calls = []
      const generationExit = Number(process.argv[1])
      const prepare = () => vm.runInNewContext(fs.readFileSync('js/scripts/prepack.js', 'utf8'), {
        __dirname: path.resolve('js/scripts'), console,
        process: {env: {}, execPath: process.execPath, exit: c => {throw Error('exit '+c)}},
        require: n => n === 'child_process' ? {spawnSync: (cmd,args) => {
          calls.push([cmd,...args]); return {status:calls.length === 1 ? generationExit : 0}
        }} : n === 'fs' ? {existsSync: () => true} : require(n)
      })
      if (generationExit) {
        assert.throws(prepare, /exit 17/)
        assert.equal(calls.length, 1)
        assert.equal(calls[0].at(-1), 'extension-bindings')
        process.exit(0)
      }
      prepare()
      assert.equal(calls[0].at(-1), 'extension-bindings')
      assert.equal(calls[1].at(-1), 'extension-sdk')
      assert.match(calls[2].at(-1), /sync-csrc.js$/)
      assert.equal(calls[3].at(-1), 'wasm-pkg')
    ''', str(generation_exit)], cwd=ROOT, check=True)


def test_python_package_preparation_generates_only_in_checkout(tmp_path, monkeypatch):
    import runpy
    import shutil
    import setuptools
    package = tmp_path / 'py'
    package.mkdir()
    for name in ('setup.py', 'pyproject.toml'):
        shutil.copyfile(ROOT / 'py' / name, package / name)
    (package / 'csrc').mkdir()
    calls = []
    monkeypatch.setattr(subprocess, 'check_call', lambda cmd: calls.append(cmd))
    monkeypatch.setattr(setuptools, 'setup', lambda **kw: None)
    monkeypatch.chdir(package)
    runpy.run_path(str(package / 'setup.py'))
    assert not calls  # Extracted packages contain generated bindings already.
    (tmp_path / 'scripts').mkdir()
    generator = tmp_path / 'scripts/extension_bindings.py'
    generator.touch()
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/core.h').touch()
    runpy.run_path(str(package / 'setup.py'))
    assert calls == [[sys.executable, str(generator), '--core'],
                     [sys.executable, str(generator), '--sdk', str(package / 'polygrad/_sdk')]]


def test_python_special_struct_layouts_match_c(tmp_path):
    import ctypes
    import shlex
    from polygrad import _ffi
    records = {name: value for name, value in vars(_ffi).items()
               if isinstance(value, type) and issubclass(value, ctypes.Structure)}
    expected, checks = {}, []
    for name, record in records.items():
        expected[name] = ctypes.sizeof(record)
        checks.append(f'printf("{name} %zu\\n", sizeof({name}));')
        for field, _ in record._fields_:
            expected[f'{name}.{field}'] = getattr(record, field).offset
            checks.append(f'printf("{name}.{field} %zu\\n", offsetof({name}, {field}));')
    source = tmp_path / 'abi.c'
    source.write_text('#include <stddef.h>\n#include <stdio.h>\n'
                      '#include "model.h"\n#include "nn/optim.h"\n'
                      '#include "loaders/onnx_loader.h"\nint main(void) {\n'
                      + '\n'.join(checks) + '\n}\n')
    binary = tmp_path / 'abi'
    subprocess.run([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Isrc',
                    str(source), '-o', str(binary)], cwd=ROOT, check=True)
    actual = subprocess.check_output([str(binary)], text=True)
    assert {name: int(size) for name, size in (line.split() for line in actual.splitlines())} == expected


def test_clean_sdist_regenerates_missing_bindings(tmp_path):
    import shutil
    import tarfile
    # No build tree, source mirrors or generated bindings are available here.
    for directory in ('src', 'vendor', 'scripts', 'py'):
        shutil.copytree(ROOT / directory, tmp_path / directory,
                        ignore=shutil.ignore_patterns('csrc', 'build', 'dist', 'tests',
                                                     '__pycache__', '*.egg-info', '*.so',
                                                     '_sdk', '_generated*.py', 'extension_api.json',
                                                     'frontend_exports.h'))
    shutil.copyfile(ROOT / 'Makefile', tmp_path / 'Makefile')
    (tmp_path / 'js').mkdir()
    shutil.copyfile(ROOT / 'js/extension_native.h', tmp_path / 'js/extension_native.h')
    shutil.copyfile(ROOT / 'js/extension_method_native.h', tmp_path / 'js/extension_method_native.h')
    for directory in ('test', 'bench'):
        (tmp_path / directory).mkdir()
    for name in ('extension_methods_fixture.c', 'extension_methods_fixture.json'):
        shutil.copyfile(ROOT / 'test' / name, tmp_path / 'test' / name)
    # Honor pyproject.toml's build requirements even when the test interpreter
    # has an older setuptools than the isolated package builder requires.
    subprocess.run([sys.executable, '-m', 'build', '--sdist'], cwd=tmp_path / 'py',
                   capture_output=True, text=True, encoding='utf-8', check=True)
    artifact, = (tmp_path / 'py/dist').glob('*.tar.gz')
    with tarfile.open(artifact) as archive:
        paths = {n.split('/', 1)[-1] for n in archive.getnames()}
        assert {'polygrad/_generated_ffi.py', 'polygrad/_generated_special_ffi.py',
                'polygrad/extension_api.json', 'csrc/src/frontend_exports.h',
                'polygrad/sdk.py', 'polygrad/_sdk/extension_bindings.py',
                'polygrad/_sdk/extension_methods.py', 'polygrad/_sdk/include/extension_method_native.h',
                'polygrad/_sdk/examples/author.c', 'polygrad/_sdk/examples/methods.json',
                'polygrad/_sdk/include/polygrad.h',
                'polygrad/_sdk/vendor/cjson/cJSON.h'} <= paths
        assert 'scripts/extension_bindings.py' not in paths


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
    assert manifest['construction_abi'] == 1
    assert 'poly_extension_abi_version' in manifest['imports']
    assert 'poly_abi_version' not in manifest['imports']
    assert 'poly_tensor_exp' in manifest['imports']
    assert 'poly_tensor_log' not in manifest['imports']


def test_sdk_author_uses_only_installed_files(generator, tmp_path):
    import shutil
    sdk, consumer = tmp_path / 'sdk', tmp_path / 'consumer'
    generator.sdk(sdk)
    consumer.mkdir()
    shutil.copyfile(ROOT / 'test/extension_macro_fixture.c', consumer / 'author.c')
    subprocess.run([sys.executable, str(sdk / 'extension_bindings.py'), '--source',
                    'author.c', '--entry', 'macro_build', '--inputs', '1', '--outputs', '1',
                    '--output', 'generated'], cwd=consumer, check=True)
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-O2', '-shared', '-fPIC',
                    '-I' + str(sdk / 'include'), 'author.c', 'generated/extension.c',
                    '-lm', '-o', 'author.so'], cwd=consumer, check=True)
    assert (consumer / 'author.so').is_file()


def test_native_thunks_are_hidden_without_consumer_flags(tmp_path):
    library = tmp_path / 'author.so'
    subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-shared', '-fPIC', '-Isrc',
                    'build/extension/extension.c', 'test/extension_fixture.c', '-lm',
                    '-o', str(library)], cwd=ROOT, check=True)
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    assert 'poly_extension_build' in symbols
    assert 'poly_tensor_' not in symbols


def test_native_model_import_contract(generator):
    decls = generator.declarations('#include "model.h"\n')
    rules = {name: {'exclude': 'test'} for name, d in decls.items() if name.startswith('poly_') and d.get('storageClass') != 'static'}
    rules['poly_model_call'] = {'extension': {
        'native': True, 'failure': -1, 'ownership': 'value', 'cleanup': False}}
    generator.validate_rules(decls, rules)
    rules['poly_model_call']['extension']['native'] = 'yes'
    with pytest.raises(ValueError, match='extension contract'):
        generator.validate_rules(decls, rules)


def test_model_exports_do_not_force_core_link_dependencies(generator, tmp_path):
    generator.core(output=tmp_path)
    header = (tmp_path / 'src/frontend_exports.h').read_text()
    assert '#pragma weak poly_model_call\n' in header
    assert '#pragma weak poly_tensor_abs\n' not in header
    assert 'POLY_EXTENSION_SYMBOL(poly_model_call)\n' in header
    assert 'POLY_EXTENSION_SYMBOL(poly_tensor_abs)\n' in header


def test_borrowed_model_method(generator):
    spec = importlib.util.spec_from_file_location('extension_methods', ROOT / 'scripts/extension_methods.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    decls = generator.declarations('#include "model.h"\nint inspect(PolyModel *model);')
    methods, _ = module.describe({'inspect': {'args': {'model': {'model': 'borrowed'}},
                                            'return': 'value'}}, decls, generator.signature)
    assert methods[0]['args'][0]['kind'] == 'model'
