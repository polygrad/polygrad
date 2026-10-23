#!/usr/bin/env python3
"""Generate bindings to existing C declarations, never a second tensor API.

--core refreshes the resolver/export metadata. --source builds a consumer's
typed imports and a construction entry adapter (ctx, tensors, scalars,
owned outputs). binding_rules.json owns exposure and pointer rules; C headers
own signatures. Handwritten adapters remain explicit rather than auto-selected.

Rules select a shared frontend adapter (direct, status, owned pair, or optional
owned output), construction extension metadata, handwritten consumers, or a
C-only exclusion. Host-only helpers explicitly have no C declaration.
Arrays name their count argument; a negative count means a
fixed length. The width is bytes per item on the extension/Wasm wire, and limit
is a multiple of POLY_MAX_DIMS for frontend dimension arrays. Opaque input
handles remain nullable at the ABI boundary; C owns operation-specific checks.

may_wait records a reviewed C-level device-wait possibility, not a Promise
return type. wasm_call selects generated ccall signatures for marshalled args;
only its named strings are converted by Emscripten. Other pointers are numeric
addresses owned by the calling adapter. Admission, cleanup and Asyncify's
internal stack instrumentation remain outside these call-signature adapters.
An absent may_wait means unaudited, never a nonblocking guarantee.

Python special adapters also derive signatures from headers. Their rules only
select record, raw-address or byte-string pointer representations and optional
symbols; records/callback objects stay owned by the Python frontend. Native
special-adapter registrations come from the same exposure decisions.
"""

import argparse
import json
from pathlib import Path
import subprocess
import shutil
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SDK = Path(__file__).resolve().parent
INSTALLED = (SDK / 'include/polygrad.h').is_file()
EXPORT_DEFINITIONS = [
    '#ifdef _WIN32', '#define EXT __declspec(dllexport)', '#define POLY_EXTENSION_LOCAL',
    '#else', '#define EXT __attribute__((visibility("default")))',
    '#define POLY_EXTENSION_LOCAL __attribute__((visibility("hidden")))', '#endif',
]


def include_dir():
    return SDK / 'include' if INSTALLED else ROOT / 'src'


def construction_abi():
    macros = subprocess.check_output(
        ['clang', '-x', 'c', f'-I{include_dir()}', '-dM', '-E', '-'],
        input=b'#include "frontend.h"\n').decode().splitlines()
    return int(next(line.split()[2] for line in macros
                    if line.startswith('#define POLYGRAD_EXTENSION_ABI_VERSION ')))


def sdk(output):
    """Package source-owned headers and this generator, not another API list."""
    if INSTALLED:
        raise ValueError('SDK preparation requires the repository')
    files = {Path('include') / p.relative_to(ROOT / 'src'): p
             for p in (ROOT / 'src').rglob('*.h')}
    files.update({p.relative_to(ROOT): p for p in (ROOT / 'vendor/cjson').glob('*.h')})
    files.update({Path('extension_bindings.py'): Path(__file__),
                  Path('extension_methods.py'): SDK / 'extension_methods.py',
                  Path('extension_api.json'): ROOT / 'js/src/extension_api.json',
                  Path('examples/author.c'): ROOT / 'test/extension_methods_fixture.c',
                  Path('examples/methods.json'): ROOT / 'test/extension_methods_fixture.json',
                  Path('include/extension_native.h'): ROOT / 'js/extension_native.h',
                  Path('include/extension_method_native.h'): ROOT / 'js/extension_method_native.h'})
    for relative, source in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != source.read_bytes():
            shutil.copyfile(source, target)
    print(f'prepared author SDK: {output}')


# Declaration roots used by the frontends, including their handwritten adapters.
# Adding a header audits its transitive C API without granting it exposure.
CORE_HEADERS = (
    'polygrad.h', 'model.h', 'frontend.h', 'device.h', 'nn/nn.h', 'nn/optim.h',
    'models/models.h', 'models/mlp.h', 'models/transformer.h', 'tokenizer.h',
    'loaders/hf_decode.h', 'loaders/gguf_decode.h', 'loaders/onnx_loader.h',
    'loaders/gguf_loader.h', 'models/hf_loader.h', 'loaders/import_error.h',
    'bundle.h', 'engine/schedule.h', 'engine/realize.h', 'codegen/codegen.h',
)
OPAQUE = {"PolyCtx *", "PolyTensor *", "PolyUOp *", "PolyTensorCapture *", "PolyModel *"}
SCALARS = {
    "int",
    "_Bool",
    "bool",
    "int64_t",
    "uint64_t",
    "double",
    "float",
    "PolyOps",
    "PolyDevice",
}


def frontend_bindings(decls, rules):
    """Emit the explicitly selected, common Python/native JS/Wasm adapters."""
    macros = subprocess.check_output(
        ['clang', '-x', 'c', f'-I{ROOT}/src', '-dM', '-E', '-'],
        input=b'#include "polygrad.h"\n',
    ).decode().splitlines()
    max_dims = int(next(line.split()[2] for line in macros if line.startswith('#define POLY_MAX_DIMS ')))
    selected = {name: signature(decls[name]) for name, rule in sorted(rules.items())
                if 'frontend' in rule}
    native, exports = [], []
    wasm = ["// Generated by scripts/extension_bindings.py --core.\n'use strict'\nconst { integer64 } = require('./integer')\n",
            "module.exports = (Module, callWithInt64, callWithArrays, callTensorPair) => ({"]
    python = ["# Generated by scripts/extension_bindings.py --core.\nimport ctypes\n\ndef bind(lib):"]
    ctypes_types = {"int": "c_int", "PolyOps": "c_int", "PolyDevice": "c_int",
                    "bool": "c_bool", "_Bool": "c_bool", "int64_t": "c_int64",
                    "uint64_t": "c_uint64", "double": "c_double", "float": "c_float"}
    for name, (ret, args) in selected.items():
        rule = rules[name]
        pair = rule['frontend'] == 'pair'
        optional = rule['frontend'] == 'optional'
        inputs = args[:-2] if pair else args
        names = [f"a{i}" for i in range(len(args))]
        input_names = names[:len(inputs)]
        native += [f"static napi_value napi_{name}(napi_env env, napi_callback_info info) {{",
                   f"  napi_value argv[{max(1,len(inputs))}];", f"  size_t argc = {len(inputs)};",
                   "  NAPI_CALL(env, napi_get_cb_info(env, info, &argc, argv, NULL, NULL));",
                   f'  if (argc != {len(inputs)}) {{ napi_throw_type_error(env, NULL, "{name}: wrong argument count"); return NULL; }}']
        js_args, py_args, arrays, c_args, strings = [], [], [], [], []
        for i, arg in enumerate(inputs):
            t = arg["type"]["qualType"]
            if optional and i == len(args) - 1:
                t = "bool"
            n = names[i]
            c_args.append(n)
            if t in ("int64_t *", "const int64_t *", "int64_t (*)[2]"):
                array = rule['arrays'][arg['name']]
                count_name, width = array['length'], array['width']
                elements = width // 8
                count = next(j for j, a in enumerate(args) if a['name'] == count_name)
                limit = array.get('limit', 1)
                arrays.append((i, count, elements, limit))
                native += [f"  int64_t {n}[POLY_MAX_DIMS * {elements * limit}];", f"  uint32_t length{i};",
                           f"  NAPI_CALL(env, napi_get_array_length(env, argv[{i}], &length{i}));",
                           f'  if (length{i} > POLY_MAX_DIMS * {elements * limit}) {{ napi_throw_range_error(env, NULL, "too many dimensions"); return NULL; }}',
                           f"  if (read_int64_array(env, argv[{i}], {n}, POLY_MAX_DIMS * {elements * limit}) < 0) return NULL;"]
                if elements == 2 and t == "int64_t (*)[2]":
                    c_args[-1] = f"(int64_t (*)[2]){n}"
                js_args.append(n)
                # Python passes flattened pairs, with the same contiguous C layout.
                py_args.append("ctypes.POINTER(ctypes.c_int64)")
            elif t == "const char *":
                strings.append(i)
                js_args.append(n)
                py_args.append("ctypes.c_char_p")
            elif t.removeprefix("const ") in OPAQUE:
                native.append(f"  {t} {n} = get_external_nullable(env, argv[{i}]);")
                js_args.append(f"({n} || 0)")
                py_args.append("ctypes.c_void_p")
            else:
                py_args.append("ctypes." + ctypes_types[t])
                if t in ("bool", "_Bool"):
                    native += [f"  bool {n};", f"  napi_value b{i};",
                               f"  NAPI_CALL(env, napi_coerce_to_bool(env, argv[{i}], &b{i}));",
                               f"  NAPI_CALL(env, napi_get_value_bool(env, b{i}, &{n}));"]
                    js_args.append(f"+!!{n}")
                elif t in ("int64_t", "uint64_t"):
                    native += [f"  {t} {n};",
                               f"  if (!generated_read_{t}(env, argv[{i}], &{n})) return NULL;"]
                    js_args.append(f"integer64({n}, {str(t == 'uint64_t').lower()})")
                elif t in ("double", "float"):
                    native += [f"  double {n};", f"  NAPI_CALL(env, napi_get_value_double(env, argv[{i}], &{n}));"]
                    js_args.append(n)
                else:
                    native += [f"  int32_t {n};", f"  napi_value v{i};",
                               f"  NAPI_CALL(env, napi_coerce_to_number(env, argv[{i}], &v{i}));",
                               f"  NAPI_CALL(env, napi_get_value_int32(env, v{i}, &{n}));"]
                    js_args.append(f"Number({n})")
        for index, count, elements, limit in arrays:
            # Check the original Number, not the already narrowed int32 value.
            native += [f'  double count{index};',
                       f'  NAPI_CALL(env, napi_get_value_double(env, argv[{count}], &count{index}));',
                       f'  if (count{index} < 0 || count{index} > POLY_MAX_DIMS * {limit} || count{index} != floor(count{index}) || count{index} * {elements} != length{index}) {{ napi_throw_range_error(env, NULL, "array length mismatch"); return NULL; }}']
        # Decode strings last: no earlier argument error can leak an allocation.
        for i in strings:
            native += [f"  char *a{i} = NULL;", f"  size_t bytes{i};",
                       f"  napi_status status{i} = napi_get_value_string_utf8(env, argv[{i}], NULL, 0, &bytes{i});",
                       f"  if (status{i} == napi_ok && (a{i} = malloc(bytes{i} + 1)))",
                       f"    status{i} = napi_get_value_string_utf8(env, argv[{i}], a{i}, bytes{i} + 1, NULL);",
                       f"  if (status{i} != napi_ok || !a{i}) {{"]
            native += [f"    free(a{j});" for j in strings if j <= i]
            native += ['    napi_throw_type_error(env, NULL, "expected a UTF-8 string"); return NULL;', "  }"]
        if rule['frontend'] == 'owned_string':
            native += [f"  char *text = {name}({', '.join(c_args)});"]
            native += [f"  free(a{i});" for i in strings]
            native += ["  napi_value result;", "  napi_status status = text ? napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &result) : napi_get_null(env, &result);",
                       "  free(text);", "  NAPI_CALL(env, status);", "  return result;", "}"]
        elif rule['frontend'] == 'status':
            native += [f"  int rc = {name}({', '.join(c_args)});"]
            native += [f"  free(a{i});" for i in strings]
            native += ["  napi_value result;", "  NAPI_CALL(env, napi_create_int32(env, rc, &result));",
                       "  return result;", "}"]
        elif optional:
            py_args[-1] = "ctypes.POINTER(ctypes.c_void_p)"
            native += ["  PolyTensor *second = NULL;",
                       f"  PolyTensor *first = {name}({', '.join(c_args[:-1])}, {names[-1]} ? &second : NULL);",
                       f'  if (first && {names[-1]}) return generated_tensor_pair(env, 0, first, second, "{name}");',
                       "  if (second) poly_tensor_release(second);",
                       "  return make_external(env, first);", "}"]
        elif pair:
            native += ["  PolyTensor *first = NULL, *second = NULL;",
                       f"  int rc = {name}({', '.join(c_args)}, &first, &second);",
                       f'  return generated_tensor_pair(env, rc, first, second, "{name}");', "}"]
            py_args += ["ctypes.POINTER(ctypes.c_void_p)"] * 2
        else:
            native += [f"  return make_external(env, {name}({', '.join(c_args)}));", "}"]
        exports.append(f'DECLARE_NAPI_METHOD("{name}", napi_{name}),')
        if strings:
            types = ['string' if i in strings else 'number' for i in range(len(args))]
            call = f"Module.ccall('{name}', 'number', {json.dumps(types)}, [{', '.join(js_args)}])"
            if rule['frontend'] == 'owned_string':
                wasm.append(f"  {name}: ({', '.join(names)}) => {{ const p = {call} >>> 0; try {{ return p ? Module.UTF8ToString(p) : null }} finally {{ if (p) Module._free(p) }} }},")
            else:
                wasm.append(f"  {name}: ({', '.join(names)}) => {call},")
        elif arrays:
            wasm += [f"  {name}: ({', '.join(input_names)}) => {{"]
            for index, count, elements, limit in arrays:
                wasm += [f"    if (!Array.isArray(a{index}) || !Number.isInteger(a{count}) || a{count} < 0 || a{count} > {max_dims * limit} || a{index}.length !== a{count} * {elements}) throw new RangeError('invalid dimension array')",
                         f"    for (const value of a{index}) integer64(value)"]
            if len(arrays) == 1 and arrays[0][0] == 2:
                wasm.append(f"    return callWithInt64(Module._{name}, {', '.join(js_args)})")
            else:
                fn = f"(...args) => callTensorPair(Module._{name}, args, '{name}', true)" if optional else f"Module._{name}"
                wasm.append(f"    return callWithArrays({fn}, [{', '.join(js_args)}], {json.dumps([a[0] for a in arrays])})")
            wasm.append("  },")
        elif pair:
            wasm.append(f"  {name}: ({', '.join(input_names)}) => callTensorPair(Module._{name}, [{', '.join(js_args)}], '{name}'),")
        else:
            wasm.append(f"  {name}: ({', '.join(names)}) => Module._{name}({', '.join(js_args)}),")
        python += [f"    lib.{name}.restype = ctypes.{'c_int' if pair or rule['frontend'] == 'status' else 'c_void_p'}",
                   f"    lib.{name}.argtypes = [{', '.join(py_args)}]"]
    wasm.append("})\n")
    comment = "/* Generated by scripts/extension_bindings.py --core. */\n"
    return {
        "js/napi_generated.h": comment + "\n".join(native) + "\n",
        "js/napi_generated_exports.h": comment + "\n".join(exports) + "\n",
        "js/src/core/generated.js": "\n".join(wasm),
        "py/polygrad/_generated_ffi.py": "\n".join(python) + "\n",
    }


def declaration_ast(source):
    return json.loads(
        subprocess.check_output(
            [
                "clang",
                "-x",
                "c",
                "-std=c11",
                f"-I{include_dir()}",
                "-Xclang",
                "-ast-dump=json",
                "-fsyntax-only",
                "-",
            ],
            input=source.encode(),
        )
    )


def declarations(source):
    return {v["name"]: v for v in declaration_ast(source)["inner"] if v["kind"] == "FunctionDecl"}


def signature(d):
    return d["type"]["qualType"].split("(")[0].strip(), [
        v for v in d.get("inner", []) if v["kind"] == "ParmVarDecl"
    ]


def core_source():
    # Audit conditional declarations regardless of this machine's devices.
    # The headers use opaque driver types and do not require either SDK.
    return ('#define POLY_HAS_CUDA 1\n#define POLY_HAS_HIP 1\n'
            + ''.join(f'#include "{header}"\n' for header in CORE_HEADERS))


def core_declarations():
    return declarations(core_source())


def python_special_bindings(decls, rules, enums):
    """C owns signatures; rules describe Python pointer representation only.

    Opaque objects stay void handles. A record pointer explicitly opts into the
    frontend's ctypes Structure; byte/address overrides preserve buffer callers
    without repeating the pointed-to C type in metadata.
    """
    scalars = {t: 'ctypes.c_' + t for t in ('int', 'float', 'double', 'size_t')}
    scalars.update({t + '_t': 'ctypes.c_' + t for t in
                    ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64')})
    scalars.update({'_Bool': 'ctypes.c_bool', 'bool': 'ctypes.c_bool',
                    'char': 'ctypes.c_char', 'void': 'None'})
    scalars.update({t: 'ctypes.c_int' for t in enums})

    def ctype(t, mode=None):
        t = t.removeprefix('const ')
        if mode in ('address', 'bytes'):
            if '*' not in t:
                raise ValueError(f'pointer representation on non-pointer {t}')
            if mode == 'bytes' and t not in ('char *', 'uint8_t *'):
                raise ValueError(f'byte-string representation on non-byte pointer {t}')
            return 'ctypes.c_void_p' if mode == 'address' else 'ctypes.c_char_p'
        if mode not in (None, 'record'):
            raise ValueError(f'unknown pointer representation {mode}')
        if t == 'void *':
            return 'ctypes.c_void_p'
        if t == 'char *':
            return 'ctypes.c_char_p'
        if t.endswith('*'):
            base = t[:-1].strip()
            if base.startswith('Poly') and '*' not in base and mode != 'record':
                return 'ctypes.c_void_p'
            return f'ctypes.POINTER({ctype(base, mode)})'
        if t in scalars:
            if mode == 'record':
                raise ValueError(f'record representation on scalar {t}')
            return scalars[t]
        if t.startswith('Poly') and t.isidentifier():
            return f'types[{t!r}]'
        raise ValueError(f'unsupported Python C type {t}')

    lines = ['# Generated by scripts/extension_bindings.py --core.', 'import ctypes', '',
             'def bind(lib, types, only=None):']
    for name, rule in sorted(rules.items()):
        if 'python' not in rule.get('manual', []):
            continue
        policy = rule.get('python', {})
        if not isinstance(policy, dict) or policy.keys() - {'pointers', 'optional'}:
            raise ValueError(f'{name}: invalid Python contract')
        pointers = policy.get('pointers', {})
        ret, args = signature(decls[name])
        names = {a['name'] for a in args} | {'return'}
        if not isinstance(pointers, dict) or pointers.keys() - names:
            raise ValueError(f'{name}: unknown pointer names')
        if 'optional' in policy and type(policy['optional']) is not bool:
            raise ValueError(f'{name}: optional must be boolean')
        result = ctype(ret, pointers.get('return'))
        params = [ctype(a['type']['qualType'], pointers.get(a['name'])) for a in args]
        lines += [f'    if only is None or only == {name!r}:']
        if policy.get('optional'):
            lines += [f'        fn = getattr(lib, {name!r}, None)', '        if fn is not None:']
            indent = '            '
        else:
            lines += [f'        fn = lib.{name}']
            indent = '        '
        lines += [f'{indent}fn.restype = {result}', f'{indent}fn.argtypes = [{", ".join(params)}]']
    return '\n'.join(lines) + '\n'


def device_call_signature(decl, rule):
    name = decl['name']
    if rule.get('may_wait') is not True:
        raise ValueError(f'{name}: wasm_call requires reviewed may_wait=true')
    call = rule['wasm_call']
    if not isinstance(call, dict) or call.keys() - {'strings'}:
        raise ValueError(f'{name}: invalid wasm_call contract')
    ret, args = signature(decl)
    strings = call.get('strings', [])
    params = {a['name']: a['type']['qualType'] for a in args}
    if (not isinstance(strings, list) or len(set(strings)) != len(strings)
            or any(params.get(n) != 'const char *' for n in strings)):
        raise ValueError(f'{name}: invalid string parameters {strings}')

    def wire_type(t):
        # size_t is wasm32 here, irrespective of the host running Clang.
        # i64 scalars require a separate BigInt contract, never Number coercion.
        if t == 'void':
            return None
        if (t.endswith('*') or t in (SCALARS - {'int64_t', 'uint64_t'}) | {'size_t', 'uint32_t'}):
            return 'number'
        raise ValueError(f'{name}: unsupported Wasm call type {t}')
    return wire_type(ret), ['string' if a['name'] in strings else wire_type(a['type']['qualType']) for a in args]


def device_call_bindings(decls, rules):
    lines = ['// Generated by scripts/extension_bindings.py --core.', "'use strict'", '',
             '// Callers own admission and keep argument allocations alive until settlement.',
             'module.exports = Module => {', '  const calls = {']
    for name, rule in sorted(rules.items()):
        if 'wasm_call' not in rule:
            continue
        ret, types = device_call_signature(decls[name], rule)
        lines += [f'    {name}: (args, options) => {{',
                  f"      if (args.length !== {len(types)}) throw new TypeError('{name}: wrong argument count')",
                  f"      return Module.ccall('{name}', {json.dumps(ret)}, {json.dumps(types)}, args, options)",
                  '    },']
    lines += ['  }', '  return (name, args, asynchronous = true) => {',
              "    if (!Object.prototype.hasOwnProperty.call(calls, name)) throw new Error('unreviewed device call: ' + name)",
              '    return calls[name](args, asynchronous ? { async: true } : undefined)',
              '  }', '}', '']
    return '\n'.join(lines)


def validate_rules(decls, rules):
    """Fail closed on new declarations, stale contracts and unsupported adapters.

    The umbrella includes internal declarations too. Each has an explicit
    C-only or handwritten decision; type compatibility never grants exposure.
    Widths are extension/Wasm wire widths, not native sizeof(pointer).
    """
    if not isinstance(rules, dict):
        raise ValueError('expected a binding rule table')
    public = {n: d for n, d in decls.items()
              if n.startswith('poly_') and d.get('storageClass') != 'static'}
    for name in sorted(public.keys() - rules.keys()):
        raise ValueError(f'{name}: missing binding decision')
    for name in sorted(rules.keys() - public.keys()):
        if not isinstance(rules[name], dict) or 'host' not in rules[name]:
            raise ValueError(f'{name}: binding rule has no C declaration')
    for name, rule in rules.items():
        if not isinstance(rule, dict):
            raise ValueError(f'{name}: expected binding rule object')
        unknown = rule.keys() - {'frontend', 'extension', 'arrays', 'manual', 'exclude', 'host', 'may_wait', 'wasm_call', 'python'}
        if unknown:
            raise ValueError(f'{name}: unknown binding properties {sorted(unknown)}')
        if 'host' in rule:
            consumers = rule['host']
            if (len(rule) != 1 or name in public or not isinstance(consumers, list)
                    or not consumers or len(set(consumers)) != len(consumers)
                    or set(consumers) - {'python', 'native', 'wasm'}):
                raise ValueError(f'{name}: invalid host-only helper')
            continue
        if 'exclude' in rule:
            if len(rule) != 1 or not isinstance(rule['exclude'], str) or not rule['exclude'].strip():
                raise ValueError(f'{name}: exclusion needs a reason and cannot expose adapters')
            continue
        if not rule.keys() & {'frontend', 'extension', 'manual', 'wasm_call'}:
            raise ValueError(f'{name}: missing binding decision')
        if 'may_wait' in rule and type(rule['may_wait']) is not bool:
            raise ValueError(f'{name}: may_wait must be a C-level boolean')
        if 'python' in rule and 'python' not in rule.get('manual', []):
            raise ValueError(f'{name}: Python contract requires a Python adapter')
        if 'wasm_call' in rule:
            device_call_signature(public[name], rule)
        if 'manual' in rule:
            manual = rule['manual']
            if (not isinstance(manual, list) or not manual
                    or len(set(manual)) != len(manual)
                    or set(manual) - {'python', 'native', 'wasm'} or 'frontend' in rule):
                raise ValueError(f'{name}: invalid or duplicate handwritten adapter')
        ret, args = signature(public[name])
        params = {a['name']: a['type']['qualType'] for a in args}
        arrays = rule.get('arrays', {})
        if not isinstance(arrays, dict):
            raise ValueError(f'{name}: expected named array contracts')
        for arg, array in arrays.items():
            if arg not in params:
                raise ValueError(f'{name}: unknown array argument {arg}')
            if not isinstance(array, dict) or array.keys() - {'length', 'width', 'direction', 'limit'}:
                raise ValueError(f'{name}: invalid array contract {arg}')
            count = array.get('length')
            if isinstance(count, str):
                if params.get(count) not in {'int', 'int64_t', 'uint64_t'}:
                    raise ValueError(f'{name}: invalid array count argument {count}')
            elif type(count) is not int or count >= 0:
                raise ValueError(f'{name}: array length must name a count or be a negative fixed count')
            width = array.get('width')
            direction = array.get('direction')
            allowed = {'int64_t *': {(8, 'in'), (16, 'in')},
                       'const int64_t *': {(8, 'in'), (16, 'in')},
                       'int64_t (*)[2]': {(16, 'in')}, 'const int *': {(4, 'in')},
                       'PolyTensor **': {(4, 'in'), (4, 'out')}}
            if (type(width) is not int or direction not in {'in', 'out'}
                    or (width, direction) not in allowed.get(params[arg], set())
                    or (direction == 'out' and params[arg] != 'PolyTensor **')
                    or type(array.get('limit', 1)) is not int or array.get('limit', 1) < 1):
                raise ValueError(f'{name}: unsupported array contract {arg}')
        if 'frontend' in rule:
            kind = rule['frontend']
            pair, optional = kind == 'pair', kind == 'optional'
            if kind not in {'direct', 'pair', 'optional', 'status', 'owned_string'}:
                raise ValueError(f'{name}: unknown frontend adapter {kind}')
            outputs = 2 if pair else 1 if optional else 0
            inputs = args[:-outputs] if outputs else args
            if (outputs and (len(args) < outputs or any(a['type']['qualType'] != 'PolyTensor **'
                                                       for a in args[-outputs:]))):
                raise ValueError(f'{name}: unsupported owned-output signature')
            expected_ret = {'char *'} if kind == 'owned_string' else {'int'} if pair or kind == 'status' else {'PolyTensor *'} if optional else {'PolyTensor *', 'PolyUOp *'}
            if ret not in expected_ret:
                raise ValueError(f'{name}: unsupported frontend return {ret}')
            for arg in inputs:
                t, n = arg['type']['qualType'], arg['name']
                if (t.removeprefix('const ') in OPAQUE | SCALARS
                        or (kind in {'status', 'owned_string'} and t == 'const char *')):
                    continue
                if (t in {'int64_t *', 'const int64_t *', 'int64_t (*)[2]'}
                        and n in arrays and isinstance(arrays[n]['length'], str)
                        and arrays[n]['direction'] == 'in'):
                    continue
                raise ValueError(f'{name}: unsupported frontend argument {n}: {t}')
        if 'extension' in rule:
            ext = rule['extension']
            if (not isinstance(ext, dict) or set(ext) - {'failure', 'ownership', 'cleanup', 'native', 'strings'}
                    or not {'failure', 'ownership', 'cleanup'} <= set(ext)
                    or ('native' in ext and ext['native'] is not True)
                    or type(ext['failure']) is not int
                    or ext['ownership'] not in {'owned', 'borrowed', 'value'}
                    or type(ext['cleanup']) is not bool):
                raise ValueError(f'{name}: invalid extension contract')
            strings = ext.get('strings', [])
            if (not isinstance(strings, list) or any(not isinstance(s, str) for s in strings)
                    or len(set(strings)) != len(strings)
                    or any(params.get(s) != 'const char *' or s in arrays for s in strings)
                    or (strings and ext.get('native'))):
                raise ValueError(f'{name}: invalid extension string contract')
            if ext.get('native'):
                # Same-address-space calls use C records directly. Never marshal
                # these pointers across the separate author/core Wasm heaps.
                continue
            if ret not in OPAQUE | SCALARS | {'void'}:
                raise ValueError(f'{name}: unsupported extension return {ret}')
            for arg in args:
                if arg['type']['qualType'].removeprefix('const ') not in OPAQUE | SCALARS and arg['name'] not in arrays and arg['name'] not in strings:
                    raise ValueError(f'{name}: unsupported extension argument {arg["name"]}')


def load_rules(decls):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate binding key: {key}')
            result[key] = value
        return result
    rules = json.loads((ROOT / 'scripts/binding_rules.json').read_text(encoding='utf-8'),
                       object_pairs_hook=unique_object)
    validate_rules(decls, rules)
    return rules


def native_special_exports(rules):
    return ('/* Generated by scripts/extension_bindings.py --core. */\n'
            + ''.join(f'DECLARE_NAPI_METHOD("{name}", napi_{name}),\n'
                      for name, rule in sorted(rules.items())
                      if 'native' in rule.get('manual', []) + rule.get('host', [])))


def core(check=False, output=None):
    if INSTALLED:
        raise ValueError('core binding generation requires the repository')
    decls = core_declarations()
    rules = load_rules(decls)
    result = {}
    for name, d in sorted(decls.items()):
        rule = rules.get(name, {})
        if 'extension' not in rule:
            continue
        ret, args = signature(d)
        arrays = []
        names = [v["name"] for v in args]
        for i, v in enumerate(args if not rule['extension'].get('native') else []):
            t = v["type"]["qualType"]
            if t.removeprefix("const ") in OPAQUE or t in SCALARS or v['name'] in rule['extension'].get('strings', []):
                continue
            annotation = rule['arrays'][v['name']]
            count, width, direction = (annotation[k] for k in ('length', 'width', 'direction'))
            arrays.append([i, names.index(count) if isinstance(count, str) else count, width, direction])
        result[name] = {
            "return": ret,
            "args": [[v["type"]["qualType"], v["name"]] for v in args],
            "arrays": arrays,
            **{key: rule['extension'][key] for key in ('failure', 'ownership', 'cleanup')},
            **({'native': True} if rule['extension'].get('native') else {}),
            **({'strings': [names.index(s) for s in rule['extension']['strings']]}
               if rule['extension'].get('strings') else {}),
        }
    text = json.dumps(result, indent=2) + "\n"
    files = {
        dest: text
        for dest in ("js/src/extension_api.json", "py/polygrad/extension_api.json")
    }
    # Resolve optional product APIs only when their objects are linked. Derive
    # module membership from owning headers, not copied function/signature lists.
    core_names = declarations('#include "polygrad.h"\n#include "frontend.h"\n#include "nn/nn.h"\n')
    optional = sorted(result.keys() - core_names.keys())
    model_names = declarations('#include "model.h"\n')
    if set(optional) - model_names.keys():
        raise ValueError(f'extension exports have no reviewed module owner: {set(optional) - model_names.keys()}')
    files["src/frontend_exports.h"] = (
        "/* Generated by scripts/extension_bindings.py --core. */\n"
        + "#ifdef POLY_EXTENSION_DECLARE_OPTIONAL\n"
        + "".join(f"#pragma weak {name}\n" for name in optional)
        + "#else\n"
        + "".join(f"POLY_EXTENSION_SYMBOL({name})\n" for name in result)
        + "#endif\n"
    )
    files.update(frontend_bindings(decls, rules))
    files['js/src/core/generated_device_calls.js'] = device_call_bindings(decls, rules)
    files['js/napi_special_exports.h'] = native_special_exports(rules)
    ast = declaration_ast(core_source())
    def has_enum(node):
        return node['kind'] == 'EnumType' or any(has_enum(v) for v in node.get('inner', []))
    enums = {v['name'] for v in ast['inner'] if v['kind'] == 'TypedefDecl' and has_enum(v)}
    files['py/polygrad/_generated_special_ffi.py'] = python_special_bindings(decls, rules, enums)
    output = ROOT if output is None else output
    for dest, text in files.items():
        target = output / dest
        if dest.endswith('.h') and dest.startswith('js/'):
            text = subprocess.check_output(['clang-format-14'], input=text.encode()).decode()
        if check:
            if target.read_text(encoding='utf-8') != text:
                raise ValueError(f"stale metadata: {dest}; run make extension-bindings")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            # These headers are direct native/Wasm build prerequisites.
            # A no-op generation must not invalidate every compiled artifact.
            if not target.exists() or target.read_text(encoding='utf-8') != text:
                target.write_text(text, encoding='utf-8')
    print(
        f"{'checked' if check else 'generated'} {len(result)} construction API signatures"
    )


def consumer(a):
    if not 0 <= a.inputs <= 4096 or not 1 <= a.outputs <= 4096:
        raise ValueError("invalid input/output counts")
    api = json.loads((SDK / 'extension_api.json' if INSTALLED else ROOT / 'js/src/extension_api.json').read_text())
    source = a.source.resolve()
    decls = declarations(f'#include "{source}"\n')
    methods, method_roots = [], []
    if getattr(a, 'methods', None):
        spec = importlib.util.spec_from_file_location('polygrad_extension_methods', SDK / 'extension_methods.py')
        extension_methods = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension_methods)
        methods, method_roots = extension_methods.describe(json.loads(a.methods.read_text()), decls, signature)
    ret, params = signature(decls[a.entry])
    types = [p["type"]["qualType"] for p in params]
    if (
        ret != "int"
        or types[:2] != ["PolyCtx *", "PolyTensor **"]
        or types[-1] != "PolyTensor **"
        or any(t not in ("int", "double") for t in types[2:-1])
    ):
        raise ValueError(
            "entry must return int: (PolyCtx*, PolyTensor** inputs, int/double scalars..., PolyTensor** owned_outputs)"
        )
    # Clang has expanded macros and helper headers. Follow reachable function
    # bodies instead of scanning spelling, which misses macro calls and includes
    # dead helpers. The public declarations remain the sole signature source.
    imports, visited = {"poly_extension_abi_version", "poly_tensor_release"}, set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        bodies = [v for v in decls.get(name, {}).get("inner", [])
                  if v["kind"] == "CompoundStmt"]
        if not bodies:
            if name.startswith("poly_"):
                imports.add(name)
            return
        def walk(node):
            ref = node.get("referencedDecl", {})
            if ref.get("kind") == "FunctionDecl":
                visit(ref["name"])
            for child in node.get("inner", []):
                walk(child)
        for body in bodies:
            walk(body)

    visit(a.entry)
    for root in method_roots:
        visit(root)
    if methods:
        imports.add('poly_extension_supports_abi')
    names = sorted(imports)
    if set(names) - api.keys():
        raise ValueError(f"unsupported imports: {set(names) - api.keys()}")
    manifest = {
        "construction_abi": 2 if methods else construction_abi(),
        "entry": a.entry,
        "inputs": a.inputs,
        "outputs": a.outputs,
        "scalars": types[2:-1],
        "imports": names,
    }
    if methods:
        manifest['methods'] = methods
    manifest_text = json.dumps(manifest)
    # Includes the terminating NUL; match the bounded Wasm loader read.
    if len(manifest_text.encode('utf-8')) >= 65536:
        raise ValueError('extension manifest must fit in 65536 bytes including NUL')
    c = [
        "/* Generated from Polygrad public declarations. */",
        '#include "polygrad.h"',
        '#include "frontend.h"',
        '#include "model.h"',
        "#include <stddef.h>",
        "#include <math.h>",
        *EXPORT_DEFINITIONS,
    ]
    for name in names:
        d = api[name]
        r = d["return"]
        args = d["args"]
        # A pointer-to-array name belongs inside its declarator, not after it.
        ps = ", ".join(t.replace("(*)", f"(*{n})") if "(*)" in t else t + " " + n
                       for t, n in args) or "void"
        call = ", ".join(n for _, n in args)
        c += [
            "#ifndef __EMSCRIPTEN__",
            f"static {r} (*fn_{name})({ps});",
            f"POLY_EXTENSION_LOCAL {r} {name}({ps}) {{ "
            + ("return " if r != "void" else "")
            + f"fn_{name}({call}); }}",
            "#endif",
        ]
    c += [
        "#ifndef __EMSCRIPTEN__",
        "static int extension_bound;",
        "EXT int poly_extension_bind(PolyProcResolver resolve) {",
        "if (!resolve) return 0;",
        'int (*abi)(void) = (int (*)(void))resolve("poly_extension_abi_version");',
        "if (!abi || abi() != POLYGRAD_EXTENSION_ABI_VERSION) return 0;",
    ]
    if methods:
        c += ['int (*supports)(int) = (int (*)(int))resolve("poly_extension_supports_abi");',
              'if (!supports || !supports(2)) return 0;']
    # Validate all addresses before publishing any: a failed bind is atomic.
    for name in names:
        c += [
            f'PolyProc addr_{name}=resolve("{name}");',
            f"if (!addr_{name} || (fn_{name} && (PolyProc)fn_{name}!=addr_{name})) return 0;",
        ]
    for name in names:
        d = api[name]
        ts = ", ".join(t for t, _ in d["args"]) or "void"
        c.append(f"fn_{name}=({d['return']} (*)({ts}))addr_{name};")
    c += [
        "extension_bound=1; return 1;}",
        "#endif",
        f"EXT const char *poly_extension_manifest(void) {{ return {json.dumps(manifest_text)}; }}",
        "EXT int poly_extension_abi(void) { return POLYGRAD_ABI_VERSION; }",
        f"extern int {a.entry}("
        + ", ".join(t + " " + p["name"] for t, p in zip(types, params))
        + ");",
        "EXT int poly_extension_build(PolyCtx *ctx, PolyTensor **inputs, const double *values, PolyTensor **outputs) {",
        "#ifndef __EMSCRIPTEN__",
        "if (!extension_bound) return 0;",
        "#endif",
    ]
    for i, t in enumerate(types[2:-1]):
        c.append(f"if (!isfinite(values[{i}])) return 0;")
        if t == "int":
            c.append(
                f"if (values[{i}] < -2147483648.0 || values[{i}] > 2147483647.0 || values[{i}] != (int)values[{i}]) return 0;"
            )
    args = (
        ["ctx", "inputs"]
        + [f"({t})values[{i}]" for i, t in enumerate(types[2:-1])]
        + ["outputs"]
    )
    c += [
        f"return {a.entry}(" + ", ".join(args) + ");",
        "}",
    ]
    if methods:
        c += [extension_methods.PROTOTYPES, '#define POLY_EXTENSION_METHODS 1']
    c += [
        "#ifdef POLY_EXTENSION_NAPI",
        f"#define POLY_EXTENSION_NINPUTS {a.inputs}",
        f"#define POLY_EXTENSION_NOUTPUTS {a.outputs}",
        f"#define POLY_EXTENSION_NSCALARS {len(types) - 3}",
        '#include "extension_native.h"',
        "#endif",
    ]
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "extension.c").write_text("\n".join(c) + "\n")
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if methods:
        (a.output / 'methods.c').write_text(extension_methods.generate(source, methods, EXPORT_DEFINITIONS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--core", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--source", type=Path)
    p.add_argument("--entry")
    p.add_argument("--methods", type=Path, help="consumer method ownership rules; declarations own signatures")
    p.add_argument("--inputs", type=int)
    p.add_argument("--outputs", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--sdk", type=Path, help="prepare an installed author SDK")
    p.add_argument("--include", action="store_true", help="print the author include directory")
    a = p.parse_args()
    if a.include:
        print(include_dir())
    elif a.sdk:
        sdk(a.sdk)
    elif a.core:
        core(a.check, a.output)
    else:
        if not all(
            v is not None for v in (a.source, a.entry, a.inputs, a.outputs, a.output)
        ):
            p.error("provide source, entry, inputs, outputs and output")
        consumer(a)
