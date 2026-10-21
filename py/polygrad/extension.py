"""Managed C authors attached to an existing Runtime."""

import ctypes as c
import json
import math
import weakref
from types import SimpleNamespace
from pathlib import Path
from . import _ffi
from .tensor import _device_name_from_id


class ExtensionObject:
    """Opaque consumer object, scoped to its extension/runtime/generation."""
    def __init__(self, extension, owner):
        self._extension, self._owner = extension, owner
        self._finalizer = weakref.finalize(self, extension._release_object, owner)

    def dispose(self):
        self._finalizer()


class Extension:
    def __init__(self, runtime, path):
        runtime._check_live()
        self._rt, self._closed = runtime, False
        self._lib = c.CDLL(str(path))
        lib, core = self._lib, _ffi.get_lib()
        lib.poly_extension_abi.restype = c.c_int
        lib.poly_extension_manifest.restype = c.c_char_p
        self._spec = json.loads(lib.poly_extension_manifest())
        if 'construction_abi' in self._spec:
            version = self._spec['construction_abi']
            if (type(version) is not int or version not in (1, 2) or
                    not (core.poly_extension_abi_version() == 1 if version == 1 else
                         hasattr(core, 'poly_extension_supports_abi') and core.poly_extension_supports_abi(version))):
                raise RuntimeError("extension construction ABI mismatch")
        elif lib.poly_extension_abi() != core.poly_abi_version():
            # Existing published authors use the original core-ABI handshake.
            raise RuntimeError("extension ABI mismatch")
        api = json.loads(Path(__file__).with_name("extension_api.json").read_text())
        s = self._spec
        if not (
            isinstance(s["inputs"], int)
            and 0 <= s["inputs"] <= 4096
            and isinstance(s["outputs"], int)
            and 1 <= s["outputs"] <= 4096
            and len(s["scalars"]) <= 4096
            and all(t in ("int", "double") for t in s["scalars"])
            and isinstance(s["imports"], list)
            and all(n in api for n in s["imports"])
        ):
            raise ValueError("unsupported extension manifest")
        lib.poly_extension_bind.argtypes = [c.c_void_p]
        lib.poly_extension_bind.restype = c.c_int
        if not lib.poly_extension_bind(c.cast(core.poly_get_proc_address, c.c_void_p)):
            raise RuntimeError("extension symbol/ABI mismatch")
        lib.poly_extension_build.argtypes = [
            c.c_void_p,
            c.POINTER(c.c_void_p),
            c.POINTER(c.c_double),
            c.POINTER(c.c_void_p),
        ]
        lib.poly_extension_build.restype = c.c_int
        self._objects, self._generation = {}, 0
        methods = s.get('methods')
        if s.get('construction_abi') == 2:
            if not isinstance(methods, list) or not methods or len(methods) > 256:
                raise ValueError('extension ABI 2 requires methods')
        elif methods is not None:
            raise ValueError('extension methods require ABI 2')
        if methods:
            self._bind_methods(methods)
            if not hasattr(runtime, '_extensions'): runtime._extensions = []
            runtime._extensions.append(self)

    def _bind_methods(self, methods):
        objects, names = set(), set()
        for m in methods:
            if (not isinstance(m, dict) or not isinstance(m.get('name'), str) or not m['name'] or
                    m['name'] in names or not isinstance(m.get('args'), list) or len(m['args']) > 64 or
                    not isinstance(m.get('result'), dict) or
                    m['result'].get('kind') not in ('object', 'tensor', 'string', 'value', 'status', 'void')):
                raise ValueError('invalid extension method manifest')
            names.add(m['name'])
            if m['result']['kind'] == 'object':
                if not isinstance(m['result'].get('object'), str) or not isinstance(m['result'].get('release'), str):
                    raise ValueError('invalid extension object ownership')
                objects.add(m['result']['object'])
        for m in methods:
            for a in m['args']:
                if (a.get('kind') not in ('context','object','tensor','model','string','int','double') or
                        (a['kind'] == 'object' and a.get('object') not in objects)):
                    raise ValueError('invalid extension method argument')
        lib = self._lib
        lib.poly_extension_invoke.argtypes = [c.c_int, c.c_void_p, c.POINTER(c.c_void_p),
            c.POINTER(c.c_char_p), c.POINTER(c.c_double), c.POINTER(c.c_void_p),
            c.POINTER(c.c_void_p), c.POINTER(c.c_double)]
        lib.poly_extension_invoke.restype = c.c_int
        lib.poly_extension_release.argtypes = [c.c_int, c.c_void_p]
        lib.poly_extension_release.restype = None
        # Consumer allocations belong to its allocator, not a second Python ABI.
        lib.poly_extension_free.argtypes = [c.c_void_p]
        lib.poly_extension_free.restype = None
        self.methods = SimpleNamespace(**{m['name']: self._method(i, m) for i, m in enumerate(methods)})

    def _release_object(self, owner):
        if not owner['active']: return
        owner['active'] = False
        self._objects.pop(owner['generation'], None)
        if not self._rt._disposed:
            self._lib.poly_extension_release(owner['method'], owner['handle'])

    def _method(self, index, spec):
        def call(*args):
            self._rt._check_live()
            if self._closed: raise RuntimeError('extension is disposed')
            if len(args) != sum(a['kind'] != 'context' for a in spec['args']):
                raise ValueError('extension method argument count mismatch')
            n, it = len(spec['args']), iter(args)
            handles, strings, values = (c.c_void_p * n)(), (c.c_char_p * n)(), (c.c_double * n)()
            for i, a in enumerate(spec['args']):
                kind = a['kind']
                if kind == 'context': continue
                x = next(it)
                if kind == 'object':
                    if (not isinstance(x, ExtensionObject) or x._extension is not self or
                            not x._owner['active'] or x._owner['type'] != a['object'] or
                            self._objects.get(x._owner['generation']) is not x._owner):
                        raise ValueError('extension method requires a live object from this extension instance')
                    handles[i] = x._owner['handle']
                elif kind == 'tensor':
                    if not getattr(x, '_tensor', None) or x._ctx != self._rt._ctx:
                        raise ValueError('extension method requires a live Tensor from this runtime')
                    handles[i] = x._tensor
                elif kind == 'model':
                    from .model import Model
                    from .tensor import _ptr_value
                    if (not isinstance(x, Model) or not x._ptr or
                            _ptr_value(x._ctx) != _ptr_value(self._rt._ctx)):
                        raise ValueError('extension method requires a live Model from this runtime')
                    handles[i] = x._ptr
                elif kind == 'string':
                    if not isinstance(x, str) or '\0' in x: raise ValueError('extension string must be UTF-8 without NUL')
                    strings[i] = x.encode('utf-8')
                else:
                    if (not isinstance(x, (int, float)) or not math.isfinite(x) or
                            (kind == 'int' and (x != int(x) or not -2147483648 <= x <= 2147483647))):
                        raise ValueError('invalid extension scalar')
                    values[i] = x
            out, text, value = c.c_void_p(), c.c_void_p(), c.c_double()
            try:
                rc = self._lib.poly_extension_invoke(index, self._rt._ctx, handles, strings, values,
                    c.byref(out), c.byref(text), c.byref(value))
                if rc: raise RuntimeError(f"extension method {spec['name']} failed ({rc})")
                kind = spec['result']['kind']
                if kind == 'object':
                    self._generation += 1
                    owner = dict(handle=out.value, method=index, type=spec['result']['object'],
                                 generation=self._generation, active=True)
                    obj = ExtensionObject(self, owner)
                    self._objects[self._generation] = owner
                    out.value = None
                    return obj
                if kind == 'tensor':
                    device = _device_name_from_id(_ffi.get_lib().poly_tensor_device(out))
                    if device in ('AUTO', 'DISK'): device = self._rt._device
                    result = self._rt.Tensor(_tensor=out.value, _ctx=self._rt._ctx, _device=device)
                    out.value = None
                    return result
                if kind == 'string': return c.string_at(text).decode('utf-8')
                if kind == 'value': return int(value.value) if spec['result']['ctype'] == 'int' else value.value
            finally:
                if out.value: self._lib.poly_extension_release(index, out)
                if text.value: self._lib.poly_extension_free(text)
        return call

    def build(self, tensors=(), values=()):
        self._rt._check_live()
        if self._closed:
            raise RuntimeError("extension is disposed")
        s, rt = self._spec, self._rt
        if len(tensors) != s["inputs"] or len(values) != len(s["scalars"]):
            raise ValueError("extension argument count mismatch")
        for t in tensors:
            if not getattr(t, "_tensor", None) or t._ctx != rt._ctx:
                raise ValueError(
                    "extension inputs must be live Tensors from this runtime"
                )
        for v, t in zip(values, s["scalars"]):
            if (
                not isinstance(v, (int, float))
                or not math.isfinite(v)
                or (t == "int" and (v != int(v) or not -2147483648 <= v <= 2147483647))
            ):
                raise ValueError("invalid extension scalar")
        inputs = (c.c_void_p * len(tensors))(*(t._tensor for t in tensors))
        scalars = (c.c_double * len(values))(*values)
        outputs = (c.c_void_p * s["outputs"])()
        adopted = []
        try:
            if not self._lib.poly_extension_build(
                rt._ctx, inputs, scalars, outputs
            ) or not all(outputs):
                raise RuntimeError("extension construction failed")
            for i, h in enumerate(outputs):
                device = _device_name_from_id(_ffi.get_lib().poly_tensor_device(h))
                if device in ("AUTO", "DISK"):
                    device = rt._device
                adopted.append(rt.Tensor(_tensor=h, _ctx=rt._ctx, _device=device))
                outputs[i] = None
            return adopted
        except BaseException:
            for t in adopted:
                t.dispose()
            raise
        finally:
            for h in outputs:
                if h:
                    _ffi.get_lib().poly_tensor_release(h)

    def dispose(self):
        if self._closed: return
        for owner in reversed(list(self._objects.values())):
            self._release_object(owner)
        self._closed = True
        if hasattr(self._rt, '_extensions') and self in self._rt._extensions:
            self._rt._extensions.remove(self)
        self._lib = None
