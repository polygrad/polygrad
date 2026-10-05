"""Construction-only C extensions attached to an existing Runtime."""

import ctypes as c
import json
import math
from pathlib import Path
from . import _ffi
from .tensor import _device_name_from_id


class Extension:
    def __init__(self, runtime, path):
        runtime._check_live()
        self._rt, self._closed = runtime, False
        self._lib = c.CDLL(str(path))
        lib, core = self._lib, _ffi.get_lib()
        lib.poly_extension_abi.restype = c.c_int
        if lib.poly_extension_abi() != core.poly_abi_version():
            raise RuntimeError("extension ABI mismatch")
        lib.poly_extension_manifest.restype = c.c_char_p
        self._spec = json.loads(lib.poly_extension_manifest())
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
        self._closed = True
        self._lib = None
