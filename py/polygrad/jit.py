"""JIT capture/replay for raw Tensor functions."""

import functools
import ctypes
import time

from . import _ffi
from .tensor import BoundVariable, Tensor, Variable
from polygrad.uop.ops import UOp


capturing = []


class JitError(RuntimeError):
    pass


def _ret_tensors(ret):
    if ret is None:
        return []
    if isinstance(ret, Tensor):
        return [ret]
    if isinstance(ret, (tuple, list)):
        out = []
        for item in ret:
            out.extend(_ret_tensors(item))
        return out
    if isinstance(ret, dict):
        out = []
        for item in ret.values():
            out.extend(_ret_tensors(item))
        return out
    raise JitError(f'JIT return contains non-Tensor value of type {type(ret).__name__}')


def _realize_return(ret):
    tensors = _ret_tensors(ret)
    if tensors:
        tensors[0].realize(*tensors[1:])
    return tensors


def _input_tensors(args, kwargs):
    inputs = []
    names = []

    def add(name, value, dedup):
        if isinstance(value, Tensor) and (not dedup or not any(value is x for x in inputs)):
            names.append(name)
            inputs.append(value)

    for i, arg in enumerate(args):
        add(i, arg, False)
    for name in sorted(kwargs):
        add(name, kwargs[name], False)

    for value in list(args) + [kwargs[k] for k in sorted(kwargs)]:
        if isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, (tuple, list)):
            iterable = value
        else:
            iterable = ()
        for item in iterable:
            add(f'container:{len(inputs)}', item, True)

    return names, inputs


def _bound_var_items(args, kwargs):
    values = list(args) + [kwargs[k] for k in sorted(kwargs)]
    for value in list(values):
        if isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (tuple, list)):
            values.extend(value)

    out = []
    seen = {}
    for value in values:
        if isinstance(value, Variable):
            raise JitError('JIT variables must be bound')
        if not isinstance(value, BoundVariable):
            continue
        raw = int(value.variable.uop.raw)
        val = int(value.value)
        if raw in seen:
            if seen[raw] != val:
                raise JitError('conflicting JIT variable bindings')
            continue
        seen[raw] = val
        out.append((value.variable.uop.raw, val))
    return out


def _raw_int(raw):
    return 0 if raw is None else int(raw)


def _merge_var_bindings(*binding_lists):
    out = []
    seen = {}
    for bindings in binding_lists:
        for raw, value in bindings:
            key = _raw_int(raw)
            val = int(value)
            if key in seen:
                if seen[key] != val:
                    raise JitError('conflicting JIT variable bindings')
                continue
            seen[key] = val
            out.append((raw, val))
    return out


def _var_binding_array(bindings):
    n = len(bindings)
    arr = (_ffi.PolyVarBinding * max(1, n))()
    for i, (raw, value) in enumerate(bindings):
        arr[i].var = raw
        arr[i].value = value
    return arr, n


def _check_duplicate_input_buffers(inputs):
    seen = set()
    lib = _ffi._lib
    base_ops = {
        'RESHAPE', 'EXPAND', 'PERMUTE', 'PAD', 'SHRINK', 'FLIP', 'MULTI', 'DETACH',
    }
    for t in inputs:
        uop = t.uop
        while uop and uop.op_name in base_ops and uop.src:
            uop = uop.src[0]
        buf = lib.poly_uop_get_buffer_identity(uop.raw) if uop else None
        key = int(buf) if buf else 0
        if key == 0:
            raise JitError('JIT inputs must be real buffers')
        if key in seen:
            raise JitError('duplicate inputs to JIT')
        seen.add(key)


def _shape_key_and_bindings(shape):
    key = []
    bindings = []
    lib = _ffi._lib
    for dim in shape:
        if isinstance(dim, int):
            key.append(('i', int(dim)))
            continue
        if not isinstance(dim, UOp) or not dim.raw:
            raise JitError(f'unsupported symbolic shape dimension {dim!r}')
        var = lib.poly_uop_unbind_var(dim.raw)
        if var:
            key.append(('v', _raw_int(var)))
            value = ctypes.c_int64()
            if lib.poly_uop_bind_value(dim.raw, ctypes.byref(value)) == 0:
                bindings.append((var, int(value.value)))
        else:
            key.append(('u', _raw_int(dim.raw)))
    return tuple(key), bindings


def _input_shape(t):
    # Pinned TinyJit keys inputs from the current Tensor.uop. Polygrad's
    # current/parity root is physical; logical is export/re-placement only.
    raw = _ffi._lib.poly_tensor_uop(t._tensor)
    from .tensor import _shape_from_uop
    return _shape_from_uop(t._ctx, raw)


def _input_info(names, inputs, explicit_var_bindings):
    tensor_info = []
    shape_bindings = []
    for t in inputs:
        shape_key, binds = _shape_key_and_bindings(_input_shape(t))
        shape_bindings.extend(binds)
        tensor_info.append((shape_key, t.dtype, t.device))
    var_bindings = _merge_var_bindings(explicit_var_bindings, shape_bindings)
    signature = (
        tuple(names),
        tuple(tensor_info),
        tuple(_raw_int(raw) for raw, _ in var_bindings),
    )
    return signature, var_bindings


def _tensor_array(inputs):
    n = len(inputs)
    arr = (_ffi._ptr * max(1, n))()
    for i, t in enumerate(inputs):
        arr[i] = t._tensor
    return arr, n


class Jit:
    """Tinygrad-style capture/replay wrapper for Tensor functions.

    First call runs normally, second call captures returned tensor realization,
    and later calls replay the captured schedules with current input BUFFERs.
    """

    def __init__(self, fxn, *, prune=False):
        if fxn is None:
            raise TypeError('Jit requires a function')
        functools.update_wrapper(self, fxn)
        self.fxn = fxn
        self.prune = bool(prune)
        self.cnt = 0
        self.captured = False
        self.ret = None
        self.signature = None
        self._jit = None
        self.call_count = 0
        self.replay_count = 0
        self.last_call_ms = 0.0

    def reset(self):
        if self._jit:
            _ffi._lib.poly_jit_free(self._jit)
        self._jit = None
        self.cnt = 0
        self.captured = False
        self.ret = None
        self.signature = None
        self.call_count = 0
        self.replay_count = 0
        self.last_call_ms = 0.0

    @property
    def schedule_count(self):
        if not self._jit:
            return 0
        return int(_ffi._lib.poly_jit_schedule_count(self._jit))

    def stats(self):
        return {
            'captured': self.captured,
            'prune': self.prune,
            'call_count': self.call_count,
            'replay_count': self.replay_count,
            'last_call_ms': self.last_call_ms,
            'schedule_count': self.schedule_count,
        }

    def __del__(self):
        try:
            if self._jit:
                _ffi._lib.poly_jit_free(self._jit)
        except Exception:
            pass

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        replayed = False
        names, inputs = _input_tensors(args, kwargs)
        explicit_var_bindings = _bound_var_items(args, kwargs)
        for t in inputs:
            # tinygrad/engine/jit.py:233-235 realizes only inputs whose
            # recursive UOp base is not allocated. Polygrad also needs one
            # realization when its requested-device physical root is absent;
            # an allocated logical HOST base is not placement.
            if t.uop_physical is None or not t.uop.is_realized:
                t.realize()
        _check_duplicate_input_buffers(inputs)
        sig, var_bindings = _input_info(names, inputs, explicit_var_bindings)

        if self.cnt == 0:
            ret = self.fxn(*args, **kwargs)
            _realize_return(ret)
        elif self.cnt == 1:
            # Match tinygrad/engine/jit.py:278-284: reject nested capture
            # before allocating or replacing this wrapper's private C JIT.
            if capturing:
                raise RuntimeError(
                    f'having TinyJit inside another TinyJit is not supported '
                    f'{len(capturing)=} {capturing=}'
                )
            if inputs:
                ctx = inputs[0]._ctx
                if any(t._ctx != ctx for t in inputs):
                    raise JitError('JIT inputs must share one PolyCtx')
            else:
                raise JitError('JIT requires at least one Tensor input')

            self._jit = _ffi._lib.poly_jit_new(ctx)
            if not self._jit:
                raise JitError('poly_jit_new failed')
            if _ffi._lib.poly_jit_set_prune(self._jit, self.prune) != 0:
                _ffi._lib.poly_jit_free(self._jit)
                self._jit = None
                raise JitError('poly_jit_set_prune failed')
            arr, n = _tensor_array(inputs)
            if _ffi._lib.poly_jit_begin_capture(self._jit, arr, n) != 0:
                _ffi._lib.poly_jit_free(self._jit)
                self._jit = None
                raise JitError('poly_jit_begin_capture failed')
            capturing.append(self)
            try:
                ret = self.fxn(*args, **kwargs)
                _realize_return(ret)
                if _ffi._lib.poly_jit_end_capture(self._jit) != 0:
                    raise JitError("didn't JIT anything")
            except Exception:
                _ffi._lib.poly_jit_cancel_capture(self._jit)
                raise
            finally:
                capturing.clear()
            self.ret = ret
            self.signature = sig
            self.captured = True
        else:
            if not self.captured or not self._jit:
                raise JitError('JIT has not captured')
            if sig != self.signature:
                raise JitError(f'args mismatch in JIT: expected {self.signature}, got {sig}')
            arr, n = _tensor_array(inputs)
            var_arr, n_vars = _var_binding_array(var_bindings)
            if _ffi._lib.poly_jit_run_with_vars(self._jit, arr, n, var_arr, n_vars) != 0:
                raise JitError('poly_jit_run failed')
            ret = self.ret
            replayed = True

        self.cnt += 1
        self.call_count += 1
        if replayed:
            self.replay_count += 1
        self.last_call_ms = (time.perf_counter() - start) * 1000.0
        return ret


def jit(fxn=None, *, prune=False):
    if fxn is None:
        return lambda f: Jit(f, prune=prune)
    return Jit(fxn, prune=prune)


class CompiledCallable:
    """Explicit wrapper around the same tinygrad-style JIT capture/replay path.

    Construction performs the normal first run and second capture run. Later
    calls replay the captured schedules through the wrapped Jit object.
    """

    def __init__(self, jit_obj, compile_ms, input_count):
        self._jit = jit_obj
        self.compile_ms = float(compile_ms)
        self.input_count = int(input_count)
        self.last_run_ms = 0.0
        self.run_count = 0
        self.disposed = False

    @property
    def schedule_count(self):
        return 0 if self.disposed else self._jit.schedule_count

    def stats(self):
        return {
            'captured': not self.disposed and self._jit.captured,
            'capture_runs': 2,
            'compile_ms': self.compile_ms,
            'last_run_ms': self.last_run_ms,
            'run_count': self.run_count,
            'call_count': self.run_count,
            'schedule_count': self.schedule_count,
            'input_count': self.input_count,
        }

    def run(self, inputs):
        if self.disposed:
            raise JitError('compiled callable has been disposed')
        if isinstance(inputs, Tensor):
            args = (inputs,)
        elif isinstance(inputs, (tuple, list)):
            args = tuple(inputs)
        else:
            raise TypeError('CompiledCallable.run requires a Tensor or sequence of Tensors')

        start = time.perf_counter()
        ret = self._jit(*args)
        self.last_run_ms = (time.perf_counter() - start) * 1000.0
        self.run_count += 1
        return ret

    def __call__(self, *inputs):
        return self.run(inputs)

    def dispose(self):
        if self.disposed:
            return
        self._jit.reset()
        self.disposed = True


def compile(fxn, sample_inputs, *, prune=False):
    """Warm and capture a Jit function, returning an explicit replay wrapper."""
    if not callable(fxn):
        raise TypeError('compile requires a function')
    if isinstance(sample_inputs, Tensor):
        args = (sample_inputs,)
    elif isinstance(sample_inputs, (tuple, list)):
        args = tuple(sample_inputs)
    else:
        raise TypeError('compile requires a Tensor or sequence of Tensors')

    j = Jit(fxn, prune=prune)
    start = time.perf_counter()
    j(*args)
    j(*args)
    compile_ms = (time.perf_counter() - start) * 1000.0
    if not j.captured or j.schedule_count <= 0:
        j.reset()
        raise JitError("didn't JIT anything")
    return CompiledCallable(j, compile_ms, len(args))
