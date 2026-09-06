"""PolyModel -- Python wrapper for the C PolyModel runtime.

Provides forward pass, training, and weight I/O for runnable/exportable
models created from IR bytes, bundles, or frontend tensor graphs. Named
architecture factories live in ``polygrad.models``.
"""

import ctypes
import ctypes.util
import math
import pathlib
import weakref
import numpy as np
from . import _ffi
from .device import _device_id

_get_lib = _ffi.get_lib
_live_models = weakref.WeakSet()


def _dispose_models_for_ctx(ctx):
    """Free borrowed-context Models before their PolyCtx is destroyed."""
    from .tensor import _ptr_value

    ctx_key = _ptr_value(ctx)
    for inst in list(_live_models):
        if inst._ptr and _ptr_value(inst._ctx) == ctx_key:
            inst.free()

_MODEL_DTYPE_NAMES = (
    'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
    'int64', 'uint64', 'float16', 'bfloat16', 'float32', 'float64',
)


def _model_dtype_name(dtype_id):
    lib = _get_lib()
    for name in _MODEL_DTYPE_NAMES:
        if lib.poly_dtype_id_by_name(name.encode('utf-8')) == dtype_id:
            return name
    raise RuntimeError(f'unsupported Model storage dtype id {dtype_id}')


def _storage_dtype(dtype_name):
    # NumPy has no native BF16; preserve the existing raw uint16 contract.
    return np.dtype('uint16' if dtype_name == 'bfloat16' else dtype_name)

# libc free for caller-frees byte arrays
_libc = ctypes.CDLL(ctypes.util.find_library('c'))
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]

# Role constants
ROLE_PARAM = 0
ROLE_INPUT = 1
ROLE_TARGET = 2
ROLE_OUTPUT = 3
ROLE_AUX = 4

_ROLE_IDS = {
    'param': ROLE_PARAM,
    'state': ROLE_PARAM,
    'input': ROLE_INPUT,
    'target': ROLE_TARGET,
    'output': ROLE_OUTPUT,
    'aux': ROLE_AUX,
}

# Optimizer constants
OPTIM_NONE = 0
OPTIM_SGD = 1
OPTIM_ADAM = 2
OPTIM_ADAMW = 3

EXPORT_WEIGHTS_PARAMS = 1
EXPORT_WEIGHTS_OPTIMIZER = 2
EXPORT_WEIGHTS_DEFAULT = EXPORT_WEIGHTS_PARAMS | EXPORT_WEIGHTS_OPTIMIZER
_BIND_F_FROZEN = 1 << 2


def _optimizer_kind(kind):
    if isinstance(kind, str):
        k = kind.lower()
        if k == 'sgd':
            return OPTIM_SGD
        if k == 'adam':
            return OPTIM_ADAM
        if k == 'adamw':
            return OPTIM_ADAMW
    return int(kind)


def _normalize_named_tensors(value, default_name):
    if value is None:
        return {}
    from .tensor import Tensor
    if isinstance(value, Tensor):
        return {default_name: value}
    return dict(value)


def _param_items(params):
    if params is None:
        return []
    if isinstance(params, dict):
        return list(params.items())
    return [(f'param_{i}', p) for i, p in enumerate(params)]


def _name_bytes(name):
    text = str(name)
    if not text:
        raise ValueError('Model binding names must be non-empty')
    if '\x00' in text:
        raise ValueError(f'Model binding name contains NUL: {text!r}')
    return text.encode('utf-8')


def _require_tensor(name, tensor):
    from .tensor import Tensor

    if not isinstance(tensor, Tensor):
        raise TypeError(f'{name!r} is not a Tensor')
    if not tensor._tensor:
        raise RuntimeError(f'{name!r} has no core PolyTensor')
    return tensor


def _role_id(role):
    if isinstance(role, str):
        key = role.lower()
        if key not in _ROLE_IDS:
            raise ValueError(f'unknown Model binding role: {role!r}')
        return _ROLE_IDS[key]
    return int(role)


def _binding_fields(binding):
    if isinstance(binding, dict):
        return (
            binding.get('name'),
            binding.get('role'),
            binding.get('tensor'),
            binding.get('flags', 0),
            binding.get('trainable'),
        )
    if len(binding) == 3:
        name, role, tensor = binding
        return name, role, tensor, 0, None
    if len(binding) == 4:
        name, role, tensor, flags = binding
        return name, role, tensor, flags, None
    raise TypeError('Model bindings must be dicts or (name, role, tensor[, flags]) tuples')


def _entry_fields(entry):
    if isinstance(entry, dict):
        return (
            entry.get('name'),
            entry.get('inputs', ()),
            entry.get('outputs', ()),
            entry.get('objective'),
            entry.get('flags', 0),
        )
    if len(entry) == 3:
        name, inputs, outputs = entry
        return name, inputs, outputs, None, 0
    if len(entry) == 4:
        name, inputs, outputs, objective = entry
        return name, inputs, outputs, objective, 0
    if len(entry) == 5:
        return entry
    raise TypeError(
        'Model entrypoints must be dicts or '
        '(name, inputs, outputs[, objective[, flags]]) tuples'
    )


def _module_fields(module):
    if isinstance(module, dict):
        return module.get('name'), module.get('inputs', ()), module.get('output')
    if len(module) == 3:
        return module
    raise TypeError(
        'Model modules must be dicts or (name, inputs, output) tuples'
    )


def _module_inputs(inputs):
    from .tensor import Tensor

    if inputs is None:
        return []
    if isinstance(inputs, Tensor):
        return [inputs]
    return list(inputs)


def _entry_name_list(names):
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    return list(names)


def _check_ctx(name, tensor, ctx, ctx_key):
    from .tensor import _ptr_value

    if _ptr_value(tensor._ctx) != ctx_key:
        raise ValueError(f'{name!r} belongs to another PolyCtx')
    return ctx


def _ensure_storage_binding(name, tensor):
    if not tensor.uop.has_buffer_identity():
        raise RuntimeError(f'{name!r} has no buffer identity')


def _entrypoint_spec(name, inputs, outputs, objective=None, flags=0, keepalive=None):
    keepalive = keepalive if keepalive is not None else []
    name_b = _name_bytes(name)
    keepalive.append(name_b)

    input_bs = [_name_bytes(n) for n in inputs]
    output_bs = [_name_bytes(n) for n in outputs]
    keepalive.extend(input_bs)
    keepalive.extend(output_bs)

    input_arr = None
    output_arr = None
    if input_bs:
        input_arr = (ctypes.c_char_p * len(input_bs))(*input_bs)
        keepalive.append(input_arr)
    if output_bs:
        output_arr = (ctypes.c_char_p * len(output_bs))(*output_bs)
        keepalive.append(output_arr)

    objective_b = _name_bytes(objective) if objective is not None else None
    if objective_b is not None:
        keepalive.append(objective_b)

    return _ffi.PolyEntrypointSpec(
        name_b,
        input_arr,
        len(input_bs),
        output_arr,
        len(output_bs),
        objective_b,
        int(flags),
    )


def _model_from_binding_specs(ctx, binding_rows, entry_rows, keepalive):
    bindings = (_ffi.PolyBindingSpec * len(binding_rows))(*binding_rows)
    entries = (_ffi.PolyEntrypointSpec * len(entry_rows))(*entry_rows)
    keepalive.extend([bindings, entries])
    err = _ffi.PolyModelError()
    ptr = _ffi._lib.poly_model_from_bindings(
        ctx,
        bindings,
        len(binding_rows),
        entries,
        len(entry_rows),
        None,
        ctypes.byref(err),
    )
    if not ptr:
        msg = bytes(err.message).split(b'\0', 1)[0].decode('utf-8', 'replace')
        func = err.func.decode('utf-8', 'replace') if err.func else 'poly_model_from_bindings'
        detail = f'{func}: {msg}' if msg else func
        raise RuntimeError(f'poly_model_from_bindings failed: {detail}')
    return Model(ptr, _ctx=ctx)


class Model:
    """C-owned state and callables with sealed topology and optional training.

    Author with ordinary Tensor code or ``trace``; overriding a Python method
    does not change the captured executable or synchronize authoring attributes.
    """

    def __init__(self, ptr=None, *, inputs=None, targets=None, state=None,
                 outputs=None, entrypoints=None, params=None, losses=None,
                 modules=None, _ctx=None):
        spec_args = (inputs, targets, state, outputs, entrypoints, params, losses, modules)
        has_spec = any(v is not None for v in spec_args)
        if has_spec:
            if ptr is not None:
                raise TypeError('Model handle cannot be combined with tensor bindings')
            built = Model.from_tensors(
                inputs=inputs,
                targets=targets,
                outputs=outputs,
                losses=losses,
                params=params,
                state=state,
                entrypoints=entrypoints,
                modules=modules,
            )
            self._ptr = built._ptr
            self._ctx = built._ctx
            built._ptr = None
            built._ctx = None
            if self._ctx:
                _live_models.add(self)
            return

        if not ptr:
            raise RuntimeError('Failed to create PolyModel (NULL pointer)')
        self._ptr = ptr
        self._ctx = _ctx
        if self._ctx:
            _live_models.add(self)

    def free(self):
        if self._ptr:
            _get_lib().poly_model_free(self._ptr)
            self._ptr = None
            self._ctx = None

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self.free()

    # ── Lifecycle ────────────────────────────────────────────────────

    @staticmethod
    def from_ir(ir_bytes, weights_bytes=None):
        """Create from IR binary + optional safetensors weights."""
        ir_buf = (ctypes.c_uint8 * len(ir_bytes)).from_buffer_copy(ir_bytes)
        w_buf = None
        w_len = 0
        if weights_bytes:
            w_buf = (ctypes.c_uint8 * len(weights_bytes)).from_buffer_copy(weights_bytes)
            w_len = len(weights_bytes)
        ptr = _get_lib().poly_model_from_ir(ir_buf, len(ir_bytes), w_buf, w_len)
        return Model(ptr)

    @staticmethod
    def from_program(program_bytes, weights_bytes=None):
        """Load an ABI/device-bound compiled program plus separate weights.

        Unlike :meth:`from_ir`, this does not retain portable logical IR and
        cannot be re-placed onto another device.
        """
        program_buf = (ctypes.c_uint8 * len(program_bytes)).from_buffer_copy(program_bytes)
        w_buf = None
        w_len = 0
        if weights_bytes:
            w_buf = (ctypes.c_uint8 * len(weights_bytes)).from_buffer_copy(weights_bytes)
            w_len = len(weights_bytes)
        ptr = _get_lib().poly_model_from_program(
            program_buf, len(program_bytes), w_buf, w_len)
        return Model(ptr)

    @staticmethod
    def from_bindings(bindings, entrypoints, *, modules=None):
        """Create an Model from explicit binding and entrypoint records.

        Bindings may be dicts with ``name``, ``role``, ``tensor``, and optional
        ``flags`` fields, or ``(name, role, tensor[, flags])`` tuples. Roles
        accept the C role ids or strings: ``input``, ``target``, ``state``/
        ``param``, ``output``, and ``aux``. Entrypoints may be dicts with
        ``name``, ``inputs``, ``outputs``, optional ``objective``/``flags``, or
        matching tuples.
        """
        from .tensor import _ptr_value

        bindings = list(bindings or ())
        entrypoints = list(entrypoints or ())
        if not bindings:
            raise ValueError('Model.from_bindings requires at least one binding')
        if not entrypoints:
            raise ValueError('Model.from_bindings requires at least one entrypoint')

        parsed = []
        for binding in bindings:
            name, role, tensor, flags, trainable = _binding_fields(binding)
            if name is None:
                raise ValueError('Model binding is missing a name')
            if role is None:
                raise ValueError(f'Model binding {name!r} is missing a role')
            tensor = _require_tensor(name, tensor)
            role = _role_id(role)
            if role == ROLE_PARAM and not (tensor.is_param if trainable is None else trainable):
                flags = int(flags) | _BIND_F_FROZEN
            parsed.append((name, role, tensor, int(flags)))

        ctx = parsed[0][2]._ctx
        ctx_key = _ptr_value(ctx)
        for name, role, tensor, _ in parsed:
            _check_ctx(name, tensor, ctx, ctx_key)
            if role in (ROLE_INPUT, ROLE_TARGET):
                _ensure_storage_binding(name, tensor)

        keepalive = []
        binding_rows = []
        for name, role, tensor, flags in parsed:
            name_b = _name_bytes(name)
            keepalive.append(name_b)
            binding_rows.append(_ffi.PolyBindingSpec(name_b, role, tensor._tensor, flags))

        entry_rows = []
        for entry in entrypoints:
            name, inputs, outputs, objective, flags = _entry_fields(entry)
            if name is None:
                raise ValueError('Model entrypoint is missing a name')
            entry_rows.append(_entrypoint_spec(
                name,
                _entry_name_list(inputs),
                _entry_name_list(outputs),
                objective=objective,
                flags=flags,
                keepalive=keepalive,
            ))

        inst = _model_from_binding_specs(ctx, binding_rows, entry_rows, keepalive)
        if modules is not None:
            try:
                inst.define_modules(modules)
            except Exception:
                inst.free()
                raise
        return inst

    @staticmethod
    def trace(fn, *, inputs, targets=None, loss=None, state=None, params=None,
              entrypoints=None):
        """Call ``fn(**inputs)`` once and seal its Tensor outputs.

        ``loss(result, **targets)`` returns a Tensor or a named loss dictionary.
        Supply state explicitly (``nn.get_state_dict(net)`` is a convenience).
        Preserve logical roots before construction; no host control-flow tracing,
        automatic train/eval modes, or subsequent calls to the authoring object.
        """
        if not callable(fn):
            raise TypeError('Model.trace requires a callable')
        inputs = dict(inputs)
        targets = dict(targets or {})
        result = fn(**inputs)
        losses = loss(result, **targets) if loss is not None else None
        return Model.from_tensors(inputs=inputs, targets=targets, outputs=result,
                                  losses=losses, state=state, params=params,
                                  entrypoints=entrypoints)

    @staticmethod
    def from_tensors(
        inputs=None,
        outputs=None,
        *,
        targets=None,
        losses=None,
        params=None,
        state=None,
        entrypoints=None,
        modules=None,
    ):
        """Package named Tensor roots as a runnable/exportable Model."""
        from .tensor import _ptr_value

        if params is not None and state is not None:
            raise ValueError('Model.from_tensors accepts params or state, not both')
        if state is not None:
            params = state

        inputs = _normalize_named_tensors(inputs, 'input')
        targets = _normalize_named_tensors(targets, 'target')
        outputs = _normalize_named_tensors(outputs, 'output')
        losses = _normalize_named_tensors(losses, 'loss')
        param_items = list(_param_items(params))

        named_tensors = []
        for group in (inputs, targets, outputs, losses):
            for name, tensor in group.items():
                named_tensors.append((name, _require_tensor(name, tensor)))
        for name, tensor in param_items:
            named_tensors.append((name, _require_tensor(name, tensor)))
        if not named_tensors:
            raise ValueError('Model.from_tensors requires at least one tensor')
        if not outputs and not losses:
            raise ValueError('Model.from_tensors requires outputs or losses')

        ctx = named_tensors[0][1]._ctx
        ctx_key = _ptr_value(ctx)
        for name, tensor in named_tensors:
            _check_ctx(name, tensor, ctx, ctx_key)

        for name, tensor in list(inputs.items()) + list(targets.items()):
            _ensure_storage_binding(name, tensor)

        keepalive = []
        binding_rows = []

        def add_binding(name, role, tensor, flags=0):
            name_b = _name_bytes(name)
            keepalive.append(name_b)
            if role == ROLE_PARAM and not tensor.is_param:
                flags |= _BIND_F_FROZEN
            binding_rows.append(_ffi.PolyBindingSpec(name_b, int(role), tensor._tensor, int(flags)))

        for name, tensor in inputs.items():
            add_binding(name, ROLE_INPUT, tensor)
        for name, tensor in targets.items():
            add_binding(name, ROLE_TARGET, tensor)
        for name, tensor in param_items:
            add_binding(name, ROLE_AUX if state is not None and not tensor.is_param else ROLE_PARAM, tensor)
        for name, tensor in outputs.items():
            add_binding(name, ROLE_OUTPUT, tensor)
        for name, tensor in losses.items():
            add_binding(name, ROLE_OUTPUT, tensor)

        entry_rows = []
        if entrypoints is not None:
            for entry in entrypoints:
                name, entry_inputs, entry_outputs, objective, flags = _entry_fields(entry)
                if name is None:
                    raise ValueError('Model entrypoint is missing a name')
                entry_rows.append(_entrypoint_spec(
                    name,
                    _entry_name_list(entry_inputs),
                    _entry_name_list(entry_outputs),
                    objective=objective,
                    flags=flags,
                    keepalive=keepalive,
                ))
        else:
            if outputs:
                entry_rows.append(_entrypoint_spec(
                    'forward', list(inputs.keys()), list(outputs.keys()), keepalive=keepalive
                ))
            if losses:
                objective = next(iter(losses)) if len(losses) == 1 else None
                entry_rows.append(_entrypoint_spec(
                    'loss', list(inputs.keys()) + list(targets.keys()), list(losses.keys()),
                    objective=objective, keepalive=keepalive
                ))

        inst = _model_from_binding_specs(ctx, binding_rows, entry_rows, keepalive)
        if modules is not None:
            try:
                inst.define_modules(modules)
            except Exception:
                inst.free()
                raise
        return inst

    def define_modules(self, modules):
        """Retain exact logical module cuts for explicit device-map placement.

        Each row is ``{'name', 'inputs', 'output'}`` or
        ``(name, inputs, output)``. Inputs and output are Tensor objects from
        this Model's construction context. This records product metadata;
        it does not alter the current physical graph.
        """
        rows = list(modules or ())
        if not rows:
            raise ValueError('define_modules requires at least one module')

        names = []
        counts = []
        flat_inputs = []
        outputs = []
        for row in rows:
            name, inputs, output = _module_fields(row)
            if name is None:
                raise ValueError('Model module is missing a name')
            inputs = [_require_tensor(f'{name}.input', tensor)
                      for tensor in _module_inputs(inputs)]
            output = _require_tensor(f'{name}.output', output)
            names.append(_name_bytes(name))
            counts.append(len(inputs))
            flat_inputs.extend(tensor._tensor for tensor in inputs)
            outputs.append(output._tensor)

        name_arr = (ctypes.c_char_p * len(names))(*names)
        count_arr = (ctypes.c_int * len(counts))(*counts)
        input_arr = ((ctypes.c_void_p * len(flat_inputs))(*flat_inputs)
                     if flat_inputs else None)
        output_arr = (ctypes.c_void_p * len(outputs))(*outputs)
        rc = _get_lib().poly_model_define_module_arrays(
            self._ptr, name_arr, input_arr, count_arr, output_arr, len(rows)
        )
        if rc != 0:
            raise ValueError('invalid or ambiguous Model module cuts')
        return self

    def set_device_map(self, device_map):
        """Place retained modules on exact devices and publish atomically."""
        rows = list(device_map.items()) if isinstance(device_map, dict) else list(device_map or ())
        if not rows:
            raise ValueError('set_device_map requires at least one module mapping')
        modules = [_name_bytes(name) for name, _ in rows]
        devices = [_name_bytes(device) for _, device in rows]
        module_arr = (ctypes.c_char_p * len(modules))(*modules)
        device_arr = (ctypes.c_char_p * len(devices))(*devices)
        rc = _get_lib().poly_model_set_device_map_arrays(
            self._ptr, module_arr, device_arr, len(rows)
        )
        if rc != 0:
            raise ValueError('invalid, incomplete, or unsupported Model device map')
        return self

    def place(self, device):
        """Explicitly place retained logical roots on a device or module map."""
        if not self._ptr:
            raise RuntimeError('Model is disposed')
        if not isinstance(device, str):
            return self.set_device_map(device)
        device_id = _device_id(device)
        if device_id <= 0:
            raise ValueError(f'unsupported explicit device: {device!r}')
        if _get_lib().poly_model_set_device(self._ptr, device_id) != 0:
            raise RuntimeError(f'Model placement failed for {device!r}')
        return self

    @staticmethod
    def from_hf(model_path=None, *, config_json=None, weight_bytes_list=None,
                max_batch=1, max_seq_len=0, device=None):
        """Load a HuggingFace-format model as an Model."""
        from .hf import load_hf, load_hf_bytes

        if config_json is not None or weight_bytes_list is not None:
            if config_json is None or weight_bytes_list is None:
                raise ValueError('config_json and weight_bytes_list must be provided together')
            return load_hf_bytes(
                config_json, weight_bytes_list, max_batch, max_seq_len, device=device
            )
        if model_path is None:
            raise ValueError('model_path is required')
        return load_hf(
            model_path, max_batch=max_batch, max_seq_len=max_seq_len, device=device
        )

    @staticmethod
    def from_gguf(data, *, max_batch=1, max_seq_len=0, device=None):
        """Load a GGUF byte buffer or file path as an Model."""
        if isinstance(data, (str, pathlib.Path)):
            data = pathlib.Path(data).read_bytes()
        data = bytes(data)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ptr = _get_lib().poly_gguf_load(
            buf, len(data), int(max_batch), int(max_seq_len), _device_id(device)
        )
        if not ptr:
            raise RuntimeError('poly_gguf_load returned NULL')
        return Model(ptr)

    # ── Param Enumeration ────────────────────────────────────────────

    @property
    def param_count(self):
        return _get_lib().poly_model_param_count(self._ptr)

    def param_name(self, i):
        name = _get_lib().poly_model_param_name(self._ptr, i)
        return name.decode('utf-8') if name else None

    def param_shape(self, i):
        shape_buf = (ctypes.c_int64 * 8)()
        ndim = _get_lib().poly_model_param_shape(self._ptr, i, shape_buf, 8)
        return [shape_buf[d] for d in range(ndim)]

    def param_data(self, i):
        """Return an independent, flat copy in the parameter storage dtype."""
        name = self.param_name(i)
        if name is None:
            return None
        return self.read_buffer(name)

    def param_dtype(self, i):
        return _model_dtype_name(_get_lib().poly_model_param_dtype_id(self._ptr, i))

    def param_trainable(self, i):
        """Whether this parameter participates in convenience optimizer steps."""
        return bool(_get_lib().poly_model_param_trainable(self._ptr, i))

    def set_param_trainable(self, i, trainable):
        """Enable or freeze one parameter for Model.train_step()."""
        ret = _get_lib().poly_model_set_param_trainable(
            self._ptr, i, bool(trainable))
        if ret != 0:
            raise RuntimeError(f'set_param_trainable failed (ret={ret})')
        return self

    def params(self):
        """Iterate (name, shape, data) for all params."""
        for i in range(self.param_count):
            yield self.param_name(i), self.param_shape(i), self.param_data(i)

    # ── Buffer Enumeration ───────────────────────────────────────────

    @property
    def buf_count(self):
        return _get_lib().poly_model_buf_count(self._ptr)

    def buf_name(self, i):
        name = _get_lib().poly_model_buf_name(self._ptr, i)
        return name.decode('utf-8') if name else None

    def buf_role(self, i):
        return _get_lib().poly_model_buf_role(self._ptr, i)

    def buf_trainable(self, i):
        """Whether this named buffer is marked trainable."""
        return bool(_get_lib().poly_model_buf_trainable(self._ptr, i))

    def set_buf_trainable(self, i, trainable):
        """Enable or freeze a named buffer for Model.train_step()."""
        ret = _get_lib().poly_model_set_buf_trainable(
            self._ptr, i, bool(trainable))
        if ret != 0:
            raise RuntimeError(f'set_buf_trainable failed (ret={ret})')
        return self

    def buf_shape(self, i):
        shape_buf = (ctypes.c_int64 * 8)()
        ndim = _get_lib().poly_model_buf_shape(self._ptr, i, shape_buf, 8)
        return tuple(shape_buf[d] for d in range(ndim))

    def buf_data(self, i):
        """Return an independent, flat copy in the buffer storage dtype."""
        if not self._ptr:
            raise RuntimeError('Model is disposed')
        if i < 0 or i >= self.buf_count:
            return None
        shape = self.buf_shape(i)
        out = np.empty(math.prod(shape), dtype=_storage_dtype(self.buf_dtype(i)))
        ret = _get_lib().poly_model_read_buf(self._ptr, i, out.ctypes.data, out.nbytes)
        if ret != 0:
            raise RuntimeError(f'buffer read failed (ret={ret})')
        return out

    def read_buffer(self, name):
        """Copy named storage to a flat host array; never borrows C memory."""
        if not self._ptr:
            raise RuntimeError('Model is disposed')
        i = self.find_buf(name)
        if i < 0:
            raise KeyError(name)
        return self.buf_data(i)

    def write_buffer(self, name, data):
        """Write exact storage dtype and extent, flat or in the declared shape."""
        if not self._ptr:
            raise RuntimeError('Model is disposed')
        i = self.find_buf(name)
        if i < 0:
            raise KeyError(name)
        shape = self.buf_shape(i)
        array = np.asarray(data)
        if array.dtype != _storage_dtype(self.buf_dtype(i)):
            raise TypeError(f'{name!r}: expected {self.buf_dtype(i)} storage')
        if array.shape not in (shape, (math.prod(shape),)):
            raise ValueError(f'{name!r}: expected shape {shape} or flat storage')
        array = np.ascontiguousarray(array)
        ret = _get_lib().poly_model_write_buf(self._ptr, i, array.ctypes.data, array.nbytes)
        if ret != 0:
            raise RuntimeError(f'buffer write failed (ret={ret})')
        return self

    def buf_dtype(self, i):
        return _model_dtype_name(_get_lib().poly_model_buf_dtype_id(self._ptr, i))

    def find_buf(self, name):
        """Find buffer index by name, or -1."""
        for i in range(self.buf_count):
            if self.buf_name(i) == name:
                return i
        return -1

    def bindings(self):
        """Return value-only interface metadata, never raw UOp/storage pointers."""
        if not self._ptr:
            raise RuntimeError('Model has been disposed')
        return [{'name': self.buf_name(i), 'role': self.buf_role(i),
                 'dtype': self.buf_dtype(i), 'shape': self.buf_shape(i),
                 'trainable': self.buf_trainable(i)} for i in range(self.buf_count)]

    def entrypoints(self):
        """Return the sealed callable signatures and their declared objectives."""
        if not self._ptr:
            raise RuntimeError('Model has been disposed')
        lib = _get_lib()
        rows = []
        for i in range(lib.poly_model_entrypoint_count(self._ptr)):
            name = lib.poly_model_entrypoint_name(self._ptr, i)
            objective = lib.poly_model_entrypoint_objective(self._ptr, name)
            rows.append({'name': name.decode(), 'objective': objective.decode() if objective else None,
                         'inputs': [lib.poly_model_entrypoint_input_name(self._ptr, name, j).decode()
                                    for j in range(lib.poly_model_entrypoint_input_count(self._ptr, name))],
                         'outputs': [lib.poly_model_entrypoint_output_name(self._ptr, name, j).decode()
                                     for j in range(lib.poly_model_entrypoint_output_count(self._ptr, name))]})
        return rows

    def set_trainable(self, name, trainable):
        i = self.find_buf(name)
        if i < 0:
            raise KeyError(name)
        return self.set_buf_trainable(i, trainable)

    # ── Weight I/O ───────────────────────────────────────────────────

    def export_weights(self, *, include_optimizer=True):
        """Export model weights as safetensors bytes.

        Optimizer state is included by default because a PolyModel packages
        model and optimizer state together. Pass include_optimizer=False for a
        model-only checkpoint, matching tinygrad's separate model/optimizer
        state_dict calls.
        """
        flags = EXPORT_WEIGHTS_PARAMS
        if include_optimizer:
            flags |= EXPORT_WEIGHTS_OPTIMIZER
        out_len = ctypes.c_int(0)
        ptr = _get_lib().poly_model_export_weights_ex(self._ptr, ctypes.byref(out_len), flags)
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    def import_weights(self, data):
        """Import weights from safetensors bytes."""
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ret = _get_lib().poly_model_import_weights(self._ptr, buf, len(data))
        if ret != 0:
            raise RuntimeError(f'import_weights failed (ret={ret})')

    def export_ir(self):
        """Export IR graph as binary bytes."""
        out_len = ctypes.c_int(0)
        ptr = _get_lib().poly_model_export_ir(self._ptr, ctypes.byref(out_len))
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    def export_program(self):
        """Export the current placed compiled entrypoints as bound bytes."""
        out_len = ctypes.c_int(0)
        ptr = _get_lib().poly_model_export_program(self._ptr, ctypes.byref(out_len))
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    # ── Bundle I/O ───────────────────────────────────────────────────

    def save_bundle(self, *, include_optimizer=True):
        """Save as a poly.bundle@1 byte array (IR + weights + metadata)."""
        flags = EXPORT_WEIGHTS_PARAMS
        if include_optimizer:
            flags |= EXPORT_WEIGHTS_OPTIMIZER
        out_len = ctypes.c_int(0)
        ptr = _get_lib().poly_model_save_bundle_ex(self._ptr, ctypes.byref(out_len), flags)
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    @staticmethod
    def from_bundle(data):
        """Load from a poly.bundle@1 byte array."""
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ptr = _get_lib().poly_model_from_bundle(buf, len(data))
        return Model(ptr)

    # ── Execution ────────────────────────────────────────────────────

    def set_optimizer(self, kind, lr=0.01, beta1=0.9, beta2=0.999,
                      eps=1e-8, weight_decay=0.0, momentum=0.0,
                      nesterov=False, classic=False):
        """Configure optimizer before first train_step."""
        ret = _get_lib().poly_model_set_optimizer_ex(
            self._ptr, kind,
            ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
            ctypes.c_float(eps), ctypes.c_float(weight_decay),
            ctypes.c_float(momentum), bool(nesterov), bool(classic))
        if ret != 0:
            raise RuntimeError(f'set_optimizer failed (ret={ret})')

    def forward(self, **inputs):
        """Run forward pass. Pass input arrays as keyword args (name=array).

        Returns dict of output buffer names to numpy arrays.
        """
        return self.call('forward', inputs)

    def call(self, entrypoint, inputs=None, **kwargs):
        """Run one named entrypoint with its exact declared input signature."""
        if inputs is not None and kwargs:
            raise TypeError('Model.call accepts either an input mapping or keyword inputs')
        io = dict(inputs or kwargs)
        bindings, n = self._make_bindings(io)
        ret = _get_lib().poly_model_call(
            self._ptr, str(entrypoint).encode('utf-8'), bindings, n)
        if ret != 0:
            raise RuntimeError(f"call('{entrypoint}') failed (ret={ret})")
        return self._collect_outputs(str(entrypoint))

    def train_step(self, *, entrypoint=None, **io):
        """Run one training step. Pass input+target arrays as kwargs.

        Returns the loss value (float).
        """
        bindings, n = self._make_bindings(io)
        loss = ctypes.c_float(0.0)
        ret = _get_lib().poly_model_train_step(
            self._ptr, _name_bytes(entrypoint) if entrypoint is not None else None,
            bindings, n, ctypes.byref(loss))
        if ret != 0:
            raise RuntimeError(f'train_step failed (ret={ret})')
        return float(loss.value)

    def fit(self, data=None, *, epochs=1, optimizer=None, lr=0.01,
            beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
            momentum=0.0, nesterov=False, classic=False,
            on_step=None, entrypoint=None, **io):
        """Repeat one supplied batch for ``epochs`` steps.

        This is only orchestration: optimizer update graphs are still built by
        the C core and executed through the same train_step path as custom
        loops.
        """
        bindings = {}
        if data:
            bindings.update(data)
        bindings.update(io)
        if optimizer is not None:
            self.set_optimizer(
                _optimizer_kind(optimizer), lr=lr, beta1=beta1, beta2=beta2,
                eps=eps, weight_decay=weight_decay, momentum=momentum,
                nesterov=nesterov, classic=classic,
            )
        losses = []
        for step in range(int(epochs)):
            loss = self.train_step(entrypoint=entrypoint, **bindings)
            losses.append(loss)
            if on_step is not None:
                on_step(step, loss)
        return losses

    # ── Internals ────────────────────────────────────────────────────

    def _make_bindings(self, io_dict):
        """Convert {name: array} dict to PolyIOBinding array."""
        n = len(io_dict)
        arr = (_ffi.PolyIOBinding * n)()
        owners = []
        lib = _get_lib()
        for i, (name, data) in enumerate(io_dict.items()):
            if isinstance(data, np.ndarray):
                data = np.ascontiguousarray(data)
            else:
                data = np.ascontiguousarray(data, dtype=np.float32)
            dtype_id = lib.poly_dtype_id_by_name(data.dtype.name.encode('utf-8'))
            if dtype_id < 0:
                raise TypeError(f"unsupported Model input dtype: {data.dtype}")
            owners.append(data)
            arr[i].name = name.encode('utf-8')
            arr[i].data = ctypes.c_void_p(data.ctypes.data)
            arr[i].nbytes = data.nbytes
            arr[i].dtype_id = dtype_id
        # Converted Python lists/scalars are not otherwise owned after this
        # method returns. Keep every contiguous array alive through the C call.
        arr._owners = owners
        return arr, n

    def _collect_outputs(self, entrypoint='forward'):
        """Read only outputs declared by the selected entrypoint."""
        result = {}
        lib = _get_lib()
        encoded = str(entrypoint).encode('utf-8')
        n_outputs = lib.poly_model_entrypoint_output_count(self._ptr, encoded)
        if n_outputs < 0:
            raise RuntimeError(f"unknown Model entrypoint '{entrypoint}'")
        for output_index in range(n_outputs):
            raw_name = lib.poly_model_entrypoint_output_name(
                self._ptr, encoded, output_index)
            if not raw_name:
                raise RuntimeError(f"invalid output {output_index} for entrypoint '{entrypoint}'")
            name = raw_name.decode('utf-8')
            i = self.find_buf(name)
            if i < 0:
                raise RuntimeError(f"entrypoint '{entrypoint}' references missing output '{name}'")
            data = self.buf_data(i)
            if data is not None:
                # Tensor.numpy() preserves scalar and multidimensional shape.
                # buf_data already owns a copy; reshape without copying again.
                result[name] = data.reshape(self.buf_shape(i))
        return result
