"""PolyModel -- Python wrapper for the C PolyModel runtime.

Provides forward pass, training, and weight I/O for runnable/exportable
models created from IR bytes, bundles, or frontend tensor graphs. Named
architecture factories live in ``polygrad.models``.
"""

import ctypes
import ctypes.util
import math
import pathlib
import inspect
import weakref
import numpy as np
from . import _ffi
from .device import _device_id

_get_lib = _ffi.get_lib
_live_models = weakref.WeakSet()


def _import_context(runtime):
    from . import Runtime, _default_ctx
    if runtime is not None:
        if not isinstance(runtime, Runtime):
            raise TypeError('runtime must be a Polygrad Runtime')
        runtime._check_live()
    ctx = runtime._ctx if runtime is not None else _default_ctx
    if not ctx:
        raise RuntimeError('polygrad runtime has been disposed')
    return ctx


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
    if isinstance(params, (list, tuple)):
        return [(f'param_{i}', p) for i, p in enumerate(params)]
    from .nn.state import get_state_dict
    return list(get_state_dict(params).items())

def _synchronous_result(value):
    if inspect.isawaitable(value) or inspect.isasyncgen(value):
        if inspect.iscoroutine(value):
            value.close()  # An unstarted coroutine owns no captured computation.
        raise TypeError('Model author and loss must be synchronous')
    return value


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
    return Model._from_handle(ptr, ctx)


class Model:
    """C-owned state and callables with sealed topology and optional training.

    Author with ordinary Tensor code or ``from_callable``; overriding a Python method
    does not change the captured executable or synchronize authoring attributes.
    """

    def __init__(self, source=None, *, inputs=None, targets=None,
                 outputs=None, entrypoints=None, params=None, losses=None,
                 loss=None, modules=None, runtime=None):
        self._ptr = self._ctx = None
        if inspect.isclass(source):
            raise TypeError('Model expects an object, not a class; instantiate it first')
        if isinstance(source, dict):
            if any(v is not None for v in (inputs, targets, outputs, entrypoints, params, losses, loss, modules)):
                raise TypeError('Model configuration cannot be combined with Tensor bindings or a callable loss')
            if source.get('format') != 'poly.modeldef@1' or source.get('type') not in ('sequential', 'graph'):
                raise ValueError('Model configuration requires format="poly.modeldef@1" and type="sequential" or "graph"')
            from .models import Sequential, Graph
            self._adopt((Sequential if source['type'] == 'sequential' else Graph)(source, runtime=runtime))
            return
        if runtime is not None:
            raise TypeError('runtime is only for configuration construction; Tensor bindings select their owning Runtime')
        if callable(source):
            if outputs is not None or losses is not None or modules is not None:
                raise TypeError('Callable Model cannot be combined with prebuilt outputs, losses or modules')
            built = Model.from_callable(source, inputs=inputs, targets=targets, loss=loss,
                                        params=params, entrypoints=entrypoints)
            self._adopt(built)
            return
        if loss is not None:
            raise TypeError('loss requires a callable Model source; use losses for prebuilt tensors')
        spec_args = (inputs, targets, outputs, entrypoints, params, losses, modules)
        has_spec = any(v is not None for v in spec_args)
        if has_spec:
            if source is not None:
                raise TypeError('Model source cannot be combined with tensor bindings')
            built = Model.from_tensors(
                inputs=inputs,
                targets=targets,
                outputs=outputs,
                losses=losses,
                params=params,
                entrypoints=entrypoints,
                modules=modules,
            )
            self._adopt(built)
            return

        raise TypeError('Model expects a callable, tagged configuration or Tensor bindings; use Model.load for bytes')

    @staticmethod
    def _from_handle(ptr, ctx=None):
        if not ptr:
            raise RuntimeError('Failed to create PolyModel (NULL pointer)')
        model = Model.__new__(Model)
        model._ptr, model._ctx = ptr, ctx
        if ctx:
            _live_models.add(model)
        return model

    def _adopt(self, built):
        self._ptr, self._ctx = built._ptr, built._ctx
        built._ptr = built._ctx = None
        if self._ctx:
            _live_models.add(self)

    def dispose(self):
        """Release this Model's C ownership; safe to call repeatedly."""
        self.free()

    def save(self, path=None, *, include_optimizer=True):
        """Return a portable bundle, optionally writing it to a local path."""
        data = self.save_bundle(include_optimizer=include_optimizer)
        if path is not None:
            pathlib.Path(path).write_bytes(data)
        return data

    @staticmethod
    def load(source, *, runtime=None):
        """Load bundle bytes or a local path, without the authoring object."""
        if isinstance(source, (str, pathlib.Path)):
            source = pathlib.Path(source).read_bytes()
        return Model.from_bundle(source, runtime=runtime)

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
    def from_ir(ir_bytes, weights_bytes=None, *, runtime=None):
        """Create from IR binary + optional safetensors weights."""
        ctx = _import_context(runtime)
        ir_buf = (ctypes.c_uint8 * len(ir_bytes)).from_buffer_copy(ir_bytes)
        w_buf = None
        w_len = 0
        if weights_bytes:
            w_buf = (ctypes.c_uint8 * len(weights_bytes)).from_buffer_copy(weights_bytes)
            w_len = len(weights_bytes)
        ptr = _get_lib().poly_model_from_ir_into(ctx, ir_buf, len(ir_bytes), w_buf, w_len, 0)
        return Model._from_handle(ptr, ctx)

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
        return Model._from_handle(ptr)

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
    def from_callable(fn, *, inputs, targets=None, loss=None, params=None,
                      entrypoints=None):
        """Call ``fn(**inputs)`` once and seal its Tensor outputs.

        ``loss(result, **targets)`` returns a Tensor or a named loss dictionary.
        Callable objects supply named state unless ``params`` overrides it.
        Preserve newly constructed logical roots; no host control-flow tracing,
        automatic train/eval modes, or subsequent calls to the authoring object.
        """
        if not callable(fn) or inspect.isclass(fn):
            raise TypeError('Model.from_callable requires a callable instance, not a class')
        for callback in (fn, loss):
            if callback is None:
                continue
            if not callable(callback):
                raise TypeError('Model author and loss must be callable')
            call = callback if inspect.isroutine(callback) else callback.__call__
            if inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(call):
                raise TypeError('Model author and loss must be synchronous')
        from .tensor import _ptr_value
        inputs = dict(inputs or {})
        targets = dict(targets or {})
        collect_object = params is None and not inspect.isroutine(fn)
        if params is None:
            params = {} if inspect.isroutine(fn) else dict(_param_items(fn))
        params = dict(_param_items(params))
        named = list(inputs.items()) + list(targets.items()) + list(params.items())
        if not named:
            raise ValueError('Model capture requires an input or named state Tensor to select its Runtime')
        ctx = _require_tensor(*named[0])._ctx
        for name, tensor in named:
            _check_ctx(name, _require_tensor(name, tensor), ctx, _ptr_value(ctx))
            if tensor.uop_logical is None:
                raise ValueError(f'{name!r} has no logical source; construct it with logical retention enabled')
        for name, tensor in params.items():
            if any(tensor is t for t in (*inputs.values(), *targets.values())):
                raise ValueError(f'{name!r} is both input/target and model state; supply a params override')
        lib = _get_lib()
        policy = lib.poly_ctx_get_logical_policy(ctx)
        lib.poly_ctx_set_logical_policy(ctx, 1)  # ALWAYS applies only to new Tensor wrappers.
        try:
            result = _synchronous_result(fn(**inputs))
            losses = _synchronous_result(loss(result, **targets)) if loss is not None else None
            # Lazy layer initialization happens in the authoring call, not sealing.
            if collect_object:
                params = dict(_param_items(fn))
                for name, tensor in params.items():
                    if any(tensor is t for t in (*inputs.values(), *targets.values())):
                        raise ValueError(f'{name!r} is both input/target and model state; supply a params override')
        finally:
            lib.poly_ctx_set_logical_policy(ctx, policy)
        return Model.from_tensors(inputs=inputs, targets=targets, outputs=result,
                                  losses=losses, params=params,
                                  entrypoints=entrypoints)

    @staticmethod
    def from_tensors(
        inputs=None,
        outputs=None,
        *,
        targets=None,
        losses=None,
        params=None,
        entrypoints=None,
        modules=None,
    ):
        """Package named Tensor roots as a runnable/exportable Model."""
        from .tensor import _ptr_value

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

        keepalive = []
        binding_rows = []

        def add_binding(name, role, tensor):
            name_b = _name_bytes(name)
            keepalive.append(name_b)
            binding_rows.append(_ffi.PolyBindingSpec(name_b, int(role), tensor._tensor, 0))

        for name, tensor in inputs.items():
            add_binding(name, ROLE_INPUT, tensor)
        for name, tensor in targets.items():
            add_binding(name, ROLE_TARGET, tensor)
        for name, tensor in param_items:
            add_binding(name, ROLE_PARAM if tensor.is_param else ROLE_AUX, tensor)
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
                max_batch=1, max_seq_len=0, device=None, runtime=None):
        """Load a HuggingFace-format model as an Model."""
        from .hf import load_hf, load_hf_bytes

        if config_json is not None or weight_bytes_list is not None:
            if config_json is None or weight_bytes_list is None:
                raise ValueError('config_json and weight_bytes_list must be provided together')
            return load_hf_bytes(
                config_json, weight_bytes_list, max_batch, max_seq_len, device=device, runtime=runtime
            )
        if model_path is None:
            raise ValueError('model_path is required')
        return load_hf(
            model_path, max_batch=max_batch, max_seq_len=max_seq_len, device=device, runtime=runtime
        )

    @staticmethod
    def from_gguf(data, *, max_batch=1, max_seq_len=0, device=None, runtime=None):
        """Load a GGUF byte buffer or file path as an Model."""
        if isinstance(data, (str, pathlib.Path)):
            data = pathlib.Path(data).read_bytes()
        data = bytes(data)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ctx = _import_context(runtime)
        ptr = _get_lib().poly_gguf_load_into(
            ctx, buf, len(data), int(max_batch), int(max_seq_len), _device_id(device) if device is not None else 0
        )
        if not ptr:
            raise RuntimeError('poly_gguf_load returned NULL')
        return Model._from_handle(ptr, ctx)

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

    def buf_current_shape(self, i):
        """Concrete extents of the last successful call; capacity before first use."""
        shape = (ctypes.c_int64 * 8)()
        ndim = _get_lib().poly_model_buf_current_shape(self._ptr, i, shape, 8)
        if ndim < 0:
            raise ValueError('invalid buffer index')
        return tuple(shape[:ndim])

    def buf_shape_bounds(self, i):
        lower, upper = (ctypes.c_int64 * 8)(), (ctypes.c_int64 * 8)()
        ndim = _get_lib().poly_model_buf_shape_bounds(self._ptr, i, lower, upper, 8)
        if ndim < 0:
            raise ValueError('invalid buffer index')
        return tuple(zip(lower[:ndim], upper[:ndim]))

    def buf_data(self, i):
        """Return an independent, flat copy in the buffer storage dtype."""
        if not self._ptr:
            raise RuntimeError('Model is disposed')
        if i < 0 or i >= self.buf_count:
            return None
        shape = self.buf_current_shape(i)
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
                 'shape_bounds': self.buf_shape_bounds(i),
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

    def summary(self):
        """Describe the interface without executing graphs or reading state bytes."""
        roles = ('PARAM', 'INPUT', 'TARGET', 'OUTPUT', 'AUX')
        bindings = [f"  {b['name']}: {roles[b['role']]} {b['dtype']}[{','.join(map(str, b['shape']))}]" +
                    ((' trainable' if b['trainable'] else ' frozen') if b['role'] == ROLE_PARAM else '')
                    for b in self.bindings()]
        entries = [f"  {e['name']}({', '.join(e['inputs'])}) -> {', '.join(e['outputs'])}" +
                   (f"; objective={e['objective']}" if e['objective'] is not None else '')
                   for e in self.entrypoints()]
        return '\n'.join(['Model', 'Bindings:', *bindings, 'Entrypoints:', *entries])

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
        if not self._ptr:
            raise RuntimeError('Model has been disposed')
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
    def from_bundle(data, *, runtime=None):
        """Load from a poly.bundle@1 byte array."""
        ctx = _import_context(runtime)
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ptr = _get_lib().poly_model_from_bundle_into(ctx, buf, len(data), 0)
        return Model._from_handle(ptr, ctx)

    # ── Execution ────────────────────────────────────────────────────

    def set_optimizer(self, kind, lr=0.01, beta1=0.9, beta2=0.999,
                      eps=1e-8, weight_decay=0.0, momentum=0.0,
                      nesterov=False, classic=False):
        """Configure optimizer before first train_step."""
        ret = _get_lib().poly_model_set_optimizer_ex(
            self._ptr, _optimizer_kind(kind),
            ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
            ctypes.c_float(eps), ctypes.c_float(weight_decay),
            ctypes.c_float(momentum), bool(nesterov), bool(classic))
        if ret != 0:
            raise RuntimeError(f'set_optimizer failed (ret={ret})')

    def forward(self, **inputs):
        """Run forward pass. Pass input arrays as keyword args (name=array).

        Returns named arrays, or owned Tensor results when any input is a Tensor.
        """
        return self.call('forward', inputs)

    def call(self, entrypoint, inputs=None, **kwargs):
        """Run an entrypoint. Tensor inputs select owned, device-resident Tensor outputs.

        Results are eager snapshots, not differentiable calls through the Model.
        Their Runtime must remain alive; subsequent calls and Model disposal do
        not change them. Array-only inputs retain the NumPy output contract.
        """
        if inputs is not None and kwargs:
            raise TypeError('Model.call accepts either an input mapping or keyword inputs')
        io = dict(inputs or kwargs)
        bindings, n = self._make_bindings(io)
        from .tensor import Tensor
        if any(isinstance(value, Tensor) for value in io.values()):
            lib = _get_lib()
            ep = str(entrypoint).encode('utf-8')
            count = lib.poly_model_entrypoint_output_count(self._ptr, ep)
            if count < 0: raise ValueError(f"unknown entrypoint: {entrypoint}")
            handles = (_ffi._ptr * count)()
            if lib.poly_model_call_tensors(self._ptr, ep, bindings, n, handles, count) != 0:
                raise RuntimeError(f"Tensor call('{entrypoint}') failed")
            result = {}
            try:
                for i in range(count):
                    name = lib.poly_model_entrypoint_output_name(self._ptr, ep, i).decode()
                    result[name] = Tensor(_ctx=self._ctx, _tensor=handles[i],
                                          _device=lib.poly_device_name(lib.poly_tensor_device(handles[i])).decode()).is_param_(False)
                    handles[i] = None
            finally:
                for handle in handles:
                    if handle: lib.poly_tensor_release(handle)
            return result
        ret = _get_lib().poly_model_call(
            self._ptr, str(entrypoint).encode('utf-8'), bindings, n)
        if ret != 0:
            raise RuntimeError(f"call('{entrypoint}') failed (ret={ret})")
        return self._collect_outputs(str(entrypoint))

    def train_step(self, inputs=None, *, entrypoint=None, **io):
        """Run one training step with a named mapping or input+target kwargs.

        Returns the loss value (float).
        """
        if inputs is not None and io:
            raise TypeError('Model.train_step accepts either an input mapping or keyword inputs')
        bindings, n = self._make_bindings(dict(inputs if inputs is not None else io))
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
            on_step=None, entrypoint=None, batch_size=None, remainder='error', **io):
        """Train in input order; return one loss per optimizer step.

        With batch_size=None, repeat the supplied batch once per epoch. With
        batch_size, traverse host arrays along axis zero each epoch. The batch
        size must match the sealed signature. An incomplete batch is rejected
        before training, unless remainder='drop' is explicitly selected.
        Device Tensor datasets use an explicit train_step loop for now.
        """
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 0:
            raise ValueError('epochs must be a nonnegative integer')
        if data is not None and io:
            raise TypeError('Model.fit accepts either a data mapping or keyword inputs')
        bindings = dict(data if data is not None else io)
        if remainder not in ('error', 'drop'):
            raise ValueError("remainder must be 'error' or 'drop'")
        batches = 1
        if batch_size is not None:
            bindings, batches = self._fit_dataset(bindings, batch_size, remainder, entrypoint)
        if optimizer is not None:
            self.set_optimizer(
                _optimizer_kind(optimizer), lr=lr, beta1=beta1, beta2=beta2,
                eps=eps, weight_decay=weight_decay, momentum=momentum,
                nesterov=nesterov, classic=classic,
            )
        losses = []
        for _ in range(epochs):
            for batch in range(batches):
                current = bindings if batch_size is None else {
                    name: value[batch*batch_size:(batch+1)*batch_size] for name, value in bindings.items()}
                loss = self.train_step(current, entrypoint=entrypoint)
                losses.append(loss)
                if on_step is not None:
                    on_step(len(losses)-1, loss)
        return losses

    def _fit_dataset(self, data, batch_size, remainder, entrypoint):
        """Preflight the whole dataset before optimizer configuration or writes."""
        from .tensor import Tensor
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        entries = self.entrypoints()
        selected = [e for e in entries if e['name'] == entrypoint] if entrypoint is not None else [
            e for e in entries if e['objective']]
        if not selected and entrypoint is None:
            selected = [e for e in entries if e['name'] == 'loss']
        if len(selected) != 1:
            raise ValueError('fit requires one selected objective entrypoint')
        schema = {b['name']: b for b in self.bindings()}
        required = selected[0]['inputs'] or [n for n, b in schema.items() if b['role'] in (ROLE_INPUT, ROLE_TARGET)]
        if set(data) != set(required):
            raise ValueError('fit data must match the selected entrypoint inputs')
        normalized, count = {}, None
        for name, value in data.items():
            if isinstance(value, Tensor):
                raise TypeError('minibatch Tensor datasets require an explicit train_step loop')
            value = np.asarray(value) if isinstance(value, np.ndarray) else np.asarray(value, dtype=np.float32)
            declared = schema[name]['shape']
            if not declared or declared[0] != batch_size:
                raise ValueError(f"batch_size must match the declared first axis of '{name}'")
            if not value.dtype.isnative or value.dtype.name != schema[name]['dtype']:
                raise TypeError(f"fit input '{name}' has the wrong storage dtype")
            width = math.prod(declared[1:])
            if value.ndim == 1 and width > 0 and value.size % width == 0:
                value = value.reshape((-1, *declared[1:]))
            if value.ndim != len(declared) or value.shape[1:] != declared[1:]:
                raise ValueError(f"fit input '{name}' has incompatible sample shape")
            if count is not None and count != len(value):
                raise ValueError('fit inputs must have the same sample count')
            count = len(value)
            normalized[name] = value
        if count is None or count < batch_size:
            raise ValueError('fit requires at least one complete batch')
        if count % batch_size and remainder == 'error':
            raise ValueError('unsupported incomplete batch remainder; use remainder="drop" explicitly')
        return normalized, count // batch_size

    # ── Internals ────────────────────────────────────────────────────

    def _make_bindings(self, io_dict):
        """Pack host arrays or borrowed Tensor handles for C admission."""
        from .tensor import Tensor
        n = len(io_dict)
        arr = (_ffi.PolyIOBinding * n)()
        owners = []
        lib = _get_lib()
        for i, (name, data) in enumerate(io_dict.items()):
            arr[i].name = name.encode('utf-8')
            if isinstance(data, Tensor):
                if data._ctx != self._ctx or not data._tensor:
                    raise ValueError(f"Model input '{name}' must be a live Tensor in its Runtime")
                arr[i].tensor = data._tensor
                owners.append(data)
                continue
            if isinstance(data, np.ndarray):
                data_shape = data.shape
                data = np.ascontiguousarray(data)
            else:
                data = np.asarray(data, dtype=np.float32)
                # A plain number, like JS number input, is flat one-element
                # storage. A zero-dimensional ndarray above is explicitly scalar.
                data_shape = data.shape if data.ndim else (1,)
                data = np.ascontiguousarray(data)
            dtype_id = lib.poly_dtype_id_by_name(data.dtype.name.encode('utf-8'))
            if not data.dtype.isnative:
                raise TypeError('Model input arrays require native byte order')
            if dtype_id < 0:
                raise TypeError(f"unsupported Model input dtype: {data.dtype}")
            owners.append(data)
            arr[i].data = ctypes.c_void_p(data.ctypes.data)
            arr[i].nbytes = data.nbytes
            arr[i].dtype_id = dtype_id
            # Rank-one arrays retain the flat-storage contract. Higher ranks
            # and scalars carry concrete shape, not just an equal byte count.
            if len(data_shape) != 1:
                shape = (ctypes.c_int64 * max(1, len(data_shape)))(*data_shape)
                owners.append(shape)
                arr[i].shape = shape
                arr[i].ndim = len(data_shape)
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
                result[name] = data.reshape(self.buf_current_shape(i))
        return result
