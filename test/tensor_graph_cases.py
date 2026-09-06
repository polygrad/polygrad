#!/usr/bin/env python3
"""Emit canonical, identity-preserving Tensor graph cases for one engine.

The same program is run once with pinned tinygrad and once with Polygrad.
Engine-local allocation numbers and device spelling are normalized, while
operation, dtype, argument, source order, arity, and node sharing are retained.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re


ENGINE = os.environ.get("ENGINE", "tinygrad")
if ENGINE == "tinygrad":
    from tinygrad import Context, Tensor, dtypes, nn
    from tinygrad.nn.optim import Adam, AdamW, SGD
    from tinygrad.uop.ops import UOp
elif ENGINE == "polygrad":
    from polygrad import Context, Tensor, Variable, _ffi, dtypes, nn
    from polygrad.nn.optim import Adam, AdamW, SGD
else:
    raise RuntimeError(f"unknown ENGINE={ENGINE!r}")


def training_context():
    # The June control uses Tensor.train(); current Tinygrad and Polygrad use
    # the shared TRAINING ContextVar. This is an explicit reference-lane
    # adapter, not a graph-divergence suppression.
    if hasattr(Tensor, "train"):
        return Tensor.train()
    return Context(TRAINING=1)

if ENGINE == "polygrad":
    class _PolyDType(ctypes.Structure):
        _fields_ = [
            ("priority", ctypes.c_int8),
            ("bitsize", ctypes.c_uint16),
            ("name", ctypes.c_char_p),
            ("fmt", ctypes.c_char),
        ]

    class _PolyUOpHead(ctypes.Structure):
        _fields_ = [("op", ctypes.c_int), ("dtype", _PolyDType)]

    class _PolyArgValue(ctypes.Union):
        # Only the union alignment is needed to locate PolyArg.kind. The full
        # C union is larger, but its first member begins at the same offset.
        _fields_ = [("i", ctypes.c_int64), ("f", ctypes.c_double), ("ptr", ctypes.c_void_p)]

    class _PolyArgHead(ctypes.Structure):
        _fields_ = [("kind", ctypes.c_int), ("value", _PolyArgValue)]

    class _PolyUOpArgHead(ctypes.Structure):
        _fields_ = [
            ("op", ctypes.c_int), ("dtype", _PolyDType), ("src", ctypes.c_void_p),
            ("n_src", ctypes.c_uint16), ("arg", _PolyArgHead),
        ]


def node_key(node):
    return id(node) if ENGINE == "tinygrad" else int(node.raw)


def op_name(node):
    return node.op.name if ENGINE == "tinygrad" else node.op_name


def dtype_name(node):
    if ENGINE == "tinygrad":
        return node.dtype.name
    dtype = ctypes.cast(node.raw, ctypes.POINTER(_PolyUOpHead)).contents.dtype
    name = dtype.name.decode("utf-8") if dtype.name else "void"
    # PolyDType.name is also the renderer token for native half types, while
    # tinygrad DType.name is semantic. Canonical graph evidence compares the
    # semantic dtype; it must not mistake C spellings for a dtype mismatch.
    # Pinned tinygrad names float16 semantically as "half", while its BF16
    # semantic name is the literal "__bf16". Only normalize the former.
    name = {"__fp16": "half"}.get(name, name)
    return name


def diagnostic_dtype_name(node):
    if ENGINE == "tinygrad":
        return dtype_name(node)
    dtype = ctypes.cast(node.raw, ctypes.POINTER(_PolyUOpHead)).contents.dtype
    return dtype.name.decode("utf-8") if dtype.name else "void"


def poly_text(node):
    lib = _ffi._lib
    lib.poly_uop_str.restype = ctypes.c_void_p
    lib.poly_uop_str.argtypes = [ctypes.c_void_p]
    raw = lib.poly_uop_str(node.raw)
    if not raw:
        raise RuntimeError("poly_uop_str failed")
    try:
        text = ctypes.string_at(raw).decode("utf-8")
    finally:
        free = ctypes.CDLL(None).free
        free.argtypes = [ctypes.c_void_p]
        free(raw)
    if len(text) >= 250:
        raise RuntimeError("poly_uop_str is too close to its current 256-byte diagnostic limit")
    return text


def node_arg(node):
    if ENGINE == "tinygrad":
        # Pinned tinygrad: tinygrad/uop/ops.py:UOp.argstr.
        return node.argstr()
    text = poly_text(node)
    prefix = f"UOp({op_name(node)}"
    if diagnostic_dtype_name(node) != "void":
        prefix += f", {diagnostic_dtype_name(node)}"
    suffix = f", src={len(node.src)})" if node.src else ")"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise RuntimeError(f"cannot split UOp diagnostic: {text!r}")
    middle = text[len(prefix):-len(suffix)]
    return middle[2:] if middle.startswith(", ") else "None"


def normalize_arg(node, arg):
    op = op_name(node)
    # Absolute allocator counters are not cross-process identities. Canonical
    # node sharing and the number/order of UNIQUE nodes preserve identity.
    if op == "BUFFER":
        return "<buffer>"
    if op == "UNIQUE":
        return "<unique>"
    if op == "DEVICE":
        if ENGINE == "tinygrad":
            name = arg.strip("'\"").upper()
        else:
            raw_name = _ffi._lib.poly_uop_device_name(node.ctx, node.raw)
            if not raw_name:
                device = _ffi._lib.poly_uop_device(node.raw)
                raw_name = _ffi._lib.poly_device_name(device)
            name = raw_name.decode("utf-8").upper() if raw_name else f"DEVICE:{device}"
        # Both are the frontend-owned, creation-only source domain. Polygrad's
        # C runtime calls it HOST because no Python executor exists in core;
        # pinned tinygrad calls the same graph role PYTHON.
        return "PYTHON" if name in {"HOST", "PYTHON"} else name
    if op == "COPY" and len(arg) >= 2 and arg[0] in {'"', "'"} and arg[-1] == arg[0]:
        return arg[1:-1].upper()
    if op == "CONST" and dtype_name(node) in {
        "half", "float", "double", "bfloat16", "float16", "float32", "float64",
        "__fp16", "__bf16", "weakfloat",
    }:
        # tinygrad's diagnostic wraps float constants in ConstFloat(...);
        # Polygrad prints the same typed value directly. Compare the value,
        # retaining signed zero and non-finite classes.
        if ENGINE == "polygrad":
            kind = ctypes.cast(node.raw, ctypes.POINTER(_PolyUOpArgHead)).contents.arg.kind
            if kind != 2:  # POLY_ARG_FLOAT
                return f"<invalid-float-arg-kind:{kind}:{arg}>"
        value_text = arg
        if value_text.startswith("ConstFloat(") and value_text.endswith(")"):
            value_text = value_text[len("ConstFloat("):-1]
        value = float(value_text)
        if math.isnan(value):
            return "<float:nan>"
        if math.isinf(value):
            return "<float:+inf>" if value > 0 else "<float:-inf>"
        return f"<float:{value.hex()}>"
    if op == "CONST" and dtype_name(node) == "bool":
        return "True" if arg.lower() in {"1", "true"} else "False"
    arg = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", arg)
    return arg.replace("Ops.", "").replace(" ", "").replace(",)", ")")


def canonical_graph(root):
    ids, records = {}, []

    def visit(node):
        key = node_key(node)
        if key in ids:
            return ids[key]
        src_ids = [visit(src) for src in node.src]
        node_id = len(records)
        ids[key] = node_id
        records.append({
            "id": node_id,
            "op": op_name(node),
            "dtype": dtype_name(node),
            "arg": normalize_arg(node, node_arg(node)),
            "src": src_ids,
        })
        return node_id

    return {"root": visit(root), "nodes": records}


def logical(tensor):
    return tensor.uop if ENGINE == "tinygrad" else tensor.uop_logical


def realized_input(*shape):
    # Avoid comparing host-constructor upload policy. The acceptance subject is
    # the Tensor operation rooted at two equivalent realized buffers.
    return Tensor.zeros(*shape, device="CPU").realize()


def realized_empty(*shape):
    return Tensor.empty(*shape, device="CPU").realize()


def case_basic_alu():
    a, b = realized_input(2), realized_input(2)
    out = a + b
    return {"physical": out.uop, "logical": logical(out)}


def case_radd_operator_occurrence():
    out = 3.0 + realized_empty(2, 2).to("CUDA").to("CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_radd_named_occurrence():
    out = realized_empty(2, 2).to("CUDA").to("CPU").add(3.0, reverse=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_rmul_operator_occurrence():
    out = 3.0 * realized_empty(2, 2).to("CUDA").to("CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_rmul_named_occurrence():
    out = realized_empty(2, 2).to("CUDA").to("CPU").mul(3.0, reverse=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_radd_named_mixed_broadcast():
    current = realized_empty(2, 2).to("CUDA").to("CPU")
    other = Tensor.empty(1, 2, dtype="int32", device="CPU").realize()
    out = current.add(other, reverse=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_rmul_named_mixed_broadcast():
    current = realized_empty(2, 2).to("CUDA").to("CPU")
    other = Tensor.empty(1, 2, dtype="int32", device="CPU").realize()
    out = current.mul(other, reverse=True)
    return {"physical": out.uop, "logical": logical(out)}


def typed_realized_empty(dtype):
    return Tensor.empty(2, 2, dtype=dtype, device="CPU").realize()


def case_bfloat16_host_add():
    t = Tensor([1.0, 2.0, 3.0, 4.0], dtype="bfloat16")
    out = t + t
    return {"physical": out.uop, "logical": logical(out)}


def fp8_host_add(dtype):
    t = Tensor([0.1, 1.5], dtype=dtype)
    out = t + t
    return {"physical": out.uop, "logical": logical(out)}


def case_fp8e4m3_host_add():
    return fp8_host_add("fp8e4m3")


def case_fp8e5m2_host_add():
    return fp8_host_add("fp8e5m2")


def case_fp8e4m3fnuz_host_add():
    return fp8_host_add("fp8e4m3fnuz")


def case_fp8e5m2fnuz_host_add():
    return fp8_host_add("fp8e5m2fnuz")


def case_sub_float32():
    out = typed_realized_empty("float32") - typed_realized_empty("float32")
    return {"physical": out.uop, "logical": logical(out)}


def case_exp_float16():
    out = typed_realized_empty("float16").exp()
    return {"physical": out.uop, "logical": logical(out)}


def case_exp_float32():
    out = typed_realized_empty("float32").exp()
    return {"physical": out.uop, "logical": logical(out)}


def case_exp_int32():
    out = typed_realized_empty("int32").exp()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_occurrence():
    out = realized_empty(2, 2).to("CUDA").to("CPU").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_int32():
    out = typed_realized_empty("int32").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_bool():
    out = typed_realized_empty("bool").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_bfloat16():
    out = typed_realized_empty("bfloat16").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_float64():
    out = typed_realized_empty("float64").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_sin_uint64():
    out = typed_realized_empty("uint64").sin()
    return {"physical": out.uop, "logical": logical(out)}


def case_exp2_int32():
    out = typed_realized_empty("int32").exp2()
    return {"physical": out.uop, "logical": logical(out)}


def case_log2_int32():
    out = typed_realized_empty("int32").log2()
    return {"physical": out.uop, "logical": logical(out)}


def case_sqrt_int32():
    out = typed_realized_empty("int32").sqrt()
    return {"physical": out.uop, "logical": logical(out)}


def case_reciprocal_int32():
    out = typed_realized_empty("int32").reciprocal()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_occurrence():
    out = realized_empty(2, 2).to("CUDA").to("CPU").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_float16():
    out = typed_realized_empty("float16").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_int32():
    out = typed_realized_empty("int32").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_bfloat16():
    out = typed_realized_empty("bfloat16").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_float64():
    out = typed_realized_empty("float64").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_cos_uint64():
    out = typed_realized_empty("uint64").cos()
    return {"physical": out.uop, "logical": logical(out)}


def case_tan_occurrence():
    out = realized_empty(2, 2).to("CUDA").to("CPU").tan()
    return {"physical": out.uop, "logical": logical(out)}


def case_tan_int32():
    out = typed_realized_empty("int32").tan()
    return {"physical": out.uop, "logical": logical(out)}


def case_tan_bfloat16():
    out = typed_realized_empty("bfloat16").tan()
    return {"physical": out.uop, "logical": logical(out)}


def case_tan_float64():
    out = typed_realized_empty("float64").tan()
    return {"physical": out.uop, "logical": logical(out)}


def case_tan_uint64():
    out = typed_realized_empty("uint64").tan()
    return {"physical": out.uop, "logical": logical(out)}


def elementwise_occurrence(method, *args, **kwargs):
    out = getattr(realized_empty(2, 2).to("CUDA").to("CPU"), method)(
        *args, **kwargs
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_square_occurrence():
    return elementwise_occurrence("square")


def case_isnan_occurrence():
    return elementwise_occurrence("isnan")


def case_ceil_occurrence():
    return elementwise_occurrence("ceil")


def case_floor_occurrence():
    return elementwise_occurrence("floor")


def case_round_occurrence():
    return elementwise_occurrence("round")


def case_round_float16():
    return elementwise_typed("round", "float16")


def case_round_float32():
    return elementwise_typed("round", "float32")


def case_round_float64():
    return elementwise_typed("round", "float64")


def case_round_int32():
    return elementwise_typed("round", "int32")


def case_round_bool():
    return elementwise_typed("round", "bool")


def isinf_occurrence(detect_positive, detect_negative):
    return elementwise_occurrence(
        "isinf", detect_positive=detect_positive, detect_negative=detect_negative
    )


def isinf_typed(dtype, detect_positive, detect_negative):
    out = typed_realized_empty(dtype).isinf(
        detect_positive=detect_positive, detect_negative=detect_negative
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_isinf_occurrence_neither():
    return isinf_occurrence(False, False)


def case_isinf_occurrence_negative():
    return isinf_occurrence(False, True)


def case_isinf_occurrence_positive():
    return isinf_occurrence(True, False)


def case_isinf_occurrence_both():
    return isinf_occurrence(True, True)


def case_isinf_float16_neither():
    return isinf_typed("float16", False, False)


def case_isinf_float16_negative():
    return isinf_typed("float16", False, True)


def case_isinf_float16_positive():
    return isinf_typed("float16", True, False)


def case_isinf_float16_both():
    return isinf_typed("float16", True, True)


def case_isinf_float32_neither():
    return isinf_typed("float32", False, False)


def case_isinf_float32_negative():
    return isinf_typed("float32", False, True)


def case_isinf_float32_positive():
    return isinf_typed("float32", True, False)


def case_isinf_float32_both():
    return isinf_typed("float32", True, True)


def case_isinf_float64_neither():
    return isinf_typed("float64", False, False)


def case_isinf_float64_negative():
    return isinf_typed("float64", False, True)


def case_isinf_float64_positive():
    return isinf_typed("float64", True, False)


def case_isinf_float64_both():
    return isinf_typed("float64", True, True)


def case_isinf_int32_neither():
    return isinf_typed("int32", False, False)


def case_isinf_int32_negative():
    return isinf_typed("int32", False, True)


def case_isinf_int32_positive():
    return isinf_typed("int32", True, False)


def case_isinf_int32_both():
    return isinf_typed("int32", True, True)


def case_isinf_bool_neither():
    return isinf_typed("bool", False, False)


def case_isinf_bool_negative():
    return isinf_typed("bool", False, True)


def case_isinf_bool_positive():
    return isinf_typed("bool", True, False)


def case_isinf_bool_both():
    return isinf_typed("bool", True, True)


def case_sigmoid_occurrence():
    return elementwise_occurrence("sigmoid")


def case_tanh_occurrence():
    return elementwise_occurrence("tanh")


def case_relu6_occurrence():
    return elementwise_occurrence("relu6")


def case_leaky_relu_occurrence():
    return elementwise_occurrence("leaky_relu", 0.01)


def case_hardswish_occurrence():
    return elementwise_occurrence("hardswish")


def case_hardsigmoid_occurrence():
    return elementwise_occurrence("hardsigmoid")


def case_hardtanh_occurrence():
    return elementwise_occurrence("hardtanh", -1, 1)


def case_silu_occurrence():
    return elementwise_occurrence("silu")


def case_elu_occurrence():
    return elementwise_occurrence("elu", 1.0)


def case_sign_occurrence():
    return elementwise_occurrence("sign")


def case_abs_occurrence():
    return elementwise_occurrence("abs")


def case_logaddexp_scalar_occurrence():
    return elementwise_occurrence("logaddexp", 0.0)


def case_logaddexp_broadcast_occurrence():
    x = realized_empty(2, 2).to("CUDA").to("CPU")
    other = realized_empty(2, 1).to("CUDA").to("CPU")
    out = x.logaddexp(other)
    return {"physical": out.uop, "logical": logical(out)}


def case_softplus_occurrence():
    return elementwise_occurrence("softplus")


def case_softplus_beta2_occurrence():
    return elementwise_occurrence("softplus", 2.0)


def case_mish_occurrence():
    return elementwise_occurrence("mish")


def case_triu_host():
    out = Tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        device="CPU",
    ).triu(0)
    return {"physical": out.uop, "logical": logical(out)}


def case_triu_nondefault_device():
    out = Tensor.empty(2, 4, device="CPU").to("CUDA").triu(0)
    return {"physical": out.uop, "logical": logical(out)}


def case_tril_occurrence():
    out = realized_empty(2, 4).to("CUDA").to("CPU").tril(1)
    return {"physical": out.uop, "logical": logical(out)}


def elementwise_typed(method, dtype, *args):
    out = getattr(typed_realized_empty(dtype), method)(*args)
    return {"physical": out.uop, "logical": logical(out)}


def case_sigmoid_float16():
    return elementwise_typed("sigmoid", "float16")


def case_tanh_float16():
    return elementwise_typed("tanh", "float16")


def case_sign_float16():
    return elementwise_typed("sign", "float16")


def case_sign_int32():
    return elementwise_typed("sign", "int32")


def case_sign_uint32():
    return elementwise_typed("sign", "uint32")


def case_sign_bool():
    return elementwise_typed("sign", "bool")


def case_abs_bool():
    return elementwise_typed("abs", "bool")


def case_ceil_int32():
    return elementwise_typed("ceil", "int32")


def case_hardsigmoid_custom():
    return elementwise_typed("hardsigmoid", "float32", 0.2, 0.3)


def case_log_float16():
    out = typed_realized_empty("float16").log()
    return {"physical": out.uop, "logical": logical(out)}


def case_log_float32():
    out = typed_realized_empty("float32").log()
    return {"physical": out.uop, "logical": logical(out)}


def case_log_int32():
    out = typed_realized_empty("int32").log()
    return {"physical": out.uop, "logical": logical(out)}


def case_softmax_float16():
    out = typed_realized_empty("float16").softmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_softmax_float32():
    out = typed_realized_empty("float32").softmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_log_softmax_float16():
    out = typed_realized_empty("float16").log_softmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_log_softmax_float32():
    out = typed_realized_empty("float32").log_softmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_detach_float32():
    out = typed_realized_empty("float32").detach()
    return {"physical": out.uop, "logical": logical(out)}


def case_relu_float32():
    out = typed_realized_empty("float32").relu()
    return {"physical": out.uop, "logical": logical(out)}


def case_mlp_linear_relu():
    x = Tensor.empty(2, 3, device="CPU").realize()
    weight = Tensor.empty(4, 3, device="CPU").realize()
    bias = Tensor.empty(4, device="CPU").realize()
    # Pinned nn.Linear stores (out,in) and passes weight.transpose() to the
    # shared Tensor.linear(in,out) contract (nn/__init__.py:156-177 and
    # mixin/__init__.py:1335-1350). Exercise that same public spelling in both.
    out = x.linear(weight.transpose(), bias)
    out = out.relu()
    return {"physical": out.uop, "logical": logical(out)}


def case_mlp_mse():
    pred = Tensor.empty(2, 4, device="CPU").realize()
    target = Tensor.empty(2, 4, device="CPU").realize()
    diff = pred - target
    out = (diff * diff).mean()
    return {"physical": out.uop, "logical": logical(out)}


def case_mlp_dense_cross_entropy():
    logits = Tensor.empty(2, 4, device="CPU").realize()
    target = Tensor.empty(2, 4, device="CPU").realize()
    out = logits.cross_entropy(target)
    return {"physical": out.uop, "logical": logical(out)}


def case_bound_dense_cross_entropy():
    logits = realized_input(8, 10)
    labels = realized_input(8, 10)
    i = UOp.variable("i", 0, 4).bind(0) if ENGINE == "tinygrad" else Variable("i", 0, 4).bind(0)
    out = -(labels[i:i + 4] * logits[i:i + 4].log_softmax(axis=1)).sum(axis=1).mean()
    return {"physical": out.uop, "logical": logical(out)}


def case_nn_layernorm_apply():
    x = Tensor.empty(2, 4, device="CPU").realize()
    weight = Tensor.empty(4, device="CPU").realize()
    bias = Tensor.empty(4, device="CPU").realize()
    out = x.layernorm(eps=1e-5, axis=-1) * weight + bias
    return {"physical": out.uop, "logical": logical(out)}


def case_nn_rmsnorm_apply():
    x = Tensor.empty(2, 4, device="CPU").realize()
    weight = Tensor.empty(4, device="CPU").realize()
    xf = x.float()
    out = (xf * (xf.square().mean(-1, keepdim=True) + 1e-6).rsqrt()).cast(x.dtype) * weight
    return {"physical": out.uop, "logical": logical(out)}


def case_nn_embedding_apply():
    index = Tensor.empty(2, dtype="int32", device="CPU").realize()
    weight = Tensor.empty(4, 3, device="CPU").realize()
    arange = Tensor.arange(weight.shape[0]).to("CPU")
    out = (arange == index.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2)
    return {"physical": out.uop, "logical": logical(out)}


def case_quick_gelu_float32():
    out = typed_realized_empty("float32").quick_gelu()
    return {"physical": out.uop, "logical": logical(out)}


def typed_elementwise_occurrence(method, dtype):
    out = getattr(
        typed_realized_empty(dtype).to("CUDA").to("CPU"), method
    )()
    return {"physical": out.uop, "logical": logical(out)}


def case_gelu_occurrence():
    return typed_elementwise_occurrence("gelu", "float32")


def case_gelu_float16():
    return typed_elementwise_occurrence("gelu", "float16")


def case_gelu_float64():
    return typed_elementwise_occurrence("gelu", "float64")


def case_gelu_int32():
    return typed_elementwise_occurrence("gelu", "int32")


def case_gelu_bool():
    return typed_elementwise_occurrence("gelu", "bool")


def case_quick_gelu_float16():
    return typed_elementwise_occurrence("quick_gelu", "float16")


def case_rsqrt_float32():
    out = typed_realized_empty("float32").rsqrt()
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_axes_float32():
    out = typed_realized_empty("float32").sum(axis=(0, 1))
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_axis_float16():
    out = typed_realized_empty("float16").sum(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_axis_int16():
    out = typed_realized_empty("int16").sum(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_axis_keepdim_float32():
    out = realized_empty(2, 3, 4).sum(axis=1, keepdim=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_singleton_float32():
    out = realized_empty(2, 1, 4).sum(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_singleton_keepdim_float32():
    out = realized_empty(2, 1, 4).sum(axis=1, keepdim=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_sum_mixed_singleton_float32():
    out = realized_empty(2, 1, 4).sum(axis=(0, 1))
    return {"physical": out.uop, "logical": logical(out)}


def case_mean_axis_float32():
    out = typed_realized_empty("float32").mean(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_mean_axis_float16():
    out = typed_realized_empty("float16").mean(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_mean_axis_int32():
    out = typed_realized_empty("int32").mean(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_mean_axes_float32():
    out = typed_realized_empty("float32").mean(axis=(0, 1))
    return {"physical": out.uop, "logical": logical(out)}


def case_div_scalar_float32():
    out = typed_realized_empty("float32").div(2)
    return {"physical": out.uop, "logical": logical(out)}


def case_div_scalar_int32():
    out = typed_realized_empty("int32").div(2)
    return {"physical": out.uop, "logical": logical(out)}


def case_rdiv_scalar_float32():
    out = 2.0 / typed_realized_empty("float32")
    return {"physical": out.uop, "logical": logical(out)}


def case_max_axes_float32():
    out = realized_empty(2, 3, 4).max(axis=(1, 2))
    return {"physical": out.uop, "logical": logical(out)}


def min_axes(dtype, keepdim=False):
    out = Tensor.empty(2, 3, dtype=dtype).realize().min(axis=-1, keepdim=keepdim)
    return {"physical": out.uop, "logical": logical(out)}


def case_min_axes_uint8(): return min_axes("uint8")
def case_min_axes_int8(): return min_axes("int8")
def case_min_axes_uint64(): return min_axes("uint64")
def case_min_axes_bool(): return min_axes("bool")
def case_min_axes_float32_keepdim(): return min_axes("float32", True)


def case_minimum_int8():
    out = typed_realized_empty("int8").minimum(typed_realized_empty("int8"))
    return {"physical": out.uop, "logical": logical(out)}


def case_minimum_uint64():
    out = typed_realized_empty("uint64").minimum(typed_realized_empty("uint64"))
    return {"physical": out.uop, "logical": logical(out)}


def case_minimum_bool():
    out = typed_realized_empty("bool").minimum(typed_realized_empty("bool"))
    return {"physical": out.uop, "logical": logical(out)}


def case_minimum_bool_scalar():
    out = typed_realized_empty("bool").minimum(True)
    return {"physical": out.uop, "logical": logical(out)}


def case_scatter_amin_uint8():
    base = Tensor.empty(1, 3, dtype="uint8").realize()
    indices = Tensor.empty(1, 3, dtype="int32").realize()
    src = Tensor.empty(1, 3, dtype="uint8").realize()
    out = base.scatter_reduce(1, indices, src, "amin")
    return {"physical": out.uop, "logical": logical(out)}


def scatter_amin_without_self(dtype):
    base = Tensor.empty(1, 2, dtype=dtype).realize()
    indices = Tensor.empty(1, 2, dtype="int32").realize()
    src = Tensor.empty(1, 2, dtype=dtype).realize()
    out = base.scatter_reduce(1, indices, src, "amin", include_self=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_scatter_amin_uint8_without_self(): return scatter_amin_without_self("uint8")
def case_scatter_amin_uint16_without_self(): return scatter_amin_without_self("uint16")
def case_scatter_amin_uint32_without_self(): return scatter_amin_without_self("uint32")
def case_scatter_amin_uint64_without_self(): return scatter_amin_without_self("uint64")


def case_dot_float32():
    out = realized_empty(2, 3).dot(realized_empty(3, 4))
    return {"physical": out.uop, "logical": logical(out)}


def case_dot_singleton_batch():
    out = realized_empty(1, 12).dot(realized_empty(12, 10))
    return {"physical": out.uop, "logical": logical(out)}


def case_dot_float16_default():
    out = Tensor.empty(2, 3, device="CPU", dtype=dtypes.float16).realize().dot(
        Tensor.empty(3, 2, device="CPU", dtype=dtypes.float16).realize()
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_dot_float16_acc_float32():
    out = Tensor.empty(2, 3, device="CPU", dtype=dtypes.float16).realize().dot(
        Tensor.empty(3, 2, device="CPU", dtype=dtypes.float16).realize(),
        dtype=dtypes.float32,
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_dot_mixed_float16_float32():
    out = Tensor.empty(2, 3, device="CPU", dtype=dtypes.float16).realize().dot(
        Tensor.empty(3, 2, device="CPU", dtype=dtypes.float32).realize()
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_dot_occurrence():
    x = realized_empty(2, 2)
    out = x.dot(x.to("CUDA").to("CPU"))
    return {"physical": out.uop, "logical": logical(out)}


def case_minimum_float32():
    out = realized_empty(2, 2).minimum(realized_empty(2, 2))
    return {"physical": out.uop, "logical": logical(out)}


def case_minimum_occurrence():
    x = realized_empty(2, 2)
    out = x.minimum(x.to("CUDA").to("CPU"))
    return {"physical": out.uop, "logical": logical(out)}


def case_clamp_occurrence():
    x = realized_empty(2, 2)
    out = x.to("CUDA").to("CPU").clamp(-1.0, 1.0)
    return {"physical": out.uop, "logical": logical(out)}


def case_clamp_min_only():
    out = realized_empty(2, 2).clamp(min_=-1.0)
    return {"physical": out.uop, "logical": logical(out)}


def case_clamp_max_only():
    out = realized_empty(2, 2).clamp(max_=1.0)
    return {"physical": out.uop, "logical": logical(out)}


def case_clamp_int_float_bounds():
    out = typed_realized_empty("int32").clamp(-1.5, 1.5)
    return {"physical": out.uop, "logical": logical(out)}


def case_sort_float32():
    values, indices = realized_empty(4).sort()
    out = values + indices.cast(values.dtype)
    return {"physical": out.uop, "logical": logical(out)}


def case_topk_occurrence():
    moved = realized_empty(4).to("CUDA").to("CPU")
    values, indices = moved.topk(2)
    out = values + indices.cast(values.dtype)
    return {"physical": out.uop, "logical": logical(out)}


def case_argmax_occurrence():
    moved = realized_empty(2, 3).to("CUDA").to("CPU")
    out = moved.argmax(axis=1, keepdim=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_argmax_singleton_occurrence():
    moved = realized_empty(2, 1, 3).to("CUDA").to("CPU")
    out = moved.argmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_argmax_zero_extent():
    out = realized_empty(2, 0, 3).argmax(axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_pool2d_float32():
    out = realized_empty(1, 1, 4, 4)._pool((2, 2), stride=2)
    return {"physical": out.uop, "logical": logical(out)}


def case_max_pool2d_float32():
    out = realized_empty(1, 1, 4, 4).max_pool2d((2, 2), stride=2)
    return {"physical": out.uop, "logical": logical(out)}


def case_conv2d_occurrence():
    x = realized_empty(1, 1, 2, 2)
    moved_weight = x.to("CUDA").to("CPU")
    out = x.conv2d(moved_weight)
    return {"physical": out.uop, "logical": logical(out)}


def case_conv2d_float32():
    x = realized_empty(1, 2, 4, 4)
    weight = realized_empty(3, 2, 2, 2)
    bias = realized_empty(3)
    out = x.conv2d(weight, bias, stride=1, padding=0)
    return {"physical": out.uop, "logical": logical(out)}


def case_batchnorm_occurrence():
    x = realized_empty(1, 3, 1, 1)
    stat = realized_empty(3)
    moved_invstd = stat.to("CUDA").to("CPU")
    out = x.batchnorm(None, None, stat, moved_invstd, axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_batchnorm_float32():
    x = realized_empty(1, 3, 2, 2)
    weight = realized_empty(3)
    bias = realized_empty(3)
    mean = realized_empty(3)
    invstd = realized_empty(3)
    out = x.batchnorm(weight, bias, mean, invstd, axis=1)
    return {"physical": out.uop, "logical": logical(out)}


def case_gather_float32():
    x = realized_empty(2, 3)
    index = Tensor.empty(2, 2, dtype="int32", device="CPU").realize()
    out = x.gather(1, index)
    return {"physical": out.uop, "logical": logical(out)}


def case_gather_float16():
    x = typed_realized_empty("float16")
    index = Tensor.empty(2, 1, dtype="int32", device="CPU").realize()
    out = x.gather(1, index)
    return {"physical": out.uop, "logical": logical(out)}


def case_gather_moved_index():
    x = realized_empty(2, 3)
    index = Tensor.empty(2, 2, dtype="int32", device="CPU").realize()
    out = x.gather(1, index.to("CUDA").to("CPU"))
    return {"physical": out.uop, "logical": logical(out)}


def case_gather_occurrence():
    x = Tensor.empty(2, 2, dtype="int32", device="CPU").realize()
    out = x.gather(1, x.to("CUDA").to("CPU"))
    return {"physical": out.uop, "logical": logical(out)}


def case_index_select_float32():
    x = realized_empty(3, 4)
    index = Tensor.empty(2, dtype="int32", device="CPU").realize()
    out = x[index]
    return {"physical": out.uop, "logical": logical(out)}


def case_index_select_float16():
    x = Tensor.empty(3, 4, dtype="float16", device="CPU").realize()
    index = Tensor.empty(2, dtype="int32", device="CPU").realize()
    out = x[index]
    return {"physical": out.uop, "logical": logical(out)}


def case_index_select_moved_index():
    x = realized_empty(3, 4)
    index = Tensor.empty(2, dtype="int32", device="CPU").realize()
    out = x[index.to("CUDA").to("CPU")]
    return {"physical": out.uop, "logical": logical(out)}


def case_one_hot_int32():
    x = Tensor.empty(2, dtype="int32", device="CPU").realize()
    out = x.one_hot(4)
    return {"physical": out.uop, "logical": logical(out)}


def case_empty_storage():
    out = Tensor.empty(2, 3, device="CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_host_list_cpu():
    out = Tensor([1.0, 2.0], device="CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_host_list_cuda():
    out = Tensor([1.0, 2.0], device="CUDA")
    return {"physical": out.uop, "logical": logical(out)}


def case_host_list_2d_cpu():
    out = Tensor([[1.0, 2.0], [3.0, 4.0]], device="CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_host_list_2d_cuda():
    out = Tensor([[1.0, 2.0], [3.0, 4.0]], device="CUDA")
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_bool():
    out = Tensor(True)
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_float():
    out = Tensor(1.5)
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_float_cuda():
    out = Tensor(1.5, device="CUDA")
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_int():
    out = Tensor(7)
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_int_cuda():
    out = Tensor(7, device="CUDA")
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_where_broadcast():
    cond = Tensor.empty(2, 3, dtype="bool", device="CPU")
    x = Tensor.empty(2, 3, dtype="float32", device="CPU")
    out = cond.where(x, Tensor(0.0))
    return {"physical": out.uop, "logical": logical(out)}


def case_where_reduce_mixed_dtype():
    # Pinned Tensor.where (tensor.py:750-772) broadcasts x/y through
    # Tensor._broadcasted, including the int32 -> float32 promotion, before
    # UOp.where. Keep integer fill values: that promotion is the subject.
    x = Tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ],
        device="CPU",
    )
    out = (Tensor.full((4, 4), 7, device="CPU") > x).where(
        x, Tensor.full((4, 4), -2, device="CPU")
    ).sum(axis=0)
    return {"physical": out.uop, "logical": logical(out)}


def case_movement_reduce():
    x = realized_empty(6).reshape(2, 3)
    out = x.permute(1, 0).sum(axis=1, keepdim=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_reshape():
    out = realized_empty(2, 3).reshape(3, 2)
    return {"physical": out.uop, "logical": logical(out)}


def case_expand():
    out = realized_empty(1, 3).expand(2, 3)
    return {"physical": out.uop, "logical": logical(out)}


def case_expand_rank_add():
    out = realized_empty(3).expand(2, 3)
    return {"physical": out.uop, "logical": logical(out)}


def case_expand_middle_axis():
    out = realized_empty(2, 1, 3).expand(2, 4, 3)
    return {"physical": out.uop, "logical": logical(out)}


def case_const_like_float_shape():
    out = realized_empty(2, 2).const_like(True)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad():
    out = realized_empty(2, 3).pad(((1, 0), (0, 2)))
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_negative():
    out = realized_empty(3, 4).pad(((-1, 2), (1, -1)))
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_noop():
    out = realized_empty(2, 3).pad(((0, 0), (0, 0)))
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_value():
    out = realized_empty(3).pad(((1, 2),), value=5)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_int_source_int_value():
    out = Tensor.empty(3, dtype="int32", device="CPU").realize().pad(((1, 2),), value=5)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_int_source_float_value():
    out = Tensor.empty(3, dtype="int32", device="CPU").realize().pad(((1, 2),), value=5.5)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_int_source_bool_value():
    out = Tensor.empty(3, dtype="int32", device="CPU").realize().pad(((1, 2),), value=True)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_scalar_value():
    out = realized_empty().pad((), value=5)
    return {"physical": out.uop, "logical": logical(out)}


def case_pad_scalar_noop():
    out = realized_empty().pad(())
    return {"physical": out.uop, "logical": logical(out)}


def case_shrink():
    out = realized_empty(2, 3).shrink(((0, 1), (1, 3)))
    return {"physical": out.uop, "logical": logical(out)}


def case_shrink_noop():
    out = realized_empty(2, 3).shrink(((0, 2), (0, 3)))
    return {"physical": out.uop, "logical": logical(out)}


def case_permute():
    out = realized_empty(2, 3).permute(1, 0)
    return {"physical": out.uop, "logical": logical(out)}


def case_rearrange_flatten():
    out = Tensor.arange(6).reshape(2, 3).rearrange("h w -> (h w)")
    return {"physical": out.uop, "logical": logical(out)}


def case_rearrange_unflatten():
    out = Tensor.arange(6).rearrange("(h w) -> h w", h=2, w=3)
    return {"physical": out.uop, "logical": logical(out)}


def case_rearrange_permute():
    out = Tensor.arange(24).reshape(2, 3, 4).rearrange("b c h -> h b c")
    return {"physical": out.uop, "logical": logical(out)}


def case_rearrange_combined():
    out = Tensor.arange(24).reshape(2, 12).rearrange(
        "b (h w) -> (b h) w", h=3, w=4,
    )
    return {"physical": out.uop, "logical": logical(out)}


def case_flip():
    out = realized_empty(2, 3).flip((1,))
    return {"physical": out.uop, "logical": logical(out)}


def case_shrink_scalar_noop():
    out = realized_empty().shrink(())
    return {"physical": out.uop, "logical": logical(out)}


def case_flip_scalar_noop():
    out = realized_empty().flip(())
    return {"physical": out.uop, "logical": logical(out)}


def case_rebuilt_after_realize():
    a, b = realized_input(2), realized_input(2)
    x = a + b
    x.realize()
    y = a + b
    out = x + y
    return {"physical": out.uop, "logical": logical(out)}


def case_roundtrip_occurrence():
    x = realized_empty(2)
    moved = x.to("CUDA").to("CPU")
    out = x + moved
    return {"physical": out.uop, "logical": logical(out)}


def case_moved_assign_occurrence():
    x = Tensor.empty(2, device="CPU")
    moved = x.to("CUDA")
    moved.assign(Tensor.empty(2, device="CUDA"))
    out = x + moved.to("CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_moved_broadcast_assign_occurrence():
    target = Tensor.empty(2, 3, device="CPU")
    value = Tensor.empty(3, device="CPU").to("CUDA").to("CPU")
    target.assign(value)
    return {"physical": target.uop, "logical": logical(target)}


def case_sibling_moves():
    x = Tensor.empty(2, device="CPU")
    cuda = x.to("CUDA").to("CPU")
    hip = x.to("HIP").to("CPU")
    out = cuda + hip
    return {"physical": out.uop, "logical": logical(out)}


def case_clone():
    out = realized_empty(2).clone()
    return {"physical": out.uop, "logical": logical(out)}


def case_clone_deviceless_source():
    # Pinned UOp.clone (uop/ops.py:765-769) keys the COPY decision on the
    # source UOp device, not frontend/default-device metadata. arange remains
    # device-free even when a preferred construction device is supplied.
    source = Tensor(Tensor.arange(4).uop, device="CUDA")
    out = source.clone("CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_contiguous():
    out = Tensor.empty(2, 3, device="CPU").permute(1, 0).contiguous()
    return {"physical": out.uop, "logical": logical(out)}


def case_contiguous_deviceless():
    out = Tensor.arange(4).contiguous()
    return {"physical": out.uop, "logical": logical(out)}


def case_view_assign():
    x = Tensor.empty(4, device="CPU")
    x.shrink(((1, 3),)).assign(Tensor.empty(2, device="CPU"))
    return {"physical": x.uop, "logical": logical(x)}


def case_full_const_pair():
    a = Tensor.full((2, 3), 2.0, buffer=False)
    b = Tensor.full((2, 3), 2.0, buffer=False)
    out = a + b
    return {"physical": out.uop, "logical": logical(out)}


def case_full_singleton():
    out = Tensor.full((1,), 2, dtype="int32", buffer=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_arange():
    out = Tensor.arange(0, 4, 1)
    return {"physical": out.uop, "logical": logical(out)}


def case_arange_singleton():
    out = Tensor.arange(1, dtype="int32")
    return {"physical": out.uop, "logical": logical(out)}


def case_arange_empty():
    out = Tensor.arange(0, 0, -1, dtype="int32")
    return {"physical": out.uop, "logical": logical(out)}


def case_arange_to_cuda():
    out = Tensor.arange(0, 4, 1).to("CUDA")
    return {"physical": out.uop, "logical": logical(out)}


def case_linspace():
    out = Tensor.linspace(0, 1, 4)
    return {"physical": out.uop, "logical": logical(out)}


def case_eye():
    out = Tensor.eye(3)
    return {"physical": out.uop, "logical": logical(out)}


def case_zero_broadcast():
    a = Tensor.full((0, 3), 1.0, buffer=False)
    b = Tensor.full((1, 3), 2.0, buffer=False)
    out = a + b
    return {"physical": out.uop, "logical": logical(out)}


def case_internal_scalar_add():
    out = Tensor.empty(2, dtype="int32", device="CPU") + 1
    return {"physical": out.uop, "logical": logical(out)}


def case_integer_index_scalar():
    out = Tensor.arange(2, dtype="int32")[0]
    return {"physical": out.uop, "logical": logical(out)}


def case_scalar_squeeze():
    out = Tensor(7).squeeze()
    return {"physical": out.uop, "logical": logical(out)}


def case_uint_neg():
    out = Tensor.arange(2, dtype="uint32").neg()
    return {"physical": out.uop, "logical": logical(out)}


def case_uint_sub():
    out = Tensor.arange(2, dtype="uint32") - 1
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_single_draw():
    Tensor.manual_seed(123)
    out = Tensor.rand(4, device="CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_two_draw():
    Tensor.manual_seed(123)
    draws = [Tensor.rand(4, device="CPU") for _ in range(2)]
    out = draws[0] + draws[1]
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_float16():
    Tensor.manual_seed(123)
    out = Tensor.rand(4, device="CPU", dtype="float16", contiguous=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_float64():
    Tensor.manual_seed(123)
    out = Tensor.rand(4, device="CPU", dtype="float64", contiguous=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_bfloat16():
    Tensor.manual_seed(123)
    out = Tensor.rand(4, device="CPU", dtype="bfloat16", contiguous=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_rng_zero_extent():
    Tensor.manual_seed(123)
    out = Tensor.rand(0, device="CPU", contiguous=False)
    return {"physical": out.uop, "logical": logical(out)}


def case_randn_float32():
    Tensor.manual_seed(123)
    out = Tensor.randn(4, device="CPU")
    return {"physical": out.uop, "logical": logical(out)}


def case_bitcast_widen_u8_u32():
    out = Tensor.full((8,), 1, dtype="uint8").bitcast("uint32")
    return {"physical": out.uop, "logical": logical(out)}


def case_bitcast_narrow_u32_u8():
    out = Tensor.full((2,), 1, dtype="uint32").bitcast("uint8")
    return {"physical": out.uop, "logical": logical(out)}


def case_dropout_stateful_rng():
    x = realized_input(2, 2)
    Tensor.manual_seed(11)
    with training_context():
        out = x.dropout(0.25)
    return {"physical": out.uop, "logical": logical(out)}


def raw_gradient(loss, target):
    if ENGINE == "tinygrad":
        return loss.gradient(target)[0].uop
    from polygrad.tensor import _uop_wrap

    grad = Tensor._grad_many_raw(
        target._ctx,
        Tensor._core_uop_raw(loss._tensor),
        None,
        (Tensor._core_uop_raw(target._tensor),),
    )[0]
    return _uop_wrap(target._ctx, grad)


def case_grad_square():
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    out = raw_gradient((x * x).sum(), x)
    return {"physical": out}


def case_grad_scale():
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    out = raw_gradient((x * 2).sum(), x)
    return {"physical": out}


def primitive_gradient(method, values=(-2.5, -1.0, 0.5, 1.0, 2.5)):
    x = Tensor(list(values))
    return {"physical": raw_gradient(method(x).sum(), x)}


def case_grad_reciprocal():
    return primitive_gradient(lambda x: x.reciprocal())


def case_grad_sin():
    return primitive_gradient(lambda x: x.sin())


def case_grad_log2():
    return primitive_gradient(lambda x: x.log2(), (0.5, 1.0, 2.0, 4.0, 8.0))


def case_grad_exp2():
    return primitive_gradient(lambda x: x.exp2())


def case_grad_sqrt():
    return primitive_gradient(lambda x: x.sqrt(), (0.25, 1.0, 4.0, 9.0, 16.0))


def case_grad_max_axes():
    x = realized_input(2, 3)
    return {"physical": raw_gradient(x.max(axis=1).sum(), x)}


def case_grad_expand_reduce():
    x = realized_empty(1)
    expanded = x.reshape(()).reshape((1,)).expand((5,))
    return {"physical": raw_gradient(expanded.sum(), x)}


def case_grad_no_path_zero():
    x = realized_empty(5)
    y = realized_empty(5)
    return {"physical": raw_gradient(x.sum(), y)}


def case_reshape_scalar_roundtrip_expand():
    x = realized_empty(1)
    return {"physical": x.reshape(()).reshape((1,)).expand((5,)).uop}


def case_nn_conv2d_initializer():
    Tensor.manual_seed(201)
    conv = nn.Conv2d(4, 8, 3, bias=False)
    return {"physical": conv.weight.uop, "logical": logical(conv.weight)}


def case_nn_linear_initializer():
    Tensor.manual_seed(201)
    linear = nn.Linear(8, 3)
    return {"physical": linear.weight.uop, "logical": logical(linear.weight)}


def case_nn_batchnorm_initializer():
    norm = nn.BatchNorm(8)
    return {"physical": norm.weight.uop, "logical": logical(norm.weight)}


def scatter_inputs(generalized=False):
    if generalized:
        self = Tensor.empty(2, 5, dtype="float32", device="CPU").realize()
        index = Tensor.empty(1, 3, dtype="int32", device="CPU").realize()
        src = Tensor.empty(2, 4, dtype="float32", device="CPU").realize()
    else:
        self = Tensor.empty(1, 5, dtype="float32", device="CPU").realize()
        index = Tensor.empty(1, 5, dtype="int32", device="CPU").realize()
        src = Tensor.empty(1, 5, dtype="float32", device="CPU").realize()
    return self, index, src


def scatter_graph(reduce=None, include_self=True, generalized=False):
    self, index, src = scatter_inputs(generalized)
    out = self.scatter(1, index, src) if reduce is None else \
        self.scatter_reduce(1, index, src, reduce, include_self=include_self)
    return {"physical": out.uop, "logical": logical(out)}


def case_scatter_basic(): return scatter_graph()
def case_scatter_generalized(): return scatter_graph(generalized=True)
def case_scatter_sum_self(): return scatter_graph("sum")
def case_scatter_sum_no_self(): return scatter_graph("sum", False, True)
def case_scatter_prod_self(): return scatter_graph("prod")
def case_scatter_prod_no_self(): return scatter_graph("prod", False, True)
def case_scatter_mean_self(): return scatter_graph("mean")
def case_scatter_mean_no_self(): return scatter_graph("mean", False, True)
def case_scatter_amax_self(): return scatter_graph("amax")
def case_scatter_amax_no_self(): return scatter_graph("amax", False, True)
def case_scatter_amin_self(): return scatter_graph("amin")
def case_scatter_amin_no_self(): return scatter_graph("amin", False, True)


def case_grad_ceil():
    x = Tensor([-1.25, 0.25, 1.75])
    out = raw_gradient(x.ceil().sum(), x)
    return {"physical": out}


def case_grad_floor():
    x = Tensor([-1.25, 0.25, 1.75])
    out = raw_gradient(x.floor().sum(), x)
    return {"physical": out}


def composite_gradient(method):
    x = Tensor([-2.5, -1.0, 0.0, 0.5, 2.5])
    return {"physical": raw_gradient(method(x).sum(), x)}


def case_grad_sigmoid():
    return composite_gradient(lambda x: x.sigmoid())


def case_grad_tanh():
    return composite_gradient(lambda x: x.tanh())


def case_grad_gelu():
    return composite_gradient(lambda x: x.gelu())


def case_grad_quick_gelu():
    return composite_gradient(lambda x: x.quick_gelu())


def case_grad_pow3():
    return composite_gradient(lambda x: x**3)


def pow_tensor_gradient(target_index):
    base = Tensor([0.5, 1.0, 2.0, 4.0])
    exponent = Tensor([3.0, 2.0, 0.5, -1.0])
    targets = (base, exponent)
    return {
        "physical": raw_gradient(
            (base**exponent).sum(), targets[target_index]
        )
    }


def case_grad_pow_base():
    return pow_tensor_gradient(0)


def case_grad_pow_exponent():
    return pow_tensor_gradient(1)


def pow_broadcast_gradient(target_index):
    base = realized_empty(2, 1)
    exponent = realized_empty(1, 3)
    targets = (base, exponent)
    return {"physical": raw_gradient((base**exponent).sum(), targets[target_index])}


def case_grad_pow_broadcast_base():
    return pow_broadcast_gradient(0)


def case_grad_pow_broadcast_exponent():
    return pow_broadcast_gradient(1)


def case_grad_silu():
    return composite_gradient(lambda x: x.silu())


def case_grad_swish():
    return composite_gradient(lambda x: x.swish())


def case_grad_elu():
    return composite_gradient(lambda x: x.elu())


def case_grad_relu6():
    return composite_gradient(lambda x: x.relu6())


def case_grad_leaky_relu():
    return composite_gradient(lambda x: x.leaky_relu())


def case_grad_hardswish():
    return composite_gradient(lambda x: x.hardswish())


def case_grad_hardsigmoid():
    return composite_gradient(lambda x: x.hardsigmoid())


def case_grad_hardtanh():
    return composite_gradient(lambda x: x.hardtanh())


def case_backward_square():
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    (x * x).sum().backward()
    out = x.grad.uop if ENGINE == "tinygrad" else x.grad.uop_physical
    return {"physical": out}


def optimizer_step_graph(construct):
    param = Tensor([1.0, 2.0], device="CPU").is_param_()
    param.realize()
    optim = construct(param)
    with training_context():
        (param * param).sum().backward()
        scheduled = optim.schedule_step()
    if ENGINE == "polygrad":
        roots = [tensor.uop_physical for tensor in scheduled]
        if any(root is None for root in roots):
            raise RuntimeError(
                "Default execution requires stored physical optimizer roots; "
                f"observed {[root is not None for root in roots]}"
            )
    out = scheduled[0]
    for tensor in scheduled[1:]:
        out = out + tensor
    return {"physical": out.uop if ENGINE == "tinygrad" else out.uop_physical}


def case_sgd_step():
    return optimizer_step_graph(lambda param: SGD([param], lr=0.1, fused=False))


def case_sgd_momentum_step():
    return optimizer_step_graph(
        lambda param: SGD([param], lr=0.1, momentum=0.9, fused=False)
    )


def case_adam_step():
    return optimizer_step_graph(lambda param: Adam([param], lr=0.1, fused=False))


def case_adamw_step():
    return optimizer_step_graph(
        lambda param: AdamW([param], lr=0.1, weight_decay=0.01, fused=False)
    )


CASES = {
    "adam_step": ("optimizer", case_adam_step),
    "adamw_step": ("optimizer", case_adamw_step),
    "argmax_occurrence": ("tensor", case_argmax_occurrence),
    "argmax_singleton_occurrence": ("tensor", case_argmax_singleton_occurrence),
    "argmax_zero_extent": ("tensor", case_argmax_zero_extent),
    "abs_bool": ("tensor", case_abs_bool),
    "abs_occurrence": ("tensor", case_abs_occurrence),
    "arange": ("tensor", case_arange),
    "arange_empty": ("tensor", case_arange_empty),
    "arange_singleton": ("tensor", case_arange_singleton),
    "arange_to_cuda": ("tensor", case_arange_to_cuda),
    "basic_alu": ("tensor", case_basic_alu),
    "batchnorm_float32": ("tensor", case_batchnorm_float32),
    "batchnorm_occurrence": ("tensor", case_batchnorm_occurrence),
    "bfloat16_host_add": ("tensor", case_bfloat16_host_add),
    "fp8e4m3_host_add": ("tensor", case_fp8e4m3_host_add),
    "fp8e5m2_host_add": ("tensor", case_fp8e5m2_host_add),
    "fp8e4m3fnuz_host_add": ("tensor", case_fp8e4m3fnuz_host_add),
    "fp8e5m2fnuz_host_add": ("tensor", case_fp8e5m2fnuz_host_add),
    "ceil_int32": ("tensor", case_ceil_int32),
    "ceil_occurrence": ("tensor", case_ceil_occurrence),
    "clone": ("tensor", case_clone),
    "clone_deviceless_source": ("tensor", case_clone_deviceless_source),
    "clamp_int_float_bounds": ("tensor", case_clamp_int_float_bounds),
    "clamp_max_only": ("tensor", case_clamp_max_only),
    "clamp_min_only": ("tensor", case_clamp_min_only),
    "clamp_occurrence": ("tensor", case_clamp_occurrence),
    "contiguous": ("tensor", case_contiguous),
    "contiguous_deviceless": ("tensor", case_contiguous_deviceless),
    "conv2d_float32": ("tensor", case_conv2d_float32),
    "const_like_float_shape": ("tensor", case_const_like_float_shape),
    "conv2d_occurrence": ("tensor", case_conv2d_occurrence),
    "cos_bfloat16": ("tensor", case_cos_bfloat16),
    "cos_float16": ("tensor", case_cos_float16),
    "cos_float64": ("tensor", case_cos_float64),
    "cos_int32": ("tensor", case_cos_int32),
    "cos_occurrence": ("tensor", case_cos_occurrence),
    "cos_uint64": ("tensor", case_cos_uint64),
    "detach_float32": ("tensor", case_detach_float32),
    "dot_float32": ("tensor", case_dot_float32),
    "dot_singleton_batch": ("tensor", case_dot_singleton_batch),
    "dot_float16_default": ("tensor", case_dot_float16_default),
    "dot_float16_acc_float32": ("tensor", case_dot_float16_acc_float32),
    "dot_mixed_float16_float32": ("tensor", case_dot_mixed_float16_float32),
    "dot_occurrence": ("tensor", case_dot_occurrence),
    "empty_storage": ("tensor", case_empty_storage),
    "exp_float16": ("tensor", case_exp_float16),
    "exp_float32": ("tensor", case_exp_float32),
    "exp2_int32": ("tensor", case_exp2_int32),
    "exp_int32": ("tensor", case_exp_int32),
    "eye": ("tensor", case_eye),
    "elu_occurrence": ("tensor", case_elu_occurrence),
    "expand": ("tensor", case_expand),
    "expand_middle_axis": ("tensor", case_expand_middle_axis),
    "expand_rank_add": ("tensor", case_expand_rank_add),
    "flip": ("tensor", case_flip),
    "flip_scalar_noop": ("tensor", case_flip_scalar_noop),
    "floor_occurrence": ("tensor", case_floor_occurrence),
    "gelu_bool": ("tensor", case_gelu_bool),
    "gelu_float16": ("tensor", case_gelu_float16),
    "gelu_float64": ("tensor", case_gelu_float64),
    "gelu_int32": ("tensor", case_gelu_int32),
    "gelu_occurrence": ("tensor", case_gelu_occurrence),
    "full_const_pair": ("tensor", case_full_const_pair),
    "full_singleton": ("tensor", case_full_singleton),
    "gather_float16": ("tensor", case_gather_float16),
    "gather_float32": ("tensor", case_gather_float32),
    "gather_moved_index": ("tensor", case_gather_moved_index),
    "gather_occurrence": ("tensor", case_gather_occurrence),
    "backward_square": ("tensor", case_backward_square),
    "grad_ceil": ("tensor", case_grad_ceil),
    "grad_elu": ("tensor", case_grad_elu),
    "grad_exp2": ("tensor", case_grad_exp2),
    "grad_floor": ("tensor", case_grad_floor),
    "grad_gelu": ("tensor", case_grad_gelu),
    "grad_hardsigmoid": ("tensor", case_grad_hardsigmoid),
    "grad_hardswish": ("tensor", case_grad_hardswish),
    "grad_hardtanh": ("tensor", case_grad_hardtanh),
    "grad_leaky_relu": ("tensor", case_grad_leaky_relu),
    "grad_log2": ("tensor", case_grad_log2),
    "grad_expand_reduce": ("tensor", case_grad_expand_reduce),
    "grad_no_path_zero": ("tensor", case_grad_no_path_zero),
    "reshape_scalar_roundtrip_expand": ("tensor", case_reshape_scalar_roundtrip_expand),
    "grad_max_axes": ("tensor", case_grad_max_axes),
    "nn_batchnorm_initializer": ("tensor", case_nn_batchnorm_initializer),
    "nn_conv2d_initializer": ("tensor", case_nn_conv2d_initializer),
    "nn_linear_initializer": ("tensor", case_nn_linear_initializer),
    "grad_pow3": ("tensor", case_grad_pow3),
    "grad_pow_base": ("tensor", case_grad_pow_base),
    "grad_pow_broadcast_base": ("tensor", case_grad_pow_broadcast_base),
    "grad_pow_broadcast_exponent": ("tensor", case_grad_pow_broadcast_exponent),
    "grad_pow_exponent": ("tensor", case_grad_pow_exponent),
    "grad_quick_gelu": ("tensor", case_grad_quick_gelu),
    "grad_relu6": ("tensor", case_grad_relu6),
    "grad_reciprocal": ("tensor", case_grad_reciprocal),
    "grad_scale": ("tensor", case_grad_scale),
    "grad_sigmoid": ("tensor", case_grad_sigmoid),
    "grad_sin": ("tensor", case_grad_sin),
    "grad_silu": ("tensor", case_grad_silu),
    "grad_sqrt": ("tensor", case_grad_sqrt),
    "grad_square": ("tensor", case_grad_square),
    "grad_swish": ("tensor", case_grad_swish),
    "grad_tanh": ("tensor", case_grad_tanh),
    "host_list_2d_cpu": ("tensor", case_host_list_2d_cpu),
    "host_list_2d_cuda": ("tensor", case_host_list_2d_cuda),
    "host_list_cpu": ("tensor", case_host_list_cpu),
    "host_list_cuda": ("tensor", case_host_list_cuda),
    "hardsigmoid_custom": ("tensor", case_hardsigmoid_custom),
    "hardsigmoid_occurrence": ("tensor", case_hardsigmoid_occurrence),
    "hardswish_occurrence": ("tensor", case_hardswish_occurrence),
    "hardtanh_occurrence": ("tensor", case_hardtanh_occurrence),
    "internal_scalar_add": ("tensor", case_internal_scalar_add),
    "integer_index_scalar": ("tensor", case_integer_index_scalar),
    "isinf_bool_both": ("tensor", case_isinf_bool_both),
    "isinf_bool_negative": ("tensor", case_isinf_bool_negative),
    "isinf_bool_neither": ("tensor", case_isinf_bool_neither),
    "isinf_bool_positive": ("tensor", case_isinf_bool_positive),
    "isinf_float16_both": ("tensor", case_isinf_float16_both),
    "isinf_float16_negative": ("tensor", case_isinf_float16_negative),
    "isinf_float16_neither": ("tensor", case_isinf_float16_neither),
    "isinf_float16_positive": ("tensor", case_isinf_float16_positive),
    "isinf_float32_both": ("tensor", case_isinf_float32_both),
    "isinf_float32_negative": ("tensor", case_isinf_float32_negative),
    "isinf_float32_neither": ("tensor", case_isinf_float32_neither),
    "isinf_float32_positive": ("tensor", case_isinf_float32_positive),
    "isinf_float64_both": ("tensor", case_isinf_float64_both),
    "isinf_float64_negative": ("tensor", case_isinf_float64_negative),
    "isinf_float64_neither": ("tensor", case_isinf_float64_neither),
    "isinf_float64_positive": ("tensor", case_isinf_float64_positive),
    "isinf_int32_both": ("tensor", case_isinf_int32_both),
    "isinf_int32_negative": ("tensor", case_isinf_int32_negative),
    "isinf_int32_neither": ("tensor", case_isinf_int32_neither),
    "isinf_int32_positive": ("tensor", case_isinf_int32_positive),
    "isinf_occurrence_both": ("tensor", case_isinf_occurrence_both),
    "isinf_occurrence_negative": ("tensor", case_isinf_occurrence_negative),
    "isinf_occurrence_neither": ("tensor", case_isinf_occurrence_neither),
    "isinf_occurrence_positive": ("tensor", case_isinf_occurrence_positive),
    "isnan_occurrence": ("tensor", case_isnan_occurrence),
    "index_select_float16": ("tensor", case_index_select_float16),
    "index_select_float32": ("tensor", case_index_select_float32),
    "index_select_moved_index": ("tensor", case_index_select_moved_index),
    "linspace": ("tensor", case_linspace),
    "leaky_relu_occurrence": ("tensor", case_leaky_relu_occurrence),
    "log_float16": ("tensor", case_log_float16),
    "log_float32": ("tensor", case_log_float32),
    "log_int32": ("tensor", case_log_int32),
    "log2_int32": ("tensor", case_log2_int32),
    "logaddexp_broadcast_occurrence": ("tensor", case_logaddexp_broadcast_occurrence),
    "logaddexp_scalar_occurrence": ("tensor", case_logaddexp_scalar_occurrence),
    "log_softmax_float16": ("tensor", case_log_softmax_float16),
    "log_softmax_float32": ("tensor", case_log_softmax_float32),
    "div_scalar_float32": ("tensor", case_div_scalar_float32),
    "div_scalar_int32": ("tensor", case_div_scalar_int32),
    "mean_axis_float32": ("tensor", case_mean_axis_float32),
    "mean_axis_float16": ("tensor", case_mean_axis_float16),
    "mean_axis_int32": ("tensor", case_mean_axis_int32),
    "mean_axes_float32": ("tensor", case_mean_axes_float32),
    "minimum_float32": ("tensor", case_minimum_float32),
    "minimum_occurrence": ("tensor", case_minimum_occurrence),
    "max_axes_float32": ("tensor", case_max_axes_float32),
    "min_axes_uint8": ("tensor", case_min_axes_uint8),
    "min_axes_int8": ("tensor", case_min_axes_int8),
    "min_axes_uint64": ("tensor", case_min_axes_uint64),
    "min_axes_bool": ("tensor", case_min_axes_bool),
    "min_axes_float32_keepdim": ("tensor", case_min_axes_float32_keepdim),
    "minimum_int8": ("tensor", case_minimum_int8),
    "minimum_bool": ("tensor", case_minimum_bool),
    "minimum_bool_scalar": ("tensor", case_minimum_bool_scalar),
    "scatter_amin_uint8_without_self": ("tensor", case_scatter_amin_uint8_without_self),
    "scatter_amin_uint16_without_self": ("tensor", case_scatter_amin_uint16_without_self),
    "scatter_amin_uint32_without_self": ("tensor", case_scatter_amin_uint32_without_self),
    "scatter_amin_uint64_without_self": ("tensor", case_scatter_amin_uint64_without_self),
    "minimum_uint64": ("tensor", case_minimum_uint64),
    "scatter_amin_uint8": ("tensor", case_scatter_amin_uint8),
    "max_pool2d_float32": ("tensor", case_max_pool2d_float32),
    "movement_reduce": ("tensor", case_movement_reduce),
    "mish_occurrence": ("tensor", case_mish_occurrence),
    "moved_broadcast_assign_occurrence": ("tensor", case_moved_broadcast_assign_occurrence),
    "moved_assign_occurrence": ("tensor", case_moved_assign_occurrence),
    "pad": ("tensor", case_pad),
    "pad_negative": ("tensor", case_pad_negative),
    "pad_noop": ("tensor", case_pad_noop),
    "pad_scalar_noop": ("tensor", case_pad_scalar_noop),
    "pad_scalar_value": ("tensor", case_pad_scalar_value),
    "pad_int_source_bool_value": ("tensor", case_pad_int_source_bool_value),
    "pad_int_source_float_value": ("tensor", case_pad_int_source_float_value),
    "pad_int_source_int_value": ("tensor", case_pad_int_source_int_value),
    "pad_value": ("tensor", case_pad_value),
    "one_hot_int32": ("tensor", case_one_hot_int32),
    "permute": ("tensor", case_permute),
    "pool2d_float32": ("tensor", case_pool2d_float32),
    "rebuilt_after_realize": ("realize", case_rebuilt_after_realize),
    "quick_gelu_float16": ("tensor", case_quick_gelu_float16),
    "quick_gelu_float32": ("tensor", case_quick_gelu_float32),
    "radd_named_mixed_broadcast": ("tensor", case_radd_named_mixed_broadcast),
    "radd_named_occurrence": ("tensor", case_radd_named_occurrence),
    "radd_operator_occurrence": ("tensor", case_radd_operator_occurrence),
    "rearrange_combined": ("tensor", case_rearrange_combined),
    "rearrange_flatten": ("tensor", case_rearrange_flatten),
    "rearrange_permute": ("tensor", case_rearrange_permute),
    "rearrange_unflatten": ("tensor", case_rearrange_unflatten),
    "reshape": ("tensor", case_reshape),
    "rmul_named_mixed_broadcast": ("tensor", case_rmul_named_mixed_broadcast),
    "rmul_named_occurrence": ("tensor", case_rmul_named_occurrence),
    "rmul_operator_occurrence": ("tensor", case_rmul_operator_occurrence),
    "round_bool": ("tensor", case_round_bool),
    "round_float16": ("tensor", case_round_float16),
    "round_float32": ("tensor", case_round_float32),
    "round_float64": ("tensor", case_round_float64),
    "round_int32": ("tensor", case_round_int32),
    "round_occurrence": ("tensor", case_round_occurrence),
    "reciprocal_int32": ("tensor", case_reciprocal_int32),
    "rdiv_scalar_float32": ("tensor", case_rdiv_scalar_float32),
    "roundtrip_occurrence": ("tensor", case_roundtrip_occurrence),
    "rng_single_draw": ("tensor", case_rng_single_draw),
    "rng_two_draw": ("tensor", case_rng_two_draw),
    "rng_float16": ("tensor", case_rng_float16),
    "rng_float64": ("tensor", case_rng_float64),
    "rng_bfloat16": ("tensor", case_rng_bfloat16),
    "rng_zero_extent": ("tensor", case_rng_zero_extent),
    "randn_float32": ("tensor", case_randn_float32),
    "bitcast_widen_u8_u32": ("tensor", case_bitcast_widen_u8_u32),
    "bitcast_narrow_u32_u8": ("tensor", case_bitcast_narrow_u32_u8),
    "dropout_stateful_rng": ("tensor", case_dropout_stateful_rng),
    "relu_float32": ("tensor", case_relu_float32),
    "mlp_dense_cross_entropy": ("tensor", case_mlp_dense_cross_entropy),
    "bound_dense_cross_entropy": ("tensor", case_bound_dense_cross_entropy),
    "mlp_linear_relu": ("tensor", case_mlp_linear_relu),
    "mlp_mse": ("tensor", case_mlp_mse),
    "nn_embedding_apply": ("tensor", case_nn_embedding_apply),
    "nn_layernorm_apply": ("tensor", case_nn_layernorm_apply),
    "nn_rmsnorm_apply": ("tensor", case_nn_rmsnorm_apply),
    "relu6_occurrence": ("tensor", case_relu6_occurrence),
    "rsqrt_float32": ("tensor", case_rsqrt_float32),
    "scalar_bool": ("tensor", case_scalar_bool),
    "scalar_float": ("tensor", case_scalar_float),
    "scalar_float_cuda": ("tensor", case_scalar_float_cuda),
    "scalar_int": ("tensor", case_scalar_int),
    "scalar_int_cuda": ("tensor", case_scalar_int_cuda),
    "scalar_squeeze": ("tensor", case_scalar_squeeze),
    "softplus_beta2_occurrence": ("tensor", case_softplus_beta2_occurrence),
    "softplus_occurrence": ("tensor", case_softplus_occurrence),
    "scalar_where_broadcast": ("tensor", case_scalar_where_broadcast),
    "scatter_basic": ("tensor", case_scatter_basic),
    "scatter_generalized": ("tensor", case_scatter_generalized),
    "scatter_sum_self": ("tensor", case_scatter_sum_self),
    "scatter_sum_no_self": ("tensor", case_scatter_sum_no_self),
    "scatter_prod_self": ("tensor", case_scatter_prod_self),
    "scatter_prod_no_self": ("tensor", case_scatter_prod_no_self),
    "scatter_mean_self": ("tensor", case_scatter_mean_self),
    "scatter_mean_no_self": ("tensor", case_scatter_mean_no_self),
    "scatter_amax_self": ("tensor", case_scatter_amax_self),
    "scatter_amax_no_self": ("tensor", case_scatter_amax_no_self),
    "scatter_amin_self": ("tensor", case_scatter_amin_self),
    "scatter_amin_no_self": ("tensor", case_scatter_amin_no_self),
    "sgd_momentum_step": ("optimizer", case_sgd_momentum_step),
    "sgd_step": ("optimizer", case_sgd_step),
    "sigmoid_float16": ("tensor", case_sigmoid_float16),
    "sigmoid_occurrence": ("tensor", case_sigmoid_occurrence),
    "sin_bfloat16": ("tensor", case_sin_bfloat16),
    "sin_bool": ("tensor", case_sin_bool),
    "sin_float64": ("tensor", case_sin_float64),
    "sin_int32": ("tensor", case_sin_int32),
    "sin_occurrence": ("tensor", case_sin_occurrence),
    "sin_uint64": ("tensor", case_sin_uint64),
    "sign_bool": ("tensor", case_sign_bool),
    "sign_float16": ("tensor", case_sign_float16),
    "sign_int32": ("tensor", case_sign_int32),
    "sign_occurrence": ("tensor", case_sign_occurrence),
    "sign_uint32": ("tensor", case_sign_uint32),
    "silu_occurrence": ("tensor", case_silu_occurrence),
    "softmax_float16": ("tensor", case_softmax_float16),
    "softmax_float32": ("tensor", case_softmax_float32),
    "sort_float32": ("tensor", case_sort_float32),
    "square_occurrence": ("tensor", case_square_occurrence),
    "sub_float32": ("tensor", case_sub_float32),
    "sum_axes_float32": ("tensor", case_sum_axes_float32),
    "sum_axis_float16": ("tensor", case_sum_axis_float16),
    "sum_axis_int16": ("tensor", case_sum_axis_int16),
    "sum_axis_keepdim_float32": ("tensor", case_sum_axis_keepdim_float32),
    "sum_singleton_float32": ("tensor", case_sum_singleton_float32),
    "sum_singleton_keepdim_float32": ("tensor", case_sum_singleton_keepdim_float32),
    "sum_mixed_singleton_float32": ("tensor", case_sum_mixed_singleton_float32),
    "tan_bfloat16": ("tensor", case_tan_bfloat16),
    "tan_float64": ("tensor", case_tan_float64),
    "tan_int32": ("tensor", case_tan_int32),
    "tan_occurrence": ("tensor", case_tan_occurrence),
    "tan_uint64": ("tensor", case_tan_uint64),
    "tanh_float16": ("tensor", case_tanh_float16),
    "tanh_occurrence": ("tensor", case_tanh_occurrence),
    "tril_occurrence": ("tensor", case_tril_occurrence),
    "triu_host": ("tensor", case_triu_host),
    "triu_nondefault_device": ("tensor", case_triu_nondefault_device),
    "topk_occurrence": ("tensor", case_topk_occurrence),
    "where_reduce_mixed_dtype": ("tensor", case_where_reduce_mixed_dtype),
    "sibling_moves": ("tensor", case_sibling_moves),
    "shrink": ("tensor", case_shrink),
    "shrink_noop": ("tensor", case_shrink_noop),
    "shrink_scalar_noop": ("tensor", case_shrink_scalar_noop),
    "sqrt_int32": ("tensor", case_sqrt_int32),
    "view_assign": ("tensor", case_view_assign),
    "uint_neg": ("tensor", case_uint_neg),
    "uint_sub": ("tensor", case_uint_sub),
    "zero_broadcast": ("tensor", case_zero_broadcast),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    args = parser.parse_args()
    selected = args.case or list(CASES)

    result = {"schema_version": 1, "engine": ENGINE, "cases": {}}
    for name in selected:
        stage, construct = CASES[name]
        roots = construct()
        result["cases"][name] = {
            "stage": stage,
            "roots": {root_name: canonical_graph(root) for root_name, root in roots.items()},
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
