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
    from tinygrad import Tensor
elif ENGINE == "polygrad":
    from polygrad import Tensor, _ffi
else:
    raise RuntimeError(f"unknown ENGINE={ENGINE!r}")

if ENGINE == "polygrad":
    class _PolyDType(ctypes.Structure):
        _fields_ = [
            ("priority", ctypes.c_int8),
            ("bitsize", ctypes.c_uint16),
            ("name", ctypes.c_char_p),
            ("fmt", ctypes.c_char),
            ("count", ctypes.c_uint16),
            ("is_ptr", ctypes.c_bool),
            ("addrspace", ctypes.c_int),
            ("vcount", ctypes.c_uint16),
            ("ptr_size", ctypes.c_int64),
        ]

    class _PolyUOpHead(ctypes.Structure):
        _fields_ = [("op", ctypes.c_int), ("dtype", _PolyDType)]


def node_key(node):
    return id(node) if ENGINE == "tinygrad" else int(node.raw)


def op_name(node):
    return node.op.name if ENGINE == "tinygrad" else node.op_name


def dtype_name(node):
    if ENGINE == "tinygrad":
        return node.dtype.name
    dtype = ctypes.cast(node.raw, ctypes.POINTER(_PolyUOpHead)).contents.dtype
    name = dtype.name.decode("utf-8") if dtype.name else "void"
    return f"{name}{dtype.count}" if dtype.count > 1 else name


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
            return arg.strip("'\"").upper()
        device = _ffi._lib.poly_uop_device(node.raw)
        name = _ffi._lib.poly_device_name(device)
        return name.decode("utf-8").upper() if name else f"DEVICE:{device}"
    if op == "CONST" and dtype_name(node) in {
        "half", "float", "double", "bfloat16", "float16", "float32", "float64",
        "__fp16", "__bf16",
    }:
        # tinygrad's diagnostic wraps float constants in ConstFloat(...);
        # Polygrad prints the same typed value directly. Compare the value,
        # retaining signed zero and non-finite classes.
        value_text = arg
        if value_text.startswith("ConstFloat(") and value_text.endswith(")"):
            value_text = value_text[len("ConstFloat("):-1]
        value = float(value_text)
        if math.isnan(value):
            return "<float:nan>"
        if math.isinf(value):
            return "<float:+inf>" if value > 0 else "<float:-inf>"
        return f"<float:{value.hex()}>"
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


CASES = {
    "basic_alu": ("tensor", case_basic_alu),
    "expand": ("tensor", case_expand),
    "flip": ("tensor", case_flip),
    "flip_scalar_noop": ("tensor", case_flip_scalar_noop),
    "movement_reduce": ("tensor", case_movement_reduce),
    "pad": ("tensor", case_pad),
    "pad_negative": ("tensor", case_pad_negative),
    "pad_noop": ("tensor", case_pad_noop),
    "pad_scalar_noop": ("tensor", case_pad_scalar_noop),
    "pad_scalar_value": ("tensor", case_pad_scalar_value),
    "pad_value": ("tensor", case_pad_value),
    "permute": ("tensor", case_permute),
    "rebuilt_after_realize": ("realize", case_rebuilt_after_realize),
    "reshape": ("tensor", case_reshape),
    "roundtrip_occurrence": ("tensor", case_roundtrip_occurrence),
    "shrink": ("tensor", case_shrink),
    "shrink_noop": ("tensor", case_shrink_noop),
    "shrink_scalar_noop": ("tensor", case_shrink_scalar_noop),
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
