import hashlib
import struct

import numpy as np
import pytest

from polygrad import Tensor, dtypes
from polygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, ggml_data_to_tensor, gguf_load
from polygrad.runtime.autogen import ggml_common


# Pinned tinygrad/llm/gguf.py at ba1d3baa. These one-block digests were
# produced by the paired TG/PG probe recorded in temp/gguf_loader_correspondence_20260815.md.
QUANT_DIGESTS = {
    2: "719ac4caff1179087d4915e6daf2d031da0b694e95cb7f54a19b00f8bfb552ff",
    3: "272690b04a254217d8fa8014b1e37d2eca6c1bf3370ce2bab4f87d459c474940",
    6: "1b2c6393c9c607af27050a94a60968cc5512b7a8a2f6416406e06e16a8d3c755",
    7: "041755c87db43a71cb14ef0a15f73ec68d89525dc484e09b5625c925bcbb8958",
    8: "1ca5b7e7309e05142988e0bdd7004e094008a99f989877abac9e6303f577b7d7",
    12: "667eb74a3ab99d3d0c47d7169e5222afd5159bde5601ae053e0459fdd4982889",
    13: "8c5b6c3d07a6b97a8d6285bd9246f16a4539f7f6264cfdf2cddd210b65cb7997",
    14: "b0d4aeb36070c1e838f41aaeffefcee6df48920ec3f66544666d61bf9431314b",
    18: "35aa99c61a8a85f22c5fc116e83dcc80299c6f20d820774a00f0cda44d08cdbb",
    21: "e6b8f41fe5dd4b83176b0dfc4778046eab3011a9aff1d5de3d0cf0613f82bf0f",
    22: "2d01f1334a2455d5cc979f79de2becb32e1385d15490155b6190916df6b09da9",
    23: "95dfe54ba6e4344765bb748d6d1758168691ada56e43008bd50f73b5527d64e5",
    39: "000eec122d907e5d4755ee10bbd879014fe44cf5362e8c7cfa23a55a73e1b557",
    41: "300f2f329ab1e99dcc6285cd9304674b01ef0c24bf8a3d7acddec825fb410448",
}

NATIVE_DIGESTS = {
    0: "8e85a85ddd084337955565d40c74538e30440f3db1ea25ddc20819446d17222c",
    1: "a3647a22425dd10e4292d38df449d7fa66e1dce13300e003ed4fd8b3559a80b4",
    24: "31ca2bb38aec879274fa64fafcb97acddd2d99fd819c53a7a9ae1348fbad2dfb",
    25: "eebf62ed1d18b8a447bbdcc8cc48e72da27539760e5135fc376c16e1a1c75de7",
    26: "f5af8a4f17a2e0dddecfc1606b6124b54a3d54c2820d255a5e03a97203cc4010",
    27: "ab4961a482f0d301f23be862e7718963be81c034c2a8f4fb866d9ad87242da44",
    28: "354d3e966e145f9a886c4855973879c6f28e001e461aef54b1e1de7de7b76b6f",
    30: "9739b3f2f3814e0853da20cc3fa76ae21148f78d8c1de9725ab24930c1d98282",
}

NATIVE_DTYPE_NAMES = {
    0: "float32", 1: "float16", 24: "int8", 25: "int16",
    26: "int32", 27: "int64", 28: "float64", 30: "bfloat16",
}

TABLE_DIGESTS = {
    "iq2s_grid": (1024, "d3e8814adefaf617e221cf1ce2b32993c11896f7a65d499c1bdc27a89fc247f2"),
    "iq3xxs_grid": (256, "d1e9f6569ca6c9e56d46fab04820792dfa57c648314156771027341094ac25ed"),
    "iq3s_grid": (512, "ac88dff0d209bd6a835a5e6d8749d3ba9390ffd5a2ed148b2005978dbd20dcf6"),
    "kvalues_iq4nl": (16, "a2126574965ec30e18e9d71d4eb35810ee64454294383ffcb442711bb90112b0"),
}


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _quant_block(ggml_type: int, nbytes: int) -> np.ndarray:
    data = ((np.arange(nbytes, dtype=np.uint16) * 37 + 11) & 0xFF).astype(np.uint8)

    def half(offset: int, value: float):
        data[offset:offset + 2] = np.asarray([value], dtype=np.float16).view(np.uint8)

    if ggml_type in {2, 6, 8, 18, 21, 22, 23, 41}:
        half(0, 1.0)
    elif ggml_type in {3, 7, 12, 13}:
        half(0, 1.0)
        half(2, 0.5)
    elif ggml_type == 14:
        half(nbytes - 2, 1.0)
    elif ggml_type == 39:
        data[0] = 1
    return data


def _native_bytes(ggml_type: int) -> np.ndarray:
    dtype = _GGML_NATIVE[ggml_type]
    if dtype == dtypes.float32:
        return np.linspace(-2, 2, 8, dtype=np.float32).view(np.uint8)
    if dtype == dtypes.float16:
        return np.linspace(-2, 2, 8, dtype=np.float16).view(np.uint8)
    if dtype == dtypes.float64:
        return np.linspace(-2, 2, 8, dtype=np.float64).view(np.uint8)
    if dtype == dtypes.bfloat16:
        return (np.linspace(-2, 2, 8, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)
    numpy_dtype = {
        dtypes.int8: np.int8,
        dtypes.int16: np.int16,
        dtypes.int32: np.int32,
        dtypes.int64: np.int64,
    }[dtype]
    return np.arange(-4, 4, dtype=numpy_dtype).view(np.uint8)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _build_gguf(tensors, kvs) -> bytes:
    # Pinned test/unit/test_gguf.py:_build_gguf, retained as an offline fixture.
    buf = bytearray(struct.pack("<4siqq", b"GGUF", 3, len(tensors), len(kvs)))
    for key, value in kvs:
        buf += _pack_string(key)
        if isinstance(value, str):
            buf += struct.pack("<i", 8) + _pack_string(value)
        else:
            buf += struct.pack("<iI", 4, value)
    data_offset = 0
    for name, dims, qtype, data in tensors:
        buf += _pack_string(name) + struct.pack("<I", len(dims))
        for dim in reversed(dims):
            buf += struct.pack("<Q", dim)
        buf += struct.pack("<iQ", qtype, data_offset)
        data_offset += len(data)
    buf += bytes((-len(buf)) % 32)
    for _, _, _, data in tensors:
        buf += data
    return bytes(buf)


def test_generated_iq_tables_match_pinned_literals():
    for name, (expected_len, expected_digest) in TABLE_DIGESTS.items():
        table = getattr(ggml_common, name)
        assert len(table) == expected_len
        assert hashlib.sha256(repr(table).encode()).hexdigest() == expected_digest


@pytest.mark.parametrize("ggml_type", sorted(QUANT_DIGESTS))
def test_quantized_one_block_matches_pinned_output(ggml_type):
    nelements, nbytes = _GGML_QUANT[ggml_type]
    source = Tensor(_quant_block(ggml_type, nbytes), dtype=dtypes.uint8, device="INTERP").realize()
    output = ggml_data_to_tensor(source, nelements, ggml_type)
    value = output.numpy()
    assert output.shape == ((32,) if ggml_type == 39 else (1, nelements))
    assert output.dtype == ("float16" if ggml_type == 41 else "float32")
    assert np.isfinite(value).all()
    assert _digest(value) == QUANT_DIGESTS[ggml_type]


@pytest.mark.parametrize("ggml_type", sorted(NATIVE_DIGESTS))
def test_native_one_block_matches_pinned_output(ggml_type):
    source = Tensor(_native_bytes(ggml_type), dtype=dtypes.uint8, device="INTERP").realize()
    output = ggml_data_to_tensor(source, 8, ggml_type)
    assert output.shape == (8,)
    assert output.dtype == NATIVE_DTYPE_NAMES[ggml_type]
    assert _digest(output.numpy()) == NATIVE_DIGESTS[ggml_type]


def test_gguf_load_synthetic_metadata_native_and_quantized(tmp_path):
    native = np.asarray([1.5, -2.0, 3.25, 4.5, -5.75, 6.0], dtype=np.float32)
    quant = _quant_block(2, 18)
    fixture = tmp_path / "probe.gguf"
    fixture.write_bytes(_build_gguf(
        [("native.weight", (2, 3), 0, native.tobytes()), ("quant.weight", (32,), 2, quant.tobytes())],
        [("general.alignment", 32), ("model.name", "probe")],
    ))

    metadata, state = gguf_load(fixture)
    assert metadata == {"general.alignment": 32, "model.name": "probe"}
    assert list(state) == ["native.weight", "quant.weight"]
    assert state["native.weight"].shape == (2, 3)
    assert state["quant.weight"].shape == (32,)
    assert state["native.weight"].uop.op_name == "RESHAPE"
    assert state["quant.weight"].uop.op_name == "RESHAPE"
    np.testing.assert_array_equal(state["native.weight"].numpy(), native.reshape(2, 3))
    assert _digest(state["quant.weight"].numpy()) == QUANT_DIGESTS[2]


def test_gguf_load_merges_split_files_like_pinned(tmp_path):
    first = tmp_path / "test-00001-of-00002.gguf"
    second = tmp_path / "test-00002-of-00002.gguf"
    a = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.asarray([5.0, 6.0], dtype=np.float32)
    first.write_bytes(_build_gguf([("a", (4,), 0, a.tobytes())], [("split.count", 2), ("split.no", 0)]))
    second.write_bytes(_build_gguf([("b", (2,), 0, b.tobytes())], [("split.count", 2), ("split.no", 1)]))

    metadata, state = gguf_load(first)
    assert metadata["split.count"] == 2
    assert list(state) == ["a", "b"]
    np.testing.assert_array_equal(state["a"].numpy(), a)
    np.testing.assert_array_equal(state["b"].numpy(), b)

    second.unlink()
    with pytest.raises(FileNotFoundError):
        gguf_load(first)


def test_gguf_rejects_unknown_type():
    with pytest.raises(ValueError, match="GGML type '1337' is not supported"):
        ggml_data_to_tensor(Tensor.empty(512, dtype=dtypes.uint8, device="INTERP"), 256, 1337)
