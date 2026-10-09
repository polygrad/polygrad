"""Compare all decoded bytes against Torch; checkpoints must be trusted/pinned.

Usage: POLY_LIB=... PYTHONPATH=py python test/external/check_torch_checkpoint.py FILE ...
No download, conversion, or checkpoint code execution by Polygrad.
"""
import argparse
import ctypes as C
import mmap
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'py/tests'))
from test_torch_checkpoint import Decoded
from polygrad import _ffi

parser = argparse.ArgumentParser()
parser.add_argument('checkpoints', nargs='+', type=Path)
args = parser.parse_args()
lib = _ffi.get_lib()
lib.poly_hf_decode.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int64),
                               C.c_int, C.POINTER(C.POINTER(Decoded))]
lib.poly_hf_decode.restype = C.c_int
lib.poly_hf_decoded_free.argtypes = [C.POINTER(Decoded)]
for path in args.checkpoints:
    expected = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as data:
        pointer = C.addressof(C.c_char.from_buffer(data))
        result = C.POINTER(Decoded)()
        rc = lib.poly_hf_decode(b'{}', 2, (C.c_void_p * 1)(pointer), (C.c_int64 * 1)(len(data)), 1, C.byref(result))
        if rc: raise RuntimeError(lib.poly_import_last_error_message())
        try:
            assert result.contents.n_tensors == len(expected)
            for i in range(result.contents.n_tensors):
                actual = result.contents.tensors[i]
                name = actual.name.decode()
                want = expected[name]
                assert list(actual.shape[:actual.ndim]) == list(want.shape), name
                raw = want.contiguous().reshape(-1).view(torch.uint8).numpy()
                got = np.ctypeslib.as_array((C.c_uint8 * raw.size).from_address(actual.data))
                assert np.array_equal(got, raw), name
            print(f'{path}: {len(expected)} tensors, all shapes and bytes equal', flush=True)
        finally:
            lib.poly_hf_decoded_free(result)
