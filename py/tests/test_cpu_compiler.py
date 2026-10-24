"""Native compiler policy, persistent cache identity and ordered sums."""
import ctypes as ct
import json
import shutil
import sys

import numpy as np
import pytest

from polygrad import _ffi


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    clang = shutil.which('clang')
    if clang is None:
        pytest.skip('ordered CPU reductions require Clang')
    monkeypatch.setenv('CC', clang)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    monkeypatch.setenv('POLY_CACHE', '1')
    lib = _ffi.get_lib()
    lib.poly_compile_c.argtypes = [ct.c_char_p, ct.c_char_p]
    lib.poly_compile_c.restype = ct.c_void_p
    lib.poly_program_call.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_int]
    lib.poly_program_destroy.argtypes = [ct.c_void_p]
    return lib, clang


def call(lib, source, arrays):
    program = lib.poly_compile_c(source.encode(), b'ordered_test')
    assert program, 'native compilation failed'
    try:
        args = (ct.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
        lib.poly_program_call(program, args, len(arrays))
    finally:
        lib.poly_program_destroy(program)


def test_ordered_compiler_arguments_and_persistent_cache(compiler, monkeypatch, tmp_path):
    lib, clang = compiler
    log = tmp_path / 'argv.jsonl'
    wrapper = tmp_path / 'compiler'
    # Encode the observed driver option in the compiled output. The C source
    # stays identical across toggles, so a wrong cache hit is visible too.
    wrapper.write_text(f'''#!{sys.executable}
import json, os, sys
args = sys.argv[1:]
with open({str(log)!r}, 'a') as f: f.write(json.dumps(args) + '\\n')
ordered = int('-force-ordered-reductions' in args)
os.execv({clang!r}, [{clang!r}, *args, '-DORDERED_TEST=' + str(ordered)])
''')
    wrapper.chmod(0o755)
    monkeypatch.setenv('CC', str(wrapper))
    source = 'void ordered_test_call(void **args) { *(int *)args[0] = ORDERED_TEST; }'
    for enabled in [None, 0, 1, 0, 1]:
        if enabled is None:
            monkeypatch.delenv('POLY_CPU_ORDERED_REDUCTIONS', raising=False)
        else:
            monkeypatch.setenv('POLY_CPU_ORDERED_REDUCTIONS', str(enabled))
        out = np.array([-1], dtype=np.int32)
        call(lib, source, [out])
        assert out[0] == (enabled or 0)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2, 'each compiler policy should compile once then hit disk cache'
    assert '-force-ordered-reductions' not in rows[0]
    pos = rows[1].index('-force-ordered-reductions')
    assert rows[1][pos - 1] == '-mllvm'


@pytest.mark.parametrize('dtype,ctype,large', [(np.float32, 'float', 2**24),
                                               (np.float64, 'double', 2**53)])
def test_ordered_reductions_preserve_serial_sum_and_tails(compiler, monkeypatch, dtype, ctype, large):
    lib, _ = compiler
    source = f'''void ordered_test_call(void **args) {{
      {ctype} *out = args[0], *x = args[1]; int n = *(int *)args[2];
      {ctype} sum = 0;
      for (int i = 0; i < n; i++) sum = sum + x[i];
      *out = sum;
    }}'''
    fixtures = [np.resize(np.array([large, 1, -large, 1], dtype=dtype), n)
                for n in [0, 1, 3, 7, 16, 1025]]
    fixtures += [np.array(v, dtype=dtype) for v in [[-0.0], [np.inf, 1], [-np.inf],
                                                  [np.inf, -np.inf], [np.nan, 1]]]
    outputs = []
    for enabled in [0, 1]:
        monkeypatch.setenv('POLY_CPU_ORDERED_REDUCTIONS', str(enabled))
        values = []
        for data in fixtures:
            out = np.zeros(1, dtype=dtype)
            call(lib, source, [out, data, np.array([data.size], dtype=np.int32)])
            values.append(out[0])
        outputs.append(np.array(values, dtype=dtype))
    finite = ~np.isnan(outputs[0])
    assert np.array_equal(np.isnan(outputs[0]), np.isnan(outputs[1]))
    assert outputs[0][finite].tobytes() == outputs[1][finite].tobytes()
    assert outputs[1][4] == 1, 'the cancellation fixture must retain serial addition'


def test_ordered_reductions_reject_unsupported_compiler(compiler, monkeypatch, capfd):
    lib, _ = compiler
    gcc = shutil.which('gcc')
    if gcc is None:
        pytest.skip('GCC required for unsupported-compiler check')
    monkeypatch.setenv('CC', gcc)
    monkeypatch.setenv('POLY_CPU_ORDERED_REDUCTIONS', '1')
    program = lib.poly_compile_c(b'void ordered_test_call(void **args) {}', b'ordered_test')
    if program:
        lib.poly_program_destroy(program)
    assert not program
    assert 'POLY_CPU_ORDERED_REDUCTIONS' in capfd.readouterr().err
