"""Restricted checkpoint decoding through the shared C loader (no Torch dependency)."""
import base64
import ctypes as C
import io
import json
import struct
from pathlib import Path
import zipfile

import numpy as np
import pytest
import polygrad as pg
from polygrad import _ffi

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/torch.json').read_text())


class Tensor(C.Structure):
    _fields_ = [('name', C.c_char_p), ('data', C.c_void_p), ('shape', C.c_int64 * 8),
                ('ndim', C.c_int), ('numel', C.c_int64), ('dtype', C.c_int)]


class Decoded(C.Structure):
    _fields_ = [('config', C.c_void_p), ('model_type', C.c_char_p),
                ('tensors', C.POINTER(Tensor)), ('n_tensors', C.c_int)]


def decode(data, expected=None, reason=None):
    lib = _ffi.get_lib()
    fn = lib.poly_hf_decode
    fn.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int64),
                   C.c_int, C.POINTER(C.POINTER(Decoded))]
    fn.restype = C.c_int
    lib.poly_hf_decoded_free.argtypes = [C.POINTER(Decoded)]
    buf = C.create_string_buffer(data)
    out = C.POINTER(Decoded)()
    rc = fn(b'{}', 2, (C.c_void_p * 1)(C.addressof(buf)), (C.c_int64 * 1)(len(data)), 1, C.byref(out))
    if expected is None:
        assert rc != 0 and not out
        assert lib.poly_import_last_error_message()
        if reason: assert reason in lib.poly_import_last_error_message().decode()
        return
    try:
        assert rc == 0, lib.poly_import_last_error_message()
        assert out.contents.n_tensors == len(expected)
        for i, want in enumerate(expected):
            got = out.contents.tensors[i]
            assert got.name.decode() == want['name']
            assert list(got.shape[:got.ndim]) == want['shape']
            assert got.dtype == want['dtype']
            assert C.string_at(got.data, len(want['hex']) // 2).hex() == want['hex']
    finally:
        lib.poly_hf_decoded_free(out)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda x: str(x['protocol']))
def test_torch_layouts_and_dtypes(case):
    decode(base64.b64decode(case['archive']), case['tensors'])


def rewrite(data, transform, compression=zipfile.ZIP_STORED):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(out, 'w', compression=compression) as dest:
        for name in source.namelist():
            dest.writestr(name, transform(name, source.read(name)))
    return out.getvalue()


@pytest.mark.parametrize('payload', [b"cos\nsystem\n(S'echo unsafe'\ntR.",
                                   b'\x80\x02}q\x00h\x00s.', b'\x80\x02]q\x00h\x00a.',
                                   b'\x80\x02\x82\x01.', b'\x80\x02N.', b'\x80\x02(]'])
def test_torch_rejects_executable_and_non_state_dict_pickles(payload):
    data = base64.b64decode(FIXTURE['cases'][0]['archive'])
    decode(rewrite(data, lambda n, v: payload if n.endswith('/data.pkl') else v))


def test_torch_truncation_and_compression():
    data = base64.b64decode(FIXTURE['cases'][0]['archive'])
    for n in [0, 1, 4, 30, len(data)//2, len(data)-1]: decode(data[:n])
    decode(rewrite(data, lambda n, v: v, zipfile.ZIP_DEFLATED))


def test_torch_storage_bounds_duplicates_and_byteorder():
    case = FIXTURE['cases'][0]
    data = base64.b64decode(case['archive'])
    decode(rewrite(data, lambda n, v: v[:-1] if '/data/' in n else v))
    decode(rewrite(data, lambda n, v: b'unknown' if n.endswith('/byteorder') else v))
    out = io.BytesIO(data)
    with zipfile.ZipFile(out, 'a') as z:
        with pytest.warns(UserWarning, match='Duplicate name'):
            z.writestr(z.namelist()[0], z.read(z.namelist()[0]))
    decode(out.getvalue())
    # One dtype per storage. Transform payloads and byteorder without changing
    # shapes, including a transposed/shared view and a scalar.
    data = base64.b64decode(FIXTURE['model']['archive'])
    big = rewrite(data, lambda n, v: b'big' if n.endswith('/byteorder') else
                  b''.join(v[i:i+4][::-1] for i in range(0, len(v), 4)) if '/data/' in n else v)
    decode(big, FIXTURE['model']['tensors'])


def tensor_pickle(shape=(4, 6), strides=(6, 1), offset=0, count=24):
    def number(n):
        return b'\x8a\x08' + struct.pack('<q', n)
    def string(s):
        return b'X' + struct.pack('<I', len(s)) + s
    def tuple_(items):
        return b'(' + b''.join(items) + b't'
    storage = tuple_([string(b'storage'), b'ctorch\nFloatStorage\n', string(b'0'), string(b'cpu'), number(count)]) + b'Q'
    args = tuple_([storage, number(offset), tuple_(map(number, shape)), tuple_(map(number, strides)), b'\x89', b'}'])
    return b'\x80\x02}' + string(b'x') + b'ctorch._utils\n_rebuild_tensor_v2\n' + args + b'Rs.'


@pytest.mark.parametrize('kwargs,reason', [
    ({'offset': -1}, 'layout'), ({'offset': 24}, 'exceeds storage'),
    ({'strides': (60, 1)}, 'exceeds storage'), ({'strides': (-6, 1)}, 'shape/stride'),
    ({'shape': (2**62, 8)}, 'overflow'), ({'count': 25}, 'storage size'),
    ({'shape': (1,)*9, 'strides': (1,)*9}, 'layout'),
    ({'shape': (1000000000,), 'strides': (0,)}, 'resource limit'),
])
def test_torch_rejects_invalid_views(kwargs, reason):
    data = base64.b64decode(FIXTURE['cases'][0]['archive'])
    decode(rewrite(data, lambda n, v: tensor_pickle(**kwargs) if n.endswith('/data.pkl') else v), reason=reason)


def test_torch_handbuilt_constructor_and_zip64():
    data = base64.b64decode(FIXTURE['cases'][0]['archive'])
    want = dict(FIXTURE['cases'][0]['tensors'][0], name='x')
    decode(rewrite(data, lambda n, v: tensor_pickle() if n.endswith('/data.pkl') else v), [want])
    # Force ZIP64 local fields even on a small archive. torch.save already uses
    # ZIP64 end/locator records; this also exercises local sentinel expansion.
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(out, 'w') as dest:
        for name in source.namelist():
            with dest.open(name, 'w', force_zip64=True) as target:
                target.write(source.read(name))
    decode(out.getvalue(), FIXTURE['cases'][0]['tensors'])


def test_hf_checkpoint_selection_is_fail_closed(tmp_path):
    from polygrad.hf import plan_hf
    (tmp_path / 'pytorch_model.bin').write_bytes(base64.b64decode(FIXTURE['model']['archive']))
    def plan(index=None):
        return plan_hf([p.name for p in tmp_path.iterdir()], index)
    assert plan()['files'] == ['pytorch_model.bin']
    (tmp_path / 'model.fp16.safetensors').write_bytes(b'duplicate variant')
    assert plan()['files'] == ['pytorch_model.bin']
    (tmp_path / 'model.safetensors').write_bytes(b'bad')
    assert plan()['files'] == ['model.safetensors']
    (tmp_path / 'model.safetensors').unlink()
    index = tmp_path / 'model.safetensors.index.json'
    for name in ['missing.safetensors', '../outside', '/absolute', 'sub\\outside']:
        index.write_text(json.dumps({'weight_map': {'x': name}}), encoding='utf-8')
        with pytest.raises((ValueError, FileNotFoundError)):
            plan(index.read_text(encoding='utf-8'))


def test_shared_hf_plan():
    from polygrad.hf import plan_hf
    names = ['model.fp16.safetensors', 'pytorch_model.bin', 'model.safetensors']
    assert plan_hf(names) == dict(format='safetensors', index=None, files=['model.safetensors'])
    names += ['model.safetensors.index.json', 'b.safetensors', 'a.safetensors']
    assert plan_hf(names)['files'] == ['model.safetensors']
    names.remove('model.safetensors')
    assert plan_hf(names)['files'] == []
    index = {'weight_map': {'z': 'b.safetensors', 'x': 'a.safetensors', 'y': 'b.safetensors'}}
    assert plan_hf(names, index)['files'] == ['a.safetensors', 'b.safetensors']
    for bad in ['../x', '/x', 'x\\y', 'x\x00y', 'https://host/x']:
        with pytest.raises(ValueError):
            plan_hf(names + [bad], {'weight_map': {'x': bad}})
    with pytest.raises(ValueError):
        plan_hf(names, {'weight_map': {'x': 'absent'}})
    with pytest.raises(ValueError):
        plan_hf(['model.fp16.safetensors'])


@pytest.mark.parametrize('row', json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/hf_plan.json').read_text()))
def test_hf_plan_cross_frontend_contract(row):
    from polygrad.hf import plan_hf
    if 'error' in row:
        with pytest.raises(ValueError, match=row['error']):
            plan_hf(row['files'], row.get('index'))
    else:
        assert plan_hf(row['files'], row.get('index')) == row['expected']


def test_torch_global_error_names_symbol():
    decode(rewrite(base64.b64decode(FIXTURE['cases'][0]['archive']),
                   lambda name, data: b'\x80\x02cposix\nsystem\n.' if name.endswith('/data.pkl') else data),
           reason='posix.system')


@pytest.mark.parametrize('module,name', [(b'bad\x1b[31m', b'run\x00x'), (b'x' * 1000, b'y' * 1000)])
def test_torch_global_error_is_bounded_and_sanitized(module, name):
    payload = b'\x80\x02c' + module + b'\n' + name + b'\n.'
    decode(rewrite(base64.b64decode(FIXTURE['cases'][0]['archive']),
                   lambda n, data: payload if n.endswith('/data.pkl') else data))
    message = _ffi.get_lib().poly_import_last_error_message()
    assert len(message) < 240 and all(32 <= c <= 126 for c in message)
    assert b'?' in message if len(module) < 64 else b'...' in message


def test_hf_shard_index_accepts_hub_cache_symlinks(tmp_path):
    from polygrad.hf import plan_hf
    snapshot = tmp_path / 'snapshots' / 'commit'
    snapshot.mkdir(parents=True)
    blobs = tmp_path / 'blobs'
    blobs.mkdir()
    (blobs / 'weight').write_bytes(b'cached shard')
    shard = snapshot / 'part.safetensors'
    shard.symlink_to('../../blobs/weight')
    (snapshot / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'x': 'part.safetensors'}}), encoding='utf-8')
    assert plan_hf([p.name for p in snapshot.iterdir()],
                   (snapshot / 'model.safetensors.index.json').read_text())['files'] == [shard.name]


def test_hf_local_bin_and_sharded_selection(tmp_path):
    from polygrad.hf import load_hf
    case = FIXTURE['model']
    (tmp_path / 'config.json').write_text(json.dumps(case['config']), encoding='utf-8')
    data = base64.b64decode(case['archive'])
    (tmp_path / 'pytorch_model.bin').write_bytes(data)
    for sharded in [False, True]:
        if sharded:
            (tmp_path / 'pytorch_model.bin').unlink()
            (tmp_path / 'shard.bin').write_bytes(data)
            (tmp_path / 'pytorch_model.bin.index.json').write_text(
                json.dumps({'weight_map': {t['name']: 'shard.bin' for t in case['tensors']}}), encoding='utf-8')
        model = load_hf(tmp_path, max_seq_len=4)
        try:
            np.testing.assert_array_equal(model.read_buffer('wte.weight'),
                np.frombuffer(bytes.fromhex(case['tensors'][0]['hex']), dtype='<f4'))
        finally:
            model.dispose()


def test_hf_download_pins_fallback_without_fetching_both_formats(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from polygrad.hf import download_hf
    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    calls = []
    names = ['config.json', 'pytorch_model.bin', 'model.fp16.safetensors']
    def info(repo, revision):
        assert revision == 'release'
        return SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=n) for n in names])
    def download(repo, filename, **kwargs):
        calls.append((filename, kwargs))
        assert kwargs['revision'] == 'a' * 40
        return str(snapshot / filename)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(model_info=info, hf_hub_download=download))
    assert download_hf('org/model', revision='release') == str(snapshot)
    assert [c[0] for c in calls] == ['config.json', 'pytorch_model.bin']
    calls.clear()
    (snapshot / 'pytorch_model.bin.index.json').write_text(
        json.dumps({'weight_map': {'x': 'weights/part1.bin', 'y': 'weights/part2.bin'}}), encoding='utf-8')
    names += ['pytorch_model.bin.index.json', 'weights/part1.bin', 'weights/part2.bin']
    download_hf('org/model', revision='release')
    assert [c[0] for c in calls] == ['config.json', 'pytorch_model.bin']
    calls.clear()
    names.remove('pytorch_model.bin')
    download_hf('org/model', revision='release')
    assert [c[0] for c in calls] == ['pytorch_model.bin.index.json', 'config.json', 'weights/part1.bin', 'weights/part2.bin']
    calls.clear()
    names += ['model.safetensors']
    download_hf('org/model', revision='release')
    assert [c[0] for c in calls] == ['config.json', 'model.safetensors']
    calls.clear()
    names += ['model.safetensors.index.json']
    (snapshot / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {'x': 'missing.safetensors'}}), encoding='utf-8')
    download_hf('org/model', revision='release')
    assert [c[0] for c in calls] == ['config.json', 'model.safetensors']
    calls.clear()
    names.remove('model.safetensors')
    with pytest.raises(ValueError, match='shard not found'):
        download_hf('org/model', revision='release')
    assert [c[0] for c in calls] == ['model.safetensors.index.json']


def test_torch_model_matches_safetensors_and_roundtrips():
    case = FIXTURE['model']
    header, raw = {}, bytearray()
    for t in case['tensors']:
        begin = len(raw)
        raw.extend(bytes.fromhex(t['hex']))
        header[t['name']] = dict(dtype='F32', shape=t['shape'], data_offsets=[begin, len(raw)])
    encoded = json.dumps(header).encode()
    safe = len(encoded).to_bytes(8, 'little') + encoded + raw
    with pg.create(device='CPU') as rt:
        models = []
        try:
            for data in [base64.b64decode(case['archive']), safe]:
                models.append(pg.Model.from_hf(config_json=json.dumps(case['config']),
                                              weight_bytes_list=[data], max_seq_len=4, runtime=rt))
            x = np.array([[1, 2, 3, 4]], np.int32)
            inputs = dict(x=x, positions=np.arange(4, dtype=np.int32).reshape(1, 4))
            expected = models[1].forward(**inputs)['output']
            np.testing.assert_array_equal(models[0].forward(**inputs)['output'], expected)
            models.append(pg.Model.load(models[0].save(), runtime=rt))
            np.testing.assert_array_equal(models[2].forward(**inputs)['output'], expected)
        finally:
            for model in models: model.dispose()
