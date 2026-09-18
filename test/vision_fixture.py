"""Reconstruct fixed vision inputs; hashes bind formulas to the recorded oracle."""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def weight_bytes(spec):
    # Preserve the original safetensors header/order/padding, not merely values.
    header = spec['header'].encode('utf-8')
    entries = json.loads(header)
    size = max(v['data_offsets'][1] for k, v in entries.items() if k != '__metadata__')
    data = bytearray(size)
    for name, entry in entries.items():
        if name == '__metadata__': continue
        assert entry['dtype'] == 'F32'
        start, end = entry['data_offsets']
        assert end - start == math.prod(entry['shape']) * 4
        for i in range((end - start) // 4):
            value = f32(((i * 7 + sum(name.encode())) % 29 - 14) * f32(.013))
            if ('norm' in name and name.endswith('weight')) or name.endswith('lambda1'):
                value = f32(value + 1)
            if name == 'logit_scale': value = f32(1.3)
            struct.pack_into('<f', data, start + i * 4, value)
    result = struct.pack('<Q', len(header)) + header + data
    assert hashlib.sha256(result).hexdigest() == spec['sha256'], 'vision checkpoint formula drift'
    return result


def pixel_values(spec):
    values = [f32(f32((i * 11 % 101) / 50) - 1) for i in range(math.prod(spec['shape']))]
    raw = struct.pack('<' + 'f' * len(values), *values)
    assert hashlib.sha256(raw).hexdigest() == spec['sha256'], 'vision input formula drift'
    for width in reversed(spec['shape'][1:]):
        values = [values[i:i + width] for i in range(0, len(values), width)]
    return values


def expand_case(case):
    return {**case, 'weights': base64.b64encode(weight_bytes(case['weights'])).decode(),
            'inputs': {**case['inputs'], 'pixel_values': pixel_values(case['inputs']['pixel_values'])}}


def load_vision_cases():
    fixture = json.loads((Path(__file__).parent / 'fixtures/vision.json').read_text())
    assert fixture['payload_formula'] == 'vision-f32@1'
    return [expand_case(case) for case in fixture['cases']]


def compact_case(case):
    """Used by the oracle generator; verify reconstruction before dropping bytes."""
    raw = base64.b64decode(case['weights'])
    size = struct.unpack('<Q', raw[:8])[0]
    pixels = case['inputs']['pixel_values']
    shape = []
    cursor = pixels
    while isinstance(cursor, list):
        shape.append(len(cursor))
        cursor = cursor[0]
    flat = pixels
    for _ in shape[1:]: flat = [v for row in flat for v in row]
    pixel_bytes = struct.pack('<' + 'f' * len(flat), *flat)
    packed = {**case, 'weights': {'header': raw[8:8 + size].decode(), 'sha256': hashlib.sha256(raw).hexdigest()},
              'inputs': {**case['inputs'], 'pixel_values': {'shape': shape, 'sha256': hashlib.sha256(pixel_bytes).hexdigest()}}}
    assert expand_case(packed) == case
    return packed
