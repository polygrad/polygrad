"""Pinned, trained BERT encoder: ORT/Tinygrad correspondence and portable Model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = 'sentence-transformers-testing/stsb-bert-tiny-onnx'
REVISION = 'da60a9ed87aeb7e57d9b15be5d021b28d153f584'
SHA256 = 'c26b1b3e2f210b1e49e7a1a5067f6451c7c9039ac5a91848c5d455372fd5d618'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, 'onnx/model.onnx', revision=REVISION, local_dir=args.directory)
    path = args.directory / 'onnx/model.onnx'
    raw = path.read_bytes()  # Missing checkpoints fail; this is an opt-in lane.
    assert hashlib.sha256(raw).hexdigest() == SHA256, 'encoder checkpoint hash mismatch'
    if args.fetch:
        print(f'Fetched {REPO}@{REVISION}, SHA256={SHA256}')
        return

    import onnxruntime as ort
    sys.path.insert(0, str(ROOT / 'references/tinygrad_014'))
    from tinygrad.nn.onnx import OnnxRunner
    import polygrad as pg
    from polygrad import _ffi

    reference = ort.InferenceSession(raw, providers=['CPUExecutionProvider'])
    tinygrad = OnnxRunner(path)
    dimensions = {'batch_size': 1, 'sequence_length': 8}
    cases = []
    for ids, mask, types in [
        ([101,2023,2003,1037,3231,102,0,0], [1,1,1,1,1,1,0,0], [0]*8),
        ([101,2009,2003,2204,102,0,0,0], [1,1,1,1,1,0,0,0], [0,0,1,1,1,0,0,0]),
    ]:
        inputs = {k: np.array([v], np.int64) for k,v in
                  zip(('input_ids','attention_mask','token_type_ids'), (ids,mask,types))}
        expected = reference.run(['last_hidden_state'], inputs)[0]
        assert np.isfinite(expected).all(), 'encoder reference contains nonfinite values'
        actual = tinygrad(inputs)['last_hidden_state'].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        cases.append({'inputs': {k: v.reshape(-1).tolist() for k,v in inputs.items()},
                      'output': expected.reshape(-1).tolist()})
        print('Tinygrad max error:', float(np.max(np.abs(actual-expected))), flush=True)

    with pg.Runtime(device='CPU') as rt:
        model = rt.Model.from_onnx(raw, dimensions=dimensions)
        restored = None
        try:
            bundle = model.save(str(args.directory / 'model.pgb'))
            restored = rt.Model.load(bundle)
            for label, current in [('ONNX',model), ('bundle',restored)]:
                for case in cases:
                    inputs = {k: np.array([v], np.int64) for k,v in case['inputs'].items()}
                    actual = current.call('forward', inputs)['last_hidden_state']
                    assert actual.shape == (1,8,128)
                    expected = np.array(case['output'], np.float32).reshape(actual.shape)
                    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                    print(label, 'max error:', float(np.max(np.abs(actual-expected))), flush=True)
        finally:
            if restored is not None:
                restored.dispose()
            model.dispose()
    oracle = {'repository': REPO, 'revision': REVISION, 'sha256': SHA256,
              'onnxruntime': ort.__version__, 'dimensions': dimensions, 'cases': cases}
    (args.directory / 'oracle.json').write_text(json.dumps(oracle, separators=(',', ':')) + '\n')
    print('PASS: trained encoder and bundle; library:', _ffi.get_lib()._name)


if __name__ == '__main__':
    main()
