"""Inventory pinned ONNX dispatch against the C loader; not a conformance claim."""
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'references/tinygrad_014'))
from tinygrad.nn.onnx import onnx_ops

source = (root / 'src/loaders/onnx_loader.c').read_text()
direct = set(re.findall(r'UNARY\("([A-Z][A-Za-z0-9]+)"', source))
direct.update(re.findall(r'strcmp\((?:d->)?op, "([A-Z][A-Za-z0-9]+)"\)', source))
direct.update('Reduce' + k for k in re.findall(r'strcmp\(kind, "([A-Za-z0-9]+)"\)', source))
present = sorted(set(onnx_ops) & direct)
missing = sorted(set(onnx_ops) - direct)
print(json.dumps({'reference': 'references/tinygrad_014/tinygrad/nn/onnx.py',
                  'meaning': 'dispatch presence only; attributes, dtypes, opsets and shapes may be restricted',
                  'reference_count': len(onnx_ops), 'dispatch_count': len(present),
                  'dispatch_present': present, 'missing': missing}, indent=2))
