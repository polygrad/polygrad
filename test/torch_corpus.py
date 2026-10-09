"""Seed the restricted-checkpoint fuzzer with the reviewed, small Torch fixtures."""
import base64
import json
from pathlib import Path
import sys

fixture = json.loads((Path(__file__).parent / 'fixtures/torch.json').read_text())
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
for i, case in enumerate(fixture['cases'] + [fixture['model']]):
    (destination / f'torch-{i}').write_bytes(base64.b64decode(case['archive']))
