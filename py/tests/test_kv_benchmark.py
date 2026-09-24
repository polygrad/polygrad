"""Benchmark reports retain compact evidence, not vocabulary-sized JSON arrays."""
import json
from pathlib import Path
import runpy

import numpy as np


def test_trajectory_summary_is_compact_and_binds_all_logits():
    summarize = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'bench/bench_kv.py'))['trajectory_summary']
    values = np.zeros((225, 1, 32000), dtype=np.float32)
    values[:, :, 17] = 2
    summary = summarize(values)
    assert summary['shape'] == [225, 1, 32000]
    assert summary['dtype'] == 'float32'
    assert summary['argmax_tokens'] == [17] * 225
    assert len(json.dumps(summary)) < 2048
    # A non-winning logit still changes the digest; argmax alone is not evidence
    # that the full numerical comparison used the same trajectory.
    values[0, 0, 0] = 1
    changed = summarize(values)
    assert changed['argmax_tokens'] == summary['argmax_tokens']
    assert changed['sha256'] != summary['sha256']
