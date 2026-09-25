"""The encoder comparison must preserve matched pairs and failed arms."""
from pathlib import Path
import runpy

import numpy as np

BENCH = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'bench/bench_onnx_encoders.py'))


def test_encoder_inputs_change_values_and_padding():
    for length in (8, 128, 512):
        first, second = [BENCH['inputs'](length, i) for i in range(2)]
        for values in (first, second):
            assert all(v.shape == (1, length) and v.dtype == np.int64 for v in values.values())
            assert np.all(values['input_ids'][values['attention_mask'] == 0] == 0)
            assert 0 < values['attention_mask'].sum() < length
        assert not np.array_equal(first['input_ids'], second['input_ids'])


def test_encoder_summary_pairs_before_taking_median():
    rows = [dict(sequence=128, round=i, engine=e, beam=0, median_ms=value)
            for i, pair in enumerate(((2, 1), (300, 100), (40, 10)))
            for e, value in zip(('polygrad', 'ort'), pair)]
    result = BENCH['summarize'](rows, [128], [('polygrad', 0)])[0]
    assert result['paired_ort_ratio'] == 3
    rows[0] = dict(sequence=128, round=0, engine='polygrad', beam=0, status='failed')
    assert BENCH['summarize'](rows, [128], [('polygrad', 0)])[0]['status'] == 'incomplete'


def test_encoder_summary_does_not_silently_drop_missing_reference():
    rows = [dict(sequence=128, round=i, engine=e, beam=0, median_ms=value)
            for i, pair in enumerate(((2, 1), (30, 10), (400, 100)))
            for e, value in zip(('polygrad', 'ort'), pair)]
    del rows[1]
    assert BENCH['summarize'](rows, [128], [('polygrad', 0)])[0]['paired_ort_ratio'] is None
