#!/usr/bin/env python3
"""Paired FP32 cached-Llama diagnostic, not a release speed threshold.

Both engines receive host int32 tokens and return last-token host logits. The
reference is the pin's unmodified extra/models/llama.py (FP32 cache), NOT its
llm/model.py (FP16 cache and in-graph sampling). Defaults to a small synthetic
fixture; --checkpoint selects a local, unscaled Llama HF checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def trajectory_summary(values):
    return dict(shape=list(values.shape), dtype=str(values.dtype),
                sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                argmax_tokens=np.argmax(values, axis=-1).reshape(-1).tolist())


def worker(args):
    case = json.loads((ROOT / 'test/fixtures/llama.json').read_text())['cases'][0]
    cfg = case['config']
    weights = {}
    for name, shape in case['weights'].items():
        values = ((np.arange(np.prod(shape)) * 7 + sum(name.encode())) % 23 - 11) * .017
        weights[name] = (values + (len(shape) == 1)).astype(np.float32).reshape(shape)
    if args.checkpoint:
        cfg = json.loads((args.checkpoint / 'config.json').read_text())
        if cfg.get('model_type') != 'llama' or cfg.get('rope_scaling') or cfg.get('attention_bias') or cfg.get('mlp_bias'):
            raise ValueError('benchmark reference requires unscaled, bias-free Llama')
    tolerance = 3e-4 if args.checkpoint else 3e-5
    prompt = (np.arange(args.chunk, dtype=np.int32) % cfg['vocab_size']).reshape(1, -1)
    token = np.array([[1]], dtype=np.int32)
    if args.worker == 'polygrad':
        import polygrad as pg
        from polygrad import _ffi
        assert Path(_ffi._lib._name).resolve() == (ROOT / 'build/libpolygrad.so').resolve()
        rt = pg.create(device=args.device)
        config = {**cfg, 'max_position_embeddings':args.capacity, 'max_seq_len':args.chunk,
                  'cache_capacity':args.capacity, 'prefill_chunk_size':args.chunk}
        if args.checkpoint:
            net = rt.models.Transformer.from_model(rt.Model.from_hf(
                config_json=json.dumps(config),
                weight_bytes_list=[(args.checkpoint / 'model.safetensors').read_bytes()], max_seq_len=args.chunk))
        else:
            net = rt.models.Llama(config)
            for name, data in weights.items():
                net.write_buffer(name, data)
        def call(n, pos):
            assert net.decode_position == pos
            return net.append_tokens(prompt if n == args.chunk else token)
        reset = net.reset
        def close():
            net.dispose()
            rt.dispose()
        source = str(_ffi._lib._name)
    else:
        import tinygrad
        from tinygrad import Tensor, Variable, TinyJit, dtypes
        from tinygrad.nn.state import load_state_dict, safe_load
        from extra.models.llama import Transformer, convert_from_huggingface
        assert Path(tinygrad.__file__).resolve().is_relative_to((ROOT / 'references/tinygrad_014').resolve())
        net = Transformer(dim=cfg['hidden_size'], hidden_dim=cfg['intermediate_size'],
                          n_heads=cfg['num_attention_heads'], n_kv_heads=cfg['num_key_value_heads'],
                          n_layers=cfg['num_hidden_layers'], norm_eps=cfg['rms_norm_eps'],
                          vocab_size=cfg['vocab_size'], rope_theta=cfg['rope_theta'],
                          max_context=args.capacity, jit=False)
        # HF half-split Q/K weights must be permuted to this reference's pairs.
        state = ({k:v.cast(dtypes.float32) for k,v in safe_load(args.checkpoint / 'model.safetensors').items()} if args.checkpoint else
                 {k:Tensor(v, dtype=dtypes.float32) for k,v in weights.items()})
        if cfg.get('tie_word_embeddings'):
            # Stories15M stores only lm_head; the pin's converter fills the
            # opposite missing alias. Supply this declared tie before mapping.
            if 'model.embed_tokens.weight' not in state:
                state['model.embed_tokens.weight'] = state['lm_head.weight']
            net.output.weight = net.tok_embeddings.weight
        state = convert_from_huggingface(state,
                                        cfg['num_hidden_layers'], cfg['num_attention_heads'], cfg['num_key_value_heads'])
        state['freqs_cis'] = net.freqs_cis
        load_state_dict(net, state, verbose=False)
        positions = {n:Variable(f'position_{n}', 0, args.capacity-n) for n in (args.chunk, 1)}
        def forward(t, p):
            return net(t, p, temperature=float('nan'))[:, -1].contiguous().realize()
        entries = {n:TinyJit(forward) for n in (args.chunk, 1)}
        def call(n, pos):
            data = prompt if n == args.chunk else token
            return entries[n](Tensor(data).realize(), positions[n].bind(pos)).numpy()
        def reset():
            for layer in net.layers:
                if hasattr(layer.attention, 'cache_kv'):
                    cache = layer.attention.cache_kv
                    cache.assign(Tensor.zeros(*cache.shape, device=cache.device, dtype=cache.dtype)).realize()
        def close():
            pass
        source = str(tinygrad.__file__)

    prefill, decode = [], []
    trajectory = []
    try:
        # Reset is outside timing. Every measured pass writes the same prefix;
        # no rewind, stale cache reads or enqueue-only timing is involved.
        for repeat in range(args.warmup + args.repeats):
            reset()
            for pos in [0, *range(args.chunk, args.capacity)]:
                start = time.perf_counter_ns()
                result = call(args.chunk if pos == 0 else 1, pos)
                elapsed = (time.perf_counter_ns() - start) / 1000
                assert np.isfinite(result).all()
                if repeat >= args.warmup:
                    (prefill if pos == 0 else decode).append(elapsed)
                if repeat == args.warmup:
                    trajectory.append(np.array(result, dtype=np.float32, copy=True))
                elif repeat > args.warmup:
                    index = 0 if pos == 0 else pos - args.chunk + 1
                    np.testing.assert_allclose(result, trajectory[index], rtol=tolerance, atol=tolerance)
        trajectory = np.stack(trajectory)
        # Temporary binary data preserves the full numerical comparison without
        # multi-GB JSON reports or vocabulary-sized Python lists over stdout.
        np.save(args.trajectory, trajectory)
        return dict(engine=args.worker, source=source, prefill_us=statistics.median(prefill),
                    decode_us=statistics.median(decode), prefill_samples=len(prefill),
                    decode_samples=len(decode), trajectory=trajectory_summary(trajectory),
                    affinity=sorted(os.sched_getaffinity(0)), load=os.getloadavg())
    finally:
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['CPU', 'CUDA'], default='CPU')
    parser.add_argument('--capacity', type=int, default=64)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--cpu-core', type=int)
    parser.add_argument('--checkpoint', type=Path, help='local unscaled Llama config.json + model.safetensors')
    parser.add_argument('--output', type=Path, default=ROOT / 'temp/kv-benchmark.json')
    parser.add_argument('--worker', choices=['polygrad', 'tinygrad'], help=argparse.SUPPRESS)
    parser.add_argument('--trajectory', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    if not (1 < args.chunk < args.capacity and args.warmup >= 3 and args.repeats > 0 and args.pairs > 0):
        parser.error('require 1 < chunk < capacity, warmup >= 3, and positive repeats/pairs')
    if args.cpu_core is not None:
        os.sched_setaffinity(0, {args.cpu_core})
    if args.worker:
        if args.trajectory is None:
            parser.error('--worker requires --trajectory')
        print(json.dumps(worker(args)))
        return
    rows = []
    env = dict(os.environ, DEV=args.device, POLY_DEV=args.device, BEAM='0', NOOPT='0', JIT='1',
               POLY_LIB=str(ROOT / 'build/libpolygrad.so'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('POLY_DEVICE', None)
    with tempfile.TemporaryDirectory(prefix='polygrad-kv-') as scratch:
        for pair in range(args.pairs):
            runs = {}
            for engine in (['tinygrad', 'polygrad'] if pair % 2 == 0 else ['polygrad', 'tinygrad']):
                env['PYTHONPATH'] = str(ROOT / ('py' if engine == 'polygrad' else 'references/tinygrad_014'))
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', engine,
                           '--device', args.device, '--capacity', str(args.capacity), '--chunk', str(args.chunk),
                           '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                           '--trajectory', str(Path(scratch) / f'{engine}.npy')]
                if args.checkpoint:
                    command += ['--checkpoint', str(args.checkpoint)]
                print(f'pair {pair+1}/{args.pairs}: {engine}', flush=True)
                result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, check=True)
                runs[engine] = json.loads(result.stdout)
            tolerance = 3e-4 if args.checkpoint else 3e-5
            pg_values = np.load(Path(scratch) / 'polygrad.npy')
            tg_values = np.load(Path(scratch) / 'tinygrad.npy')
            np.testing.assert_allclose(pg_values, tg_values, rtol=tolerance, atol=tolerance)
            np.testing.assert_array_equal(np.argmax(pg_values, axis=-1), np.argmax(tg_values, axis=-1))
            runs['comparison'] = dict(max_abs_error=float(np.max(np.abs(pg_values - tg_values))),
                                     rtol=tolerance, atol=tolerance, argmax_equal=True)
            rows.append(runs)
            print({key:round(runs['polygrad'][key] / runs['tinygrad'][key], 3) for key in ('prefill_us','decode_us')}, flush=True)
    report = dict(device=args.device, capacity=args.capacity, chunk=args.chunk, cache_dtype='float32',
                  reference='tinygrad_014/extra/models/llama.py', host_tokens_and_logits=True,
                  includes_sampling=False, pairs=rows,
                  median_pg_over_tg={key:statistics.median(r['polygrad'][key] / r['tinygrad'][key] for r in rows)
                                     for key in ('prefill_us', 'decode_us')})
    report['sha256'] = {path:hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
                        ('build/libpolygrad.so', 'test/fixtures/llama.json', 'bench/bench_kv.py',
                         'references/tinygrad_014/extra/models/llama.py')}
    if args.checkpoint:
        report['checkpoint'] = str(args.checkpoint)
        report['checkpoint_sha256'] = {name:hashlib.sha256((args.checkpoint / name).read_bytes()).hexdigest()
                                       for name in ('config.json', 'model.safetensors')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
