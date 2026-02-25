#!/usr/bin/env python
"""Benchmark: GPT-2 training loop with tinygrad (CUDA GPU) for comparison."""

import time
import sys
import os
import numpy as np
import math

os.environ['DEBUG'] = '0'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'references', 'tinygrad'))
from tinygrad import Tensor, Device
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
Device.DEFAULT = 'CUDA'

# ── Minimal GPT-2 matching polygrad's nn.gpt2 ──────────────────────

class Linear:
    def __init__(self, in_f, out_f, bias=True):
        bound = 1 / math.sqrt(in_f)
        self.weight = Tensor.uniform(out_f, in_f, low=-bound, high=bound)
        self.bias = Tensor.uniform(out_f, low=-bound, high=bound) if bias else None
    def __call__(self, x):
        return x.linear(self.weight.T, self.bias)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)
        self.eps = eps
    def __call__(self, x):
        return x.layernorm(eps=self.eps) * self.weight + self.bias

class Embedding:
    def __init__(self, vocab_size, dim):
        self.weight = Tensor.uniform(vocab_size, dim, low=-1, high=1)
        self.vocab_size = vocab_size
    def __call__(self, idx):
        return idx.one_hot(self.vocab_size).float() @ self.weight

class Attention:
    def __init__(self, dim, n_heads):
        self.c_attn = Linear(dim, 3 * dim, bias=True)
        self.c_proj = Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = scores.softmax(axis=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)

class FeedForward:
    def __init__(self, dim):
        self.c_fc = Linear(dim, 4 * dim, bias=True)
        self.c_proj = Linear(4 * dim, dim, bias=True)
    def __call__(self, x):
        return self.c_proj(self.c_fc(x).gelu())

class TransformerBlock:
    def __init__(self, dim, n_heads, norm_eps=1e-5):
        self.ln_1 = LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, n_heads)
        self.ln_2 = LayerNorm(dim, eps=norm_eps)
        self.mlp = FeedForward(dim)
    def __call__(self, x, mask=None):
        h = x + self.attn(self.ln_1(x), mask)
        return h + self.mlp(self.ln_2(h))

class GPT2:
    def __init__(self, n_layers=12, n_heads=12, dim=768,
                 vocab_size=50257, max_seq_len=1024, norm_eps=1e-5):
        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(max_seq_len, dim)
        self.h = [TransformerBlock(dim, n_heads, norm_eps)
                  for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim, eps=norm_eps)
        self.lm_head = Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        tok_emb = self.wte(tokens)
        positions = Tensor.arange(T).reshape(1, T)
        pos_emb = self.wpe(positions)
        h = tok_emb + pos_emb
        mask = Tensor.ones(T, T).triu(diagonal=1) * (-1e9)
        mask = mask.reshape(1, 1, T, T)
        for block in self.h:
            h = block(h, mask)
        h = self.ln_f(h)
        return self.lm_head(h)

# ── Benchmark ────────────────────────────────────────────────────────

CONFIGS = [
    {'name': 'tiny-1L',  'n_layers': 1, 'n_heads': 1, 'dim': 32,
     'vocab_size': 50, 'seq_len': 4, 'iters': 5},
    {'name': 'tiny-2L',  'n_layers': 2, 'n_heads': 2, 'dim': 64,
     'vocab_size': 256, 'seq_len': 8, 'iters': 3},
]

LR = 0.001
N_WARMUP = 1


def run_config(cfg):
    Tensor.manual_seed(42)
    Tensor.training = True
    np.random.seed(42)

    model = GPT2(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        dim=cfg['dim'], vocab_size=cfg['vocab_size'],
        max_seq_len=cfg['seq_len'] * 2,
    )
    params = get_parameters(model)
    n_params = sum(p.numel() for p in params)
    opt = Adam(params, lr=LR)

    tokens = np.random.randint(0, cfg['vocab_size'],
                               (1, cfg['seq_len']))

    times = []
    for i in range(N_WARMUP + cfg['iters']):
        t0 = time.perf_counter()

        x = Tensor(tokens)
        logits = model(x)
        loss = (logits * logits).sum()
        loss_val = loss.numpy().item()

        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad()
        t3 = time.perf_counter()

        rec = {
            'forward': (t1 - t0) * 1000,
            'backward': (t2 - t1) * 1000,
            'step': (t3 - t2) * 1000,
            'total': (t3 - t0) * 1000,
            'loss': loss_val,
        }
        times.append(rec)

    return times, n_params


print(f"{'='*70}")
print(f"GPT-2 Training Benchmark (tinygrad CUDA GPU)")
print(f"{'='*70}\n")

for cfg in CONFIGS:
    name = cfg['name']
    print(f"--- {name}: GPT2(L={cfg['n_layers']}, H={cfg['n_heads']}, "
          f"D={cfg['dim']}, V={cfg['vocab_size']}, T={cfg['seq_len']}) ---")

    times, n_params = run_config(cfg)
    print(f"  Parameters: {n_params:,} weights")

    cold = times[0]
    warm = times[N_WARMUP:]

    def avg(recs, key):
        return sum(r[key] for r in recs) / len(recs)

    print(f"  Cold (iter 0):  fwd={cold['forward']:8.1f}ms  "
          f"bwd={cold['backward']:8.1f}ms  step={cold['step']:8.1f}ms  "
          f"total={cold['total']:8.1f}ms")
    print(f"  Warm (avg):     fwd={avg(warm,'forward'):8.1f}ms  "
          f"bwd={avg(warm,'backward'):8.1f}ms  step={avg(warm,'step'):8.1f}ms  "
          f"total={avg(warm,'total'):8.1f}ms")
    print(f"  Loss:           {cold['loss']:.4f} -> {warm[-1]['loss']:.4f}")
    print(f"  Throughput:     {1000/avg(warm,'total'):.2f} iter/sec (warm)")
    print()

print(f"{'='*70}")
print(f"tinygrad CUDA backend (ref commit c2be31e)")
print(f"{'='*70}")
