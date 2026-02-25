#!/usr/bin/env python
"""Benchmark: GPT-2 training loop with PyTorch (CPU) for comparison.

Mirrors train_gpt2_python.py configs exactly. Minimal GPT-2 with the
same hyperparameters — no torch.compile, no Flash Attention, pure eager.
"""

import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Minimal GPT-2 matching polygrad's nn.gpt2 ──────────────────────

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim, bias=True)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=True)
        self.c_proj = nn.Linear(4 * dim, dim, bias=True)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, n_heads)
        self.ln_2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = FeedForward(dim)
    def forward(self, x, mask=None):
        h = x + self.attn(self.ln_1(x), mask)
        return h + self.mlp(self.ln_2(h))

class GPT2(nn.Module):
    def __init__(self, n_layers=12, n_heads=12, dim=768,
                 vocab_size=50257, max_seq_len=1024, norm_eps=1e-5):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(max_seq_len, dim)
        self.h = nn.ModuleList([TransformerBlock(dim, n_heads, norm_eps)
                                for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        tok_emb = self.wte(tokens)
        pos_emb = self.wpe(torch.arange(T, device=tokens.device).unsqueeze(0))
        h = tok_emb + pos_emb
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1) * (-1e9)
        mask = mask.unsqueeze(0).unsqueeze(0)
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
    torch.manual_seed(42)
    np.random.seed(42)

    model = GPT2(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        dim=cfg['dim'], vocab_size=cfg['vocab_size'],
        max_seq_len=cfg['seq_len'] * 2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tokens = torch.from_numpy(
        np.random.randint(0, cfg['vocab_size'], (1, cfg['seq_len']))
    ).long()

    times = []
    for i in range(N_WARMUP + cfg['iters']):
        t0 = time.perf_counter()

        logits = model(tokens)
        loss = (logits * logits).sum()
        loss_val = loss.item()

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
print(f"GPT-2 Training Benchmark (PyTorch {torch.__version__} CPU, eager)")
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
print(f"PyTorch {torch.__version__} CPU eager (no torch.compile, no Flash Attention)")
print(f"{'='*70}")
