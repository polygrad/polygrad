#!/usr/bin/env python
"""Plot GPT-2 training: loss vs cumulative time for all frameworks."""

import subprocess
import json
import sys
import os
import numpy as np

# ── Collect per-iteration data from each framework ──────────────────

def run_polygrad(device, cfg):
    """Run polygrad benchmark, return list of (cumtime_ms, loss)."""
    code = f"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname('{__file__}'), '..', 'py'))
import numpy as np
from polygrad import Tensor
from polygrad.device import Device
from polygrad.nn import Adam, get_parameters
from polygrad.nn.gpt2 import GPT2

Device.DEFAULT = '{device}'
Tensor.manual_seed(42)
np.random.seed(42)

model = GPT2(n_layers={cfg['n_layers']}, n_heads={cfg['n_heads']},
             dim={cfg['dim']}, vocab_size={cfg['vocab_size']},
             max_seq_len={cfg['seq_len']*2})
params = get_parameters(model)
opt = Adam(params, lr=0.001)
tokens = np.random.randint(0, {cfg['vocab_size']}, (1, {cfg['seq_len']})).astype(np.float32)

results = []
cum = 0.0
for i in range({cfg['iters']}):
    t0 = time.perf_counter()
    opt.zero_grad()
    x = Tensor(tokens)
    logits = model(x)
    logits.realize()
    loss = (logits * logits).sum()
    loss.realize()
    loss.backward()
    lv = loss.item()
    opt.step()
    elapsed = (time.perf_counter() - t0) * 1000
    cum += elapsed
    results.append({{'t': cum, 'loss': lv}})

print(json.dumps(results))
"""
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  polygrad {device} failed: {r.stderr[:200]}", file=sys.stderr)
        return []
    return json.loads(r.stdout.strip())


def run_pytorch(device, cfg):
    """Run PyTorch benchmark, return list of (cumtime_ms, loss)."""
    cuda_setup = ""
    if device == 'CUDA':
        cuda_setup = """
dev = torch.device('cuda')
sync = torch.cuda.synchronize
"""
    else:
        cuda_setup = """
dev = torch.device('cpu')
sync = lambda: None
"""

    code = f"""
import time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
{cuda_setup}

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim); self.c_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads; self.head_dim = dim // n_heads
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores + mask
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim); self.attn = Attention(dim, n_heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.c_fc = nn.Linear(dim, 4*dim); self.c_proj = nn.Linear(4*dim, dim)
    def forward(self, x, mask=None):
        h = x + self.attn(self.ln_1(x), mask)
        return h + self.c_proj(F.gelu(self.c_fc(self.ln_2(h))))

class GPT2(nn.Module):
    def __init__(self, n_layers, n_heads, dim, vocab_size, max_seq_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(max_seq_len, dim)
        self.h = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, tokens):
        B, T = tokens.shape
        h = self.wte(tokens) + self.wpe(torch.arange(T, device=tokens.device).unsqueeze(0))
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1) * -1e9
        mask = mask.unsqueeze(0).unsqueeze(0)
        for block in self.h: h = block(h, mask)
        return self.lm_head(self.ln_f(h))

torch.manual_seed(42)
np.random.seed(42)
model = GPT2({cfg['n_layers']}, {cfg['n_heads']}, {cfg['dim']}, {cfg['vocab_size']}, {cfg['seq_len']*2}).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
tokens = torch.from_numpy(np.random.randint(0, {cfg['vocab_size']}, (1, {cfg['seq_len']}))).long().to(dev)

results = []
cum = 0.0
for i in range({cfg['iters']}):
    sync()
    t0 = time.perf_counter()
    logits = model(tokens)
    loss = (logits * logits).sum()
    sync()
    lv = loss.item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    sync()
    elapsed = (time.perf_counter() - t0) * 1000
    cum += elapsed
    results.append({{'t': cum, 'loss': lv}})

print(json.dumps(results))
"""
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  pytorch {device} failed: {r.stderr[:200]}", file=sys.stderr)
        return []
    return json.loads(r.stdout.strip())


def run_tinygrad(device, cfg):
    """Run tinygrad benchmark, return list of (cumtime_ms, loss)."""
    code = f"""
import os, sys, time, json, math
os.environ['DEBUG'] = '0'
sys.path.insert(0, os.path.join('{os.path.dirname(os.path.abspath(__file__))}', '..', 'references', 'tinygrad'))
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
Device.DEFAULT = '{"CUDA" if device == "CUDA" else "CPU"}'

class Linear:
    def __init__(self, in_f, out_f, bias=True):
        b = 1/math.sqrt(in_f)
        self.weight = Tensor.uniform(out_f, in_f, low=-b, high=b)
        self.bias = Tensor.uniform(out_f, low=-b, high=b) if bias else None
    def __call__(self, x): return x.linear(self.weight.T, self.bias)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.weight = Tensor.ones(dim); self.bias = Tensor.zeros(dim); self.eps = eps
    def __call__(self, x): return x.layernorm(eps=self.eps) * self.weight + self.bias

class Embedding:
    def __init__(self, vs, dim):
        self.weight = Tensor.uniform(vs, dim, low=-1, high=1); self.vs = vs
    def __call__(self, idx): return idx.one_hot(self.vs).float() @ self.weight

class Attention:
    def __init__(self, dim, nh):
        self.c_attn = Linear(dim, 3*dim); self.c_proj = Linear(dim, dim)
        self.nh = nh; self.hd = dim // nh
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).reshape(B, T, 3, self.nh, self.hd).permute(2,0,3,1,4)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
        if mask is not None: scores = scores + mask
        out = (scores.softmax(axis=-1) @ v).transpose(1,2).reshape(B,T,C)
        return self.c_proj(out)

class TransformerBlock:
    def __init__(self, dim, nh):
        self.ln_1 = LayerNorm(dim); self.attn = Attention(dim, nh)
        self.ln_2 = LayerNorm(dim)
        self.c_fc = Linear(dim, 4*dim); self.c_proj = Linear(4*dim, dim)
    def __call__(self, x, mask=None):
        h = x + self.attn(self.ln_1(x), mask)
        return h + self.c_proj(self.c_fc(self.ln_2(h)).gelu())

class GPT2:
    def __init__(self, nl, nh, dim, vs, msl):
        self.wte = Embedding(vs, dim); self.wpe = Embedding(msl, dim)
        self.h = [TransformerBlock(dim, nh) for _ in range(nl)]
        self.ln_f = LayerNorm(dim); self.lm_head = Linear(dim, vs, bias=False)
    def __call__(self, tokens):
        B, T = tokens.shape
        h = self.wte(tokens) + self.wpe(Tensor.arange(T).reshape(1, T))
        mask = Tensor.ones(T,T).triu(diagonal=1) * -1e9
        mask = mask.reshape(1,1,T,T)
        for block in self.h: h = block(h, mask)
        return self.lm_head(self.ln_f(h))

Tensor.manual_seed(42); Tensor.training = True
np.random.seed(42)
model = GPT2({cfg['n_layers']}, {cfg['n_heads']}, {cfg['dim']}, {cfg['vocab_size']}, {cfg['seq_len']*2})
params = get_parameters(model)
opt = Adam(params, lr=0.001)
tokens = np.random.randint(0, {cfg['vocab_size']}, (1, {cfg['seq_len']}))

results = []
cum = 0.0
for i in range({cfg['iters']}):
    t0 = time.perf_counter()
    x = Tensor(tokens)
    logits = model(x)
    loss = (logits * logits).sum()
    lv = loss.numpy().item()
    loss.backward()
    opt.step()
    opt.zero_grad()
    elapsed = (time.perf_counter() - t0) * 1000
    cum += elapsed
    results.append({{'t': cum, 'loss': lv}})

print(json.dumps(results))
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'references', 'tinygrad')
    r = subprocess.run(
        ['conda', 'run', '-n', 'tiny', 'python', '-c', code],
        capture_output=True, text=True, timeout=600, env=env
    )
    if r.returncode != 0:
        print(f"  tinygrad {device} failed: {r.stderr[:300]}", file=sys.stderr)
        return []
    # Filter out conda stderr noise
    for line in r.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('['):
            return json.loads(line)
    return []


# ── Main ────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIGS = [
    {'name': 'tiny-1L',  'n_layers': 1, 'n_heads': 1, 'dim': 32,
     'vocab_size': 50, 'seq_len': 4, 'iters': 15},
    {'name': 'tiny-2L',  'n_layers': 2, 'n_heads': 2, 'dim': 64,
     'vocab_size': 256, 'seq_len': 8, 'iters': 10},
]

COLORS = {
    'polygrad': '#2563eb',   # blue
    'pytorch':  '#dc2626',   # red
    'tinygrad': '#16a34a',   # green
}

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Collecting data for {cfg['name']}...")
    print(f"{'='*60}")

    series = {}

    # Polygrad
    print("  polygrad CPU...", end=' ', flush=True)
    series['polygrad_cpu'] = run_polygrad('CPU', cfg)
    print(f"{len(series['polygrad_cpu'])} pts")

    print("  polygrad CUDA...", end=' ', flush=True)
    series['polygrad_cuda'] = run_polygrad('CUDA', cfg)
    print(f"{len(series['polygrad_cuda'])} pts")

    # PyTorch
    print("  pytorch CPU...", end=' ', flush=True)
    series['pytorch_cpu'] = run_pytorch('CPU', cfg)
    print(f"{len(series['pytorch_cpu'])} pts")

    print("  pytorch CUDA...", end=' ', flush=True)
    series['pytorch_cuda'] = run_pytorch('CUDA', cfg)
    print(f"{len(series['pytorch_cuda'])} pts")

    # tinygrad
    print("  tinygrad CPU...", end=' ', flush=True)
    series['tinygrad_cpu'] = run_tinygrad('CPU', cfg)
    print(f"{len(series['tinygrad_cpu'])} pts")

    print("  tinygrad CUDA...", end=' ', flush=True)
    series['tinygrad_cuda'] = run_tinygrad('CUDA', cfg)
    print(f"{len(series['tinygrad_cuda'])} pts")

    # ── Plot: x=cumulative wall time (warm only), y=raw loss ──────
    fig, ax = plt.subplots(figsize=(11, 6))

    for key, data in series.items():
        if not data or len(data) < 2:
            continue
        fw, dev = key.rsplit('_', 1)
        # Skip cold iteration, use per-iteration elapsed time for x-axis
        warm = data[1:]
        # Compute per-iteration durations, then cumulate from 0
        durations = []
        for j in range(len(warm)):
            if j == 0:
                durations.append(warm[0]['t'] - data[0]['t'])  # iter1 time
            else:
                durations.append(warm[j]['t'] - warm[j-1]['t'])
        cum_t = []
        s = 0
        for d in durations:
            s += d
            cum_t.append(s / 1000)  # ms → seconds

        losses = [d['loss'] for d in warm]
        # Skip NaN
        valid = [(t, l) for t, l in zip(cum_t, losses) if l == l]
        if not valid:
            continue
        ts_v, losses_v = zip(*valid)

        color = COLORS[fw]
        style = '-' if dev == 'cuda' else '--'
        lw = 2.5 if dev == 'cuda' else 1.8
        avg_ms = sum(durations) / len(durations)
        label = f"{fw} {'GPU' if dev == 'cuda' else 'CPU'} ({avg_ms:.0f} ms/it)"
        ax.plot(ts_v, losses_v, style, color=color, linewidth=lw, label=label,
                marker='o', markersize=4, alpha=0.9)

    ax.set_xlabel('Wall time (seconds, excluding cold start)', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title(f'GPT-2 Training: Loss vs Time — {cfg["name"]} '
                 f'(L={cfg["n_layers"]}, H={cfg["n_heads"]}, D={cfg["dim"]}, '
                 f'V={cfg["vocab_size"]}, T={cfg["seq_len"]})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'bench/gpt2_loss_vs_time_{cfg["name"]}.png'
    fig.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")
    plt.close(fig)

print(f"\nDone.")
