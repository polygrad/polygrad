"""Pinned Tinygrad oracle for the shared JSON component fixture; writes JSON to stdout.

Run with PYTHONPATH=references/tinygrad_latest and the pinned Python interpreter.
The fixture uses two-element heads, where split-half and interleaved RoPE agree;
tensor_graph_cases.py separately checks the four-element split-half UOp program.
"""
import json
import numpy as np
from tinygrad import Tensor, Variable
from tinygrad.nn import Embedding, RMSNorm, LayerNorm
from extra.models.llama import precompute_freqs_cis, apply_rotary_emb

table = np.arange(16, dtype=np.float32).reshape(4, 4) * .03 - .2
linear = np.arange(8, dtype=np.float32).reshape(2, 4) * .05 - .1
cases = []
for batch in (1, 3):
  tokens = np.arange(batch * 3, dtype=np.int32).reshape(batch, 3) % 4
  emb, rms, ln = Embedding(4, 4), RMSNorm(4), LayerNorm(4)
  emb.weight = Tensor(table)
  h = rms(emb(Tensor(tokens)))
  h = h.reshape(batch, 3, 2, 2)
  h, _ = apply_rotary_emb(h, h, precompute_freqs_cis(2, 3))
  h = h.permute(0, 2, 1, 3)
  h = h.scaled_dot_product_attention(h, h, is_causal=True)
  h = ln(h.permute(0, 2, 1, 3).reshape(batch, 3, 4))
  y = h.linear(Tensor(linear).T)
  cases.append(dict(batch=batch, tokens=tokens.tolist(), prediction=y.numpy().tolist(), mean=y.mean().item()))
n = Variable('batch', 1, 3)
x = Tensor.ones(n.bind(3), 4)
assert x.mean().item() == 1.0
print(json.dumps(dict(reference='tinygrad v0.14.0 6f87158d77f66a36d5f8bbe915170b24e2acabe8',
                     table=table.tolist(), linear=linear.tolist(), cases=cases), indent=2))
