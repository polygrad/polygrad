"""nn.gpt2 — GPT-2 model (pre-norm transformer, tinygrad-compatible).

Port of tinygrad's examples/gpt2.py, simplified for training
(no KV cache, no half precision, no JIT).
"""

import math
from ..tensor import Tensor
from .modules import Linear, LayerNorm, Embedding


# GPT-2 model configurations (from OpenAI)
GPT2_CONFIGS = {
    'gpt2':        dict(n_layers=12, n_heads=12, dim=768,  vocab_size=50257),
    'gpt2-medium': dict(n_layers=24, n_heads=16, dim=1024, vocab_size=50257),
    'gpt2-large':  dict(n_layers=36, n_heads=20, dim=1280, vocab_size=50257),
    'gpt2-xl':     dict(n_layers=48, n_heads=25, dim=1600, vocab_size=50257),
    # Tiny configs for benchmarking
    'gpt2-tiny':   dict(n_layers=2,  n_heads=2,  dim=64,   vocab_size=256),
    'gpt2-small':  dict(n_layers=4,  n_heads=4,  dim=128,  vocab_size=512),
}


class Attention:
    """Multi-head scaled dot-product attention."""
    def __init__(self, dim, n_heads):
        self.c_attn = Linear(dim, 3 * dim, bias=True)
        self.c_proj = Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

    def __call__(self, x, mask=None):
        B, T, C = x.shape

        # Q, K, V projection: (B, T, 3*dim) → 3 × (B, T, dim)
        # Realize after matmul to prevent fusion with downstream ops
        qkv = self.c_attn(x).realize()
        q = qkv.shrink(((0, B), (0, T), (0, C)))
        k = qkv.shrink(((0, B), (0, T), (C, 2 * C)))
        v = qkv.shrink(((0, B), (0, T), (2 * C, 3 * C)))

        # Realize Q, K, V after shrink to materialize before reshape+permute
        q = q.realize()
        k = k.realize()
        v = v.realize()

        # Reshape to multi-head: (B, T, n_heads, head_dim) → (B, n_heads, T, head_dim)
        q = q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention — realize after matmul before mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = q.matmul(k.transpose(-2, -1)).realize()
        scores = scores * scale

        if mask is not None:
            scores = (scores + mask).realize()

        # Softmax: reduce→expand→alu pattern, must realize
        attn = scores.softmax(axis=-1).realize()  # (B, H, T, T)

        # Attention-weighted values + merge heads
        out = attn.matmul(v).realize()  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(out)


class FeedForward:
    """Position-wise feed-forward network."""
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim, bias=True)
        self.c_proj = Linear(hidden_dim, dim, bias=True)

    def __call__(self, x):
        # Realize after matmul and gelu to prevent fusion
        h = self.c_fc(x).realize()
        h = h.gelu().realize()
        return self.c_proj(h).realize()


class TransformerBlock:
    """Pre-norm transformer block with residual connections."""
    def __init__(self, dim, n_heads, norm_eps=1e-5):
        self.attn = Attention(dim, n_heads)
        self.mlp = FeedForward(dim, 4 * dim)
        self.ln_1 = LayerNorm(dim, eps=norm_eps)
        self.ln_2 = LayerNorm(dim, eps=norm_eps)

    def __call__(self, x, mask=None):
        # Realize after each major stage to prevent reduce fusion
        # LayerNorm has reduce→expand→alu — must realize before attention
        ln1 = self.ln_1(x).realize()
        attn_out = self.attn(ln1, mask).realize()
        h = (x + attn_out).realize()
        ln2 = self.ln_2(h).realize()
        mlp_out = self.mlp(ln2)
        return (h + mlp_out).realize()


class GPT2:
    """GPT-2 transformer language model.

    Pre-norm architecture matching OpenAI's GPT-2:
    - Token + positional embeddings
    - N transformer blocks (pre-norm attention + MLP)
    - Final layer norm + linear head to vocabulary
    """
    def __init__(self, n_layers=12, n_heads=12, dim=768,
                 vocab_size=50257, max_seq_len=1024, norm_eps=1e-5):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(max_seq_len, dim)
        self.h = [TransformerBlock(dim, n_heads, norm_eps)
                  for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim, eps=norm_eps)
        self.lm_head = Linear(dim, vocab_size, bias=False)

    def __call__(self, tokens):
        """Forward pass for training.

        tokens: (batch, seq_len) float tensor of token indices.
        Returns logits: (batch, seq_len, vocab_size).
        """
        B, T = tokens.shape

        # Token embeddings (realize to prevent reduce fusion)
        tok_emb = self.wte(tokens).realize()  # (B, T, dim)

        # Position embeddings (realize to prevent reduce fusion)
        positions = Tensor.arange(T).reshape(1, T)
        pos_emb = self.wpe(positions).realize()  # (1, T, dim)

        h = (tok_emb + pos_emb).realize()

        # Causal mask: upper triangle filled with -inf
        mask = (Tensor.ones(T, T).triu(diagonal=1) * (-1e9)).realize()
        mask = mask.reshape(1, 1, T, T)

        # Transformer blocks (realize between blocks to prevent cross-block fusion)
        for block in self.h:
            h = block(h, mask)
            h = h.realize()

        # Final norm + logit projection (realize layernorm before linear)
        h = self.ln_f(h).realize()
        logits = self.lm_head(h)  # (B, T, vocab_size)
        return logits

    @staticmethod
    def from_config(name):
        """Create a GPT-2 model from a named config."""
        cfg = GPT2_CONFIGS[name]
        return GPT2(**cfg)
