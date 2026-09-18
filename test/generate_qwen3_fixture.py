"""Small uncached Qwen3 oracle and F32 GGUF for cross-frontend Model tests.

Run in the Torch/Transformers reference environment; emits JSON to stdout.
Pinned tinygrad's rotary helper is checked separately: its Qwen attention path
uses a half-precision KV cache, unlike Polygrad's uncached float32 Model.
"""
import base64
import json
import struct

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

torch.set_num_threads(1)
config = Qwen3Config(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                    vocab_size=11, rms_norm_eps=1e-6, rope_theta=1e6,
                    max_position_embeddings=4, tie_word_embeddings=True,
                    attention_bias=False, attn_implementation='eager')
net = Qwen3ForCausalLM(config).float().eval()
names = {'model.embed_tokens.weight': 'token_embd.weight',
         'model.norm.weight': 'output_norm.weight'}
for hf, gguf in [('input_layernorm', 'attn_norm'), ('post_attention_layernorm', 'ffn_norm'),
                 ('self_attn.q_proj', 'attn_q'), ('self_attn.k_proj', 'attn_k'),
                 ('self_attn.v_proj', 'attn_v'), ('self_attn.o_proj', 'attn_output'),
                 ('self_attn.q_norm', 'attn_q_norm'), ('self_attn.k_norm', 'attn_k_norm'),
                 ('mlp.gate_proj', 'ffn_gate'), ('mlp.up_proj', 'ffn_up'), ('mlp.down_proj', 'ffn_down')]:
    names[f'model.layers.0.{hf}.weight'] = f'blk.0.{gguf}.weight'
weights = {}
with torch.no_grad():
    for name, parameter in net.named_parameters():
        mapped = names[name]
        offset = sum(mapped.encode())
        values = [(1.0 if 'norm' in mapped else 0.0) + ((i*7+offset) % 23-11)*0.017
                  for i in range(parameter.numel())]
        parameter.copy_(torch.tensor(values).reshape(parameter.shape))
        weights[mapped] = parameter.detach().numpy()

def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw

kv = {'general.architecture': 'qwen3', 'qwen3.embedding_length': 16,
      'qwen3.attention.head_count': 2, 'qwen3.attention.head_count_kv': 1,
      'qwen3.block_count': 1, 'qwen3.feed_forward_length': 24,
      'qwen3.context_length': 4, 'qwen3.rope.freq_base': 1e6,
      'qwen3.attention.layer_norm_rms_epsilon': 1e-6}
data = b'GGUF' + struct.pack('<IQQ', 3, len(weights), len(kv))
for key, value in kv.items():
    data += string(key)
    data += (struct.pack('<I', 8) + string(value) if isinstance(value, str) else
             struct.pack('<If', 6, value) if isinstance(value, float) else struct.pack('<II', 4, value))
payload = b''
for name, value in weights.items():
    data += string(name) + struct.pack('<I', value.ndim)
    data += b''.join(struct.pack('<Q', dim) for dim in reversed(value.shape))
    data += struct.pack('<IQ', 0, len(payload))
    payload += value.astype('<f4').tobytes()
    payload += bytes((-len(payload)) % 32)
data += bytes((-len(data)) % 32) + payload
tokens = [[1, 4, 2, 7]]
with torch.no_grad():
    logits = net(torch.tensor(tokens), use_cache=False).logits.numpy().tolist()
    cos, sin = net.model.rotary_emb(torch.zeros(1,4,16), torch.arange(4).reshape(1,4))
print(json.dumps(dict(reference=f'transformers {transformers.__version__} Qwen3ForCausalLM float32, use_cache=False',
                      gguf=base64.b64encode(data).decode(), tokens=tokens, logits=logits,
                      rope_cos=cos[0,:,:4].numpy().tolist(), rope_sin=sin[0,:,:4].numpy().tolist()), indent=2))
