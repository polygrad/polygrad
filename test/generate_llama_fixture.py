"""Small deterministic Llama oracle from pinned extra/models/llama.py.

Run with the pinned Tinygrad PYTHONPATH/interpreter. Output is shared by the C,
Python and JS family tests; this is not a pretrained checkpoint.
"""
import argparse
import json
import math


def fixture(kv_heads, version, engine):
    dim = 16 if version == '3.2' else 8
    config = dict(model_type="llama", hidden_size=dim, intermediate_size=dim*3//2,
                  num_attention_heads=2, num_key_value_heads=kv_heads,
                  num_hidden_layers=2, vocab_size=11, rms_norm_eps=1e-5,
                  rope_theta=10000 if version == '2' else 500000, max_position_embeddings=16,
                  batch_size=1, max_seq_len=3, tie_word_embeddings=version == '3.2')
    if version == '3.2':
        config['rope_scaling'] = dict(rope_type='llama3', factor=32.0, low_freq_factor=1.0,
                                     high_freq_factor=4.0, original_max_position_embeddings=8192)
    weights = {}
    def weight(name, shape, norm=False):
        offset = sum(name.encode())
        data = [(1.0 if norm else 0.0) + ((i * 7 + offset) % 23 - 11) * 0.017
                for i in range(math.prod(shape))]
        weights[name] = dict(shape=shape, data=data)
    weight("model.embed_tokens.weight", [11, dim])
    for i in range(2):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight", [dim], True)
        weight(p + "post_attention_layernorm.weight", [dim], True)
        for name, size in (("q", dim), ("k", kv_heads * dim//2), ("v", kv_heads * dim//2), ("o", dim)):
            weight(p + f"self_attn.{name}_proj.weight", [size, dim])
        weight(p + "mlp.gate_proj.weight", [dim*3//2, dim])
        weight(p + "mlp.up_proj.weight", [dim*3//2, dim])
        weight(p + "mlp.down_proj.weight", [dim, dim*3//2])
    weight("model.norm.weight", [dim], True)
    if not config['tie_word_embeddings']: weight("lm_head.weight", [11, dim])
    tokens = [[1, 4, 2]]
    if engine == 'tinygrad':
        from tinygrad import Tensor, dtypes
        from tinygrad.nn.state import load_state_dict
        from extra.models.llama import Transformer, convert_from_huggingface
        net = Transformer(dim=dim, hidden_dim=dim*3//2, n_heads=2, n_kv_heads=kv_heads,
                          n_layers=2, norm_eps=1e-5, vocab_size=11, max_context=3,
                          rope_theta=config['rope_theta'], jit=False, disable_kv_cache=True)
        tensors = {k: Tensor(v['data'], dtype=dtypes.float32).reshape(v['shape']) for k,v in weights.items()}
        state = convert_from_huggingface(tensors, 2, 2, kv_heads)
        state['freqs_cis'] = net.freqs_cis
        load_state_dict(net, state, verbose=False)
        def forward(t): return net(Tensor(t), 0, temperature=float('nan')).numpy().reshape(-1).tolist()
        freqs_cos = net.freqs_cis.numpy()[0, :3, 0, :, 0].reshape(-1).tolist()
        freqs_sin = net.freqs_cis.numpy()[0, :3, 0, :, 1].reshape(-1).tolist()
    else:
        import torch
        import transformers
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        net = LlamaForCausalLM(LlamaConfig(**config, attn_implementation='eager')).eval()
        tensors = {k: torch.tensor(v['data'], dtype=torch.float32).reshape(v['shape']) for k,v in weights.items()}
        if config['tie_word_embeddings']: tensors['lm_head.weight'] = tensors['model.embed_tokens.weight']
        net.load_state_dict(tensors, strict=True)
        def forward(t):
            with torch.no_grad(): return net(torch.tensor(t), use_cache=False).logits.reshape(-1).tolist()
        with torch.no_grad():
            cos, sin = net.model.rotary_emb(torch.zeros(1,3,dim), torch.arange(3).reshape(1,3))
            freqs_cos = cos[0,:,:dim//4].reshape(-1).tolist()
            freqs_sin = sin[0,:,:dim//4].reshape(-1).tolist()
    logits = forward(tokens)
    # A later token cannot change earlier logits. Exercise a nonzero position so
    # half-split/interleaved Q/K mistakes cannot pass at position zero alone.
    changed = forward([[1, 4, 7]])
    assert logits[:22] == changed[:22]
    return dict(version=version, config=config, tokens=tokens,
                weights={k:v['shape'] for k,v in weights.items()}, logits=logits,
                freqs_cos=freqs_cos, freqs_sin=freqs_sin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=('tinygrad', 'transformers'), default='tinygrad')
    args = parser.parse_args()
    cases = [(1,'2'), (2,'2'), (1,'3')]
    if args.engine == 'transformers': cases += [(1,'3.2')]
    print(json.dumps({'reference': args.engine,
                      'cases': [fixture(kv, ver, args.engine) for kv,ver in cases]}, indent=2, allow_nan=False))
