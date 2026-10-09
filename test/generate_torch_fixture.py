"""Small real torch.save archives; generation requires Torch, ordinary tests do not."""
import base64
from collections import OrderedDict
import io
import json
import torch


def archive(state, protocol=2):
    out = io.BytesIO()
    torch.save(state, out, pickle_protocol=protocol)
    return base64.b64encode(out.getvalue()).decode()


def raw(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()


def main():
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    state = OrderedDict(contiguous=base, transpose=base.T, slice=base[1:, ::2],
                        broadcast=base[:1, :1].expand(2, 3), scalar=base[1, 2],
                        empty=base[:0], alias=base)
    names = ['float32', 'float16', 'bfloat16', 'float64', 'int64', 'int32',
             'int16', 'int8', 'uint8', 'bool']
    for name in names:
        state[name] = torch.tensor([0, 1, 2], dtype=getattr(torch, name))
    state._metadata = OrderedDict([('', {'version': 1})])
    cases = []
    for protocol in [2, 4, 5]:
        encoded = archive(state, protocol)
        # Compare the independently loaded checkpoint, not only the authoring values.
        # Torch's restricted loader does not implement all protocol 4/5 opcodes.
        # These objects were constructed above; no external pickle is executed.
        loaded = torch.load(io.BytesIO(base64.b64decode(encoded)), weights_only=protocol == 2)
        cases.append(dict(protocol=protocol, archive=encoded, tensors=[
            dict(name=k, shape=list(v.shape), dtype=names.index(str(v.dtype).split('.')[1]),
                 hex=raw(v)) for k, v in loaded.items()]))
    config = dict(model_type='gpt2', n_embd=4, n_layer=1, n_head=1,
                  n_positions=4, vocab_size=8, layer_norm_epsilon=1e-5)
    shapes = {'wte.weight': (8, 4), 'wpe.weight': (4, 4), 'ln_f.weight': (4,), 'ln_f.bias': (4,)}
    shapes.update({'h.0.' + k: v for k, v in {
        'ln_1.weight': (4,), 'ln_1.bias': (4,), 'ln_2.weight': (4,), 'ln_2.bias': (4,),
        'attn.c_attn.weight': (4, 12), 'attn.c_attn.bias': (12,),
        'attn.c_proj.weight': (4, 4), 'attn.c_proj.bias': (4,),
        'mlp.c_fc.weight': (4, 16), 'mlp.c_fc.bias': (16,),
        'mlp.c_proj.weight': (16, 4), 'mlp.c_proj.bias': (4,)}.items()})
    weights = OrderedDict()
    for name, shape in shapes.items():
        n = 1
        for d in shape: n *= d
        weights['transformer.' + name] = ((torch.arange(n) % 13 - 6) * .01).reshape(shape)
    print(json.dumps(dict(reference='torch ' + torch.__version__, cases=cases,
                         model=dict(config=config, archive=archive(weights), tensors=[
                             dict(name=k, shape=list(v.shape), dtype=0, hex=raw(v))
                             for k, v in weights.items()])), indent=2))


if __name__ == '__main__':
    main()
