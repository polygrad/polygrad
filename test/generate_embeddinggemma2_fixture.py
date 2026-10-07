"""Generate a small official-HF oracle for all Polygrad frontends.

Run with the pinned Transformers checkout described in the output provenance.
No pretrained checkpoint is needed.
"""
import base64
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import save

sys.path.insert(0, str(Path(__file__).parent / 'external'))
from test_embeddinggemma2_hf import tiny_reference, tiny_image_reference, tiny_image_inputs, tiny_audio_reference, tiny_audio_inputs


def main():
    model = tiny_reference()
    cases = []
    for tokens in [
        [[2, 7, 4, 3, 1, 0], [2, 8, 1, 0, 0, 0]],
        [[0, 2, 7, 4, 3, 1], [0, 0, 0, 2, 8, 1]],
        [[2, 7, 4, 3, 1, 8], [2, 8, 1, 7, 4, 3]],
        [[0, 0, 0, 0, 0, 0], [2, 8, 1, 0, 0, 0]],
    ]:
        ids = torch.tensor(tokens)
        mask = (ids != 0).int()
        with torch.no_grad():
            hidden = model(ids, attention_mask=mask).last_hidden_state
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1).clamp(min=1e-9)[:, None]
            normalized = torch.nn.functional.normalize(pooled, dim=-1)
        cases.append(dict(input_ids=tokens, attention_mask=mask.tolist(),
                          last_hidden_state=hidden.tolist(), sentence_embedding=normalized.tolist()))
    result = dict(reference='huggingface/transformers@6d3802a45f50ce6f2262029926265629010b2217',
                  config=model.config.to_dict(), cases=cases,
                  weights=base64.b64encode(save({k:v.contiguous() for k,v in model.state_dict().items()})).decode())
    path = Path(__file__).parent / 'fixtures/embeddinggemma2.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(path, path.stat().st_size)

    model = tiny_image_reference()
    cases = []
    for reorder in (False, True):
        inputs = tiny_image_inputs(reorder)
        with torch.no_grad():
            output = model(**{k:torch.tensor(v) for k,v in inputs.items()})
            mask = torch.tensor(inputs['attention_mask'])
            pool = (output.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        cases.append(dict(inputs={k:v.tolist() for k,v in inputs.items()},
                          last_hidden_state=output.last_hidden_state.tolist(),
                          sentence_embedding=torch.nn.functional.normalize(pool,dim=-1).tolist(),
                          image_hidden_states=output.image_hidden_states.tolist()))
    result = dict(reference=result['reference'],
                  config={**model.config.to_dict(), 'image_num_patches':16, 'max_seq_len':9},
                  cases=cases, weights=base64.b64encode(save({k:v.contiguous() for k,v in model.state_dict().items()})).decode())
    path = path.with_name('embeddinggemma2_image.json')
    path.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    print(path,path.stat().st_size)

    model=tiny_audio_reference(right=2)
    inputs=tiny_audio_inputs()
    with torch.no_grad():
        output=model(**{k:torch.tensor(v,dtype=torch.bool if k=='input_features_mask' else None) for k,v in inputs.items()})
        mask=torch.tensor(inputs['attention_mask'])
        pool=(output.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    result=dict(reference=result['reference'],
                config={**model.config.to_dict(),'audio_seq_len':17,'max_seq_len':8},
                weights=base64.b64encode(save({k:v.contiguous() for k,v in model.state_dict().items()})).decode(),
                cases=[dict(inputs={k:v.tolist() for k,v in inputs.items()},
                            last_hidden_state=output.last_hidden_state.tolist(),
                            sentence_embedding=torch.nn.functional.normalize(pool,dim=-1).tolist(),
                            audio_hidden_states=output.audio_hidden_states.tolist())])
    path=path.with_name('embeddinggemma2_audio.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(path,path.stat().st_size)


if __name__ == '__main__':
    main()
