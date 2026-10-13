"""Small real-reference fixture shared by Node native, Wasm and browser tests."""
import base64
import hashlib
import inspect
import json
from pathlib import Path
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5
from safetensors.torch import save
from external.test_qwen35_hf import reference, cached_reference


def main():
    fixture={'reference':{'transformers':transformers.__version__,
        'source_sha256':hashlib.sha256(Path(inspect.getfile(modeling_qwen3_5)).read_bytes()).hexdigest()},
        'models':[]}
    for dtype in (torch.float32,torch.bfloat16):
        row={'dtype':str(dtype).removeprefix('torch.'),'cases':[]}
        for causal in (True,False):
            config,model,ids,mask,positions,expected=reference(causal,dtype)
            payload=base64.b64encode(save(model.state_dict())).decode()
            if 'weights' in row:
                assert row['weights']==payload
            row['weights']=payload
            row['cases'].append({'causal':causal,'expected':expected.reshape(-1).tolist()})
            fixture['config']=config.to_dict()
            fixture['inputs']={k:v.numpy().reshape(-1).tolist() for k,v in zip(
                ['input_ids','attention_mask','position_ids'],[ids,mask,positions])}
        _,_,ids,expected,states=cached_reference(dtype)
        row['cached']={'expected':expected.reshape(-1).tolist(),
                       'states':{name:value.reshape(-1).tolist() for name,value in states.items()}}
        fixture['cached_inputs']=ids.numpy().reshape(-1).tolist()
        fixture['models'].append(row)
    path=Path(__file__).parent/'fixtures/qwen35.json'
    path.write_text(json.dumps(fixture,indent=2)+'\n')
    print(f'{path}: {path.stat().st_size} bytes')


if __name__=='__main__': main()
