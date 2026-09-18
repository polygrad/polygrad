"""Deterministic CLIP/ViT/DINO inference oracle (Transformers, eager float32)."""
import base64
import json
import torch
import transformers
from safetensors.torch import save
from vision_fixture import compact_case
from transformers import (CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig,
                          ViTConfig, ViTModel, Dinov2Config, Dinov2Model,
                          DINOv3ViTConfig, DINOv3ViTModel)

torch.set_num_threads(1)
common = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
              image_size=8, patch_size=4, num_channels=3)
text = CLIPTextConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                     num_attention_heads=2, vocab_size=20, max_position_embeddings=5,
                     eos_token_id=19, bos_token_id=18, pad_token_id=0)
vision = CLIPVisionConfig(**common, intermediate_size=24)
specs = [('CLIP', CLIPModel, CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=8)),
         ('ViT', ViTModel, ViTConfig(**common, intermediate_size=24)),
         ('DINOv2', Dinov2Model, Dinov2Config(**common, mlp_ratio=2)),
         ('DINOv2', Dinov2Model, Dinov2Config(**common, mlp_ratio=2, use_swiglu_ffn=True)),
         ('DINOv3', DINOv3ViTModel, DINOv3ViTConfig(**common, intermediate_size=24, num_register_tokens=2)),
         ('DINOv3', DINOv3ViTModel, DINOv3ViTConfig(**common, intermediate_size=24, num_register_tokens=0,
                                                use_gated_mlp=True, hidden_act='silu'))]
cases = []
for name, cls, config in specs:
    config._attn_implementation = 'eager'
    torch.manual_seed(42)
    model = cls(config).float().eval()
    with torch.no_grad():
        for key, p in model.named_parameters():
            values = ((torch.arange(p.numel()) * 7 + sum(key.encode())) % 29 - 14).float() * .013
            if ('norm' in key and key.endswith('weight')) or key.endswith('lambda1'): values += 1
            if key == 'logit_scale': values.fill_(1.3)
            p.copy_(values.reshape(p.shape))
        pixels = ((torch.arange(2*3*8*8) * 11 % 101).float() / 50 - 1).reshape(2,3,8,8)
        inputs = dict(pixel_values=pixels)
        if name == 'CLIP': inputs['input_ids'] = torch.tensor([[18,3,7,19,0], [18,5,19,0,0]])
        out = model(**inputs)
    keys = ('image_embeds', 'text_embeds', 'logits_per_image', 'logits_per_text') if name == 'CLIP' else ('last_hidden_state', 'pooler_output')
    cases.append(dict(name=name, config=config.to_dict(),
                      inputs={k:v.tolist() for k,v in inputs.items()},
                      outputs={k:getattr(out,k).tolist() for k in keys},
                      weights=base64.b64encode(save({k:v.contiguous() for k,v in model.state_dict().items()})).decode()))
print(json.dumps(dict(reference=f'transformers {transformers.__version__}, eager float32 eval',
                     payload_formula='vision-f32@1', cases=[compact_case(case) for case in cases]), indent=2))
