"""EmbeddingGemma2 text/image oracle using the official Transformers implementation.

Reference revision: 6d3802a45f50ce6f2262029926265629010b2217.
No remote model code is executed; pretrained files are opt-in and live on Eve.
"""
import json
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel


def tiny_audio_reference(clipped=True, right=1, mixed=False):
    from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model, Gemma4AudioConfig
    text = tiny_reference().config
    audio = Gemma4AudioConfig(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                             subsampling_conv_channels=[8, 2], output_proj_dims=6,
                             attention_chunk_size=2, attention_context_left=3,
                             attention_context_right=right, conv_kernel_size=3,
                             use_clipped_linears=clipped, gradient_clipping=0.3)
    text.dtype = audio.dtype = torch.float32
    vision=tiny_image_reference().config.vision_config if mixed else None
    config = EmbeddingGemma2Config(text_config=text, audio_config=audio, vision_config=vision,
                                  audio_token_id=28, image_token_id=30, video_token_id=29,
                                  dtype=torch.float32)
    config._attn_implementation = 'eager'
    # Audio attention consumes a boolean mask even though it computes attention
    # itself. HF's eager mask adapter returns additive logits and inverts its
    # meaning here; keep audio on the default SDPA boolean-mask adapter.
    config.audio_config._attn_implementation = 'sdpa'
    torch.manual_seed(19)
    model = EmbeddingGemma2Model(config).float().eval()
    # Exercise checkpoint clipping, not just default infinite bounds.
    for name, value in model.named_buffers():
        if name.endswith('input_min'): value.fill_(-0.4)
        if name.endswith('input_max'): value.fill_(0.5)
        if name.endswith('output_min'): value.fill_(-0.03)
        if name.endswith('output_max'): value.fill_(0.04)
    return model


def tiny_audio_inputs():
    rng = np.random.default_rng(43)
    features = rng.normal(size=(2,17,8)).astype(np.float32)
    valid = np.array([[1]*17, [1]*9+[0]*8],np.int32)
    ids = np.array([[2,28,28,28,28,28,1,0], [0,2,28,28,28,1,0,0]],np.int32)
    return dict(input_ids=ids,attention_mask=(ids!=0).astype(np.int32),
                input_features=features,input_features_mask=valid)


@pytest.mark.parametrize('clipped,right,mixed', [(True,2,False),(False,0,False),(True,2,True)])
def test_audio_checkpoint_and_bundle(clipped, right, mixed):
    import polygrad as pg
    ref = tiny_audio_reference(clipped,right,mixed)
    inputs = tiny_audio_inputs()
    if mixed:
        image=tiny_image_inputs()
        inputs.update({k:image[k] for k in ('pixel_values','image_position_ids')})
        inputs['input_ids']=np.array([[2,30,30,28,28,28,28,28,30,30,1,0],
                                     [0,2,30,28,28,28,30,1,0,0,0,0]],np.int32)
        inputs['attention_mask']=(inputs['input_ids']!=0).astype(np.int32)
    states={}
    def capture(name):
        def hook(module,args,out): states[name]=(out[0] if isinstance(out,tuple) else out).detach().numpy().copy()
        return hook
    hooks=[ref.audio_tower.subsample_conv_projection.register_forward_hook(capture('audio_hidden_states.0'))]
    hooks += [layer.register_forward_hook(capture(f'audio_hidden_states.{i+1}')) for i,layer in enumerate(ref.audio_tower.layers)]
    def check_mask(module,args,kwargs):
        mask=kwargs['attention_mask']
        assert mask.dtype==torch.bool
        assert not mask[0,0,0,0,:2].any()  # padded past context
        assert mask[0,0,0,0,2]  # self is visible
    hooks.append(ref.audio_tower.layers[0].self_attn.register_forward_pre_hook(check_mask,with_kwargs=True))
    with torch.no_grad():
        expected=ref(**{k:torch.tensor(v,dtype=torch.bool if k=='input_features_mask' else None) for k,v in inputs.items()})
        mask=torch.tensor(inputs['attention_mask'])
        pool=(expected.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    for hook in hooks: hook.remove()
    config={**ref.config.to_dict(),'audio_seq_len':17,'image_num_patches':16,'output_hidden_states':True}
    with pg.create(device=os.environ.get('POLY_DEV','CPU')) as rt:
        model=pg.Model.from_hf(config_json=json.dumps(config),
                              weight_bytes_list=[save({k:v.contiguous() for k,v in ref.state_dict().items()})],
                              max_batch=2,max_seq_len=inputs['input_ids'].shape[1],runtime=rt)
        restored=None
        try:
            restored=rt.Model.load(model.save())
            for m in (model,restored):
                out=m.forward(**inputs)
                for name,value in states.items():
                    np.testing.assert_allclose(out[name],value,atol=2e-5,rtol=2e-4,err_msg=name)
                np.testing.assert_allclose(out['last_hidden_state'],expected.last_hidden_state.numpy(),atol=2e-5,rtol=2e-4)
                np.testing.assert_allclose(out['sentence_embedding'],torch.nn.functional.normalize(pool,dim=-1).numpy(),atol=2e-5,rtol=2e-4)
                selected=out['audio_hidden_states'][out['audio_attention_mask'].astype(bool)]
                np.testing.assert_allclose(selected,expected.audio_hidden_states.numpy(),atol=2e-5,rtol=2e-4)
                np.testing.assert_array_equal(out['audio_attention_mask'],inputs['input_features_mask'][:,::4])
        finally:
            if restored is not None: restored.dispose()
            model.dispose()


def tiny_reference():
    torch.manual_seed(73)
    config = EmbeddingGemma2TextConfig(
        vocab_size=31, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        global_head_dim=8, num_global_key_value_heads=1,
        hidden_size_per_layer_input=4, embedding_dim=6,
        layer_types=['sliding_attention', 'full_attention'], sliding_window=2,
    )
    config._attn_implementation = 'eager'
    model = EmbeddingGemma2TextModel(config).float().eval()
    # Non-unit scalar verifies the checkpoint buffer is not silently discarded.
    model.layers[0].layer_scalar.fill_(0.75)
    return model


def tiny_image_reference(*, standardize=False, head_dim=4):
    from transformers import EmbeddingGemma2Config, Gemma4VisionConfig
    text = tiny_reference().config
    vision = Gemma4VisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim,
                               patch_size=2, pooling_kernel_size=2, position_embedding_size=32,
                               default_output_length=4, standardize=standardize, use_clipped_linears=False)
    config = EmbeddingGemma2Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                                   audio_config=None, image_token_id=30, video_token_id=29,
                                   audio_token_id=28, vision_soft_tokens_per_image=4)
    torch.manual_seed(91)
    config._attn_implementation = 'eager'
    model=fp32_image_reference(config)
    if standardize:
        model.vision_tower.std_bias.copy_(torch.linspace(-0.2,0.1,vision.hidden_size))
        model.vision_tower.std_scale.copy_(torch.linspace(0.8,1.3,vision.hidden_size))
    return model


def fp32_image_reference(config):
    from transformers import EmbeddingGemma2Model
    # Composite construction uses AutoModel.from_config for its children.
    # Casting afterward cannot undo BF16 rounding of nonpersistent constants.
    config.dtype = torch.float32
    config.text_config.dtype = torch.float32
    if config.vision_config is not None:
        config.vision_config.dtype = torch.float32
    return EmbeddingGemma2Model(config).float().eval()


def test_reference_constructs_constants_in_fp32():
    config=tiny_image_reference().config
    config.dtype=config.text_config.dtype=config.vision_config.dtype=torch.bfloat16
    model=fp32_image_reference(config)
    expected=torch.tensor(config.text_config.hidden_size**0.5,dtype=torch.float32)
    torch.testing.assert_close(model.language_model.embed_tokens.embed_scale,expected,atol=0,rtol=0)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def tiny_image_inputs(reordered=False):
    pixels = np.random.default_rng(25).random((2, 16, 12)).astype(np.float32)
    positions = np.full((2, 16, 2), -1, np.int32)
    positions[0] = [[x, y] for y in range(4) for x in range(4)]
    positions[1, :8] = [[x, y] for y in range(4) for x in range(2)]
    if reordered:
        order = np.random.default_rng(37).permutation(16)
        pixels, positions = pixels[:,order], positions[:,order]
    tokens = np.array([[2,30,30,7,30,30,1,0,0], [0,0,2,4,30,30,1,0,0]], np.int32)
    return dict(input_ids=tokens, attention_mask=(tokens != 0).astype(np.int32),
                pixel_values=pixels, image_position_ids=positions)


@pytest.mark.parametrize('device', [os.environ.get('POLY_DEV', os.environ.get('DEV', 'CPU'))])
@pytest.mark.parametrize('reordered', [False, True])
@pytest.mark.parametrize('standardize', [False, True])
def test_image_checkpoint_and_bundle(device, reordered, standardize):
    import polygrad as pg
    from check_embeddinggemma2_pretrained import assert_image_stages
    ref = tiny_image_reference(standardize=standardize, head_dim=8 if standardize else 4)
    inputs = tiny_image_inputs(reordered)
    states={}
    def capture(name):
        def hook(module,args,out): states[name]=out.detach().numpy().copy()
        return hook
    hooks=[ref.vision_tower.patch_embedder.register_forward_hook(capture('vision_hidden_states.0'))]
    hooks += [layer.register_forward_hook(capture(f'vision_hidden_states.{i+1}'))
              for i,layer in enumerate(ref.vision_tower.encoder.layers)]
    with torch.no_grad():
        expected = ref(**{k:torch.tensor(v) for k,v in inputs.items()})
        mask = torch.tensor(inputs['attention_mask'])
        pooled = (expected.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        normalized = torch.nn.functional.normalize(pooled,dim=-1).numpy()
    for hook in hooks: hook.remove()
    config = {**ref.config.to_dict(), 'image_num_patches':16, 'output_hidden_states':True}
    weights = save({k:v.contiguous() for k,v in ref.state_dict().items()})
    with pg.create(device=device) as rt:
        model = pg.Model.from_hf(config_json=json.dumps(config), weight_bytes_list=[weights],
                                max_batch=2, max_seq_len=9, runtime=rt)
        restored = None
        try:
            restored = rt.Model.load(model.save())
            for m in (model,restored):
                result = m.forward(**inputs)
                np.testing.assert_allclose(result['last_hidden_state'],expected.last_hidden_state.numpy(),atol=2e-5,rtol=2e-4)
                np.testing.assert_allclose(result['sentence_embedding'],normalized,atol=2e-5,rtol=2e-4)
                selected = result['image_hidden_states'][result['image_attention_mask'].astype(bool)]
                np.testing.assert_allclose(selected,expected.image_hidden_states.numpy(),atol=2e-5,rtol=2e-4)
                for name,value in states.items():
                    np.testing.assert_allclose(result[name],value,atol=2e-5,rtol=2e-4)
            checked=assert_image_stages(ref,inputs,result)
            assert len(checked)==8  # patches, 2 vision layers, insertion/projection, 2 text layers, final projection
            if device=='CPU' and not reordered and not standardize:
                # Replay must check before injecting a stage into the next one;
                # otherwise it could hide bad intermediates behind good outputs.
                for name in ('vision_hidden_states.1','image_hidden_states','hidden_states.0','last_hidden_state'):
                    bad={**result,name:result[name]+np.float32(0.125)}
                    with pytest.raises(AssertionError,match=name):
                        assert_image_stages(ref,inputs,bad)
        finally:
            if restored is not None: restored.dispose()
            model.dispose()


@pytest.mark.parametrize('device', [os.environ.get('POLY_DEV', os.environ.get('DEV', 'CPU'))])
@pytest.mark.parametrize('padding', ['right', 'left'])
def test_text_checkpoint_and_bundle(device, padding):
    import polygrad as pg
    ref = tiny_reference()
    tokens = np.array([[2, 7, 4, 3, 1, 0], [2, 8, 1, 0, 0, 0]], np.int32)
    if padding == 'left':
        tokens = np.array([[0, 2, 7, 4, 3, 1], [0, 0, 0, 2, 8, 1]], np.int32)
    mask = (tokens != 0).astype(np.int32)
    with torch.no_grad():
        hidden = ref(torch.tensor(tokens), attention_mask=torch.tensor(mask)).last_hidden_state
        pooled = (hidden * torch.tensor(mask)[..., None]).sum(1) / torch.tensor(mask).sum(1)[:, None]
        expected = torch.nn.functional.normalize(pooled, dim=-1).numpy()
    weights = save({k: v.contiguous() for k, v in ref.state_dict().items()})
    with pg.create(device=device) as rt:
        model = pg.Model.from_hf(config_json=json.dumps(ref.config.to_dict()),
                                weight_bytes_list=[weights], max_batch=2, max_seq_len=6, runtime=rt)
        try:
            result = model.forward(input_ids=tokens, attention_mask=mask)
            np.testing.assert_allclose(result['last_hidden_state'], hidden.numpy(), atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(result['sentence_embedding'], expected, atol=2e-5, rtol=2e-4)
            restored = rt.Model.load(model.save())
            try:
                again = restored.forward(input_ids=tokens, attention_mask=mask)
                np.testing.assert_array_equal(again['sentence_embedding'], result['sentence_embedding'])
            finally:
                restored.dispose()
        finally:
            model.dispose()


def test_composite_text_selection_and_weight_validation():
    import polygrad as pg
    ref = tiny_reference()
    cfg = dict(model_type='embedding_gemma2', text_config=ref.config.to_dict(),
               vision_config=None, audio_config=None)
    state = {'language_model.'+k: v.contiguous() for k,v in ref.state_dict().items()}
    # Disabled towers may be present, but every text weight remains mandatory.
    state['vision_tower.unused.weight'] = torch.ones(2)
    tokens = np.array([[2,7,1]], np.int32)
    mask = np.ones_like(tokens)
    with torch.no_grad():
        expected = ref(torch.tensor(tokens), attention_mask=torch.tensor(mask)).last_hidden_state.numpy()
    with pg.create(device='CPU') as rt:
        model = pg.Model.from_hf(config_json=json.dumps(cfg), weight_bytes_list=[save(state)],
                                max_seq_len=3, runtime=rt)
        try:
            actual = model.forward(input_ids=tokens, attention_mask=mask)
            np.testing.assert_allclose(actual['last_hidden_state'], expected, atol=2e-5, rtol=2e-4)
        finally:
            model.dispose()
        for mutation in ('missing', 'shape', 'unknown'):
            bad = dict(state)
            if mutation == 'missing':
                del bad['language_model.norm.weight']
            elif mutation == 'shape':
                bad['language_model.norm.weight'] = torch.ones(7)
            else:
                bad['language_model.unknown.weight'] = torch.ones(1)
            with pytest.raises(RuntimeError, match='weight'):
                pg.Model.from_hf(config_json=json.dumps(cfg), weight_bytes_list=[save(bad)],
                                 max_seq_len=3, runtime=rt)


def test_text_layer_outputs_and_bundle():
    import polygrad as pg
    ref=tiny_reference()
    expected={}
    def capture(name):
        def hook(module,args,out): expected[name]=out.detach().numpy().copy()
        return hook
    hooks=[ref.embed_tokens.register_forward_hook(capture('hidden_states.0'))]
    hooks += [layer.register_forward_hook(capture(f'hidden_states.{i+1}')) for i,layer in enumerate(ref.layers)]
    ids=np.array([[2,7,1]],np.int32); mask=np.ones_like(ids)
    with torch.no_grad(): ref(torch.tensor(ids),attention_mask=torch.tensor(mask))
    for hook in hooks: hook.remove()
    with pg.create(device='CPU') as rt:
        model=pg.Model.from_hf(config_json=json.dumps({**ref.config.to_dict(),'output_hidden_states':True}),
                              weight_bytes_list=[save({k:v.contiguous() for k,v in ref.state_dict().items()})],
                              max_seq_len=3,runtime=rt)
        restored=None
        try:
            restored=rt.Model.load(model.save())
            for m in (model,restored):
                result=m.forward(input_ids=ids,attention_mask=mask)
                for name,value in expected.items():
                    np.testing.assert_allclose(result[name],value,atol=2e-5,rtol=2e-4)
        finally:
            if restored is not None: restored.dispose()
            model.dispose()
