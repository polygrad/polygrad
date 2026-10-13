"""Qwen3.5 text prefill against the actual Transformers model (no weights download)."""
import json
import numpy as np
import pytest
import torch
from safetensors.torch import save
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel


def reference(causal=True, dtype=torch.float32, layers=None, left_padding=False):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=3,
        layer_types=layers or ['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':0.5,
                         'mrope_section':[1,1,0]}, is_causal=causal)
    config._attn_implementation = 'eager'
    model = Qwen3_5TextModel(config).to(dtype).eval()
    ids = torch.tensor([[1,2,3,4,5],[6,7,8,0,0]])
    mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    positions = torch.arange(5).expand(2,-1)
    if left_padding:
        ids[1]=torch.tensor([0,0,6,7,8])
        mask[1]=torch.tensor([0,0,1,1,1])
        positions=(mask.cumsum(-1)-1).clamp(min=0)
    with torch.no_grad():
        expected = model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state.float().numpy()
    return config,model,ids,mask,positions,expected


def cached_reference(dtype=torch.float32):
    config,ref,*_=reference(dtype=dtype)
    ids=torch.tensor([[1,2,3,4,5,6]])
    with torch.no_grad():
        expected=ref(input_ids=ids,use_cache=False).last_hidden_state.float().numpy()
        first=ref(input_ids=ids[:,:2],use_cache=True)
        cache=first.past_key_values
        outputs=[first.last_hidden_state]
        for pos in range(2,6):
            result=ref(input_ids=ids[:,pos:pos+1],past_key_values=cache,use_cache=True,
                       cache_position=torch.tensor([pos]))
            outputs.append(result.last_hidden_state)
        np.testing.assert_allclose(torch.cat(outputs,dim=1).float().numpy(),expected,
            atol=.04 if dtype==torch.bfloat16 else 7e-5,rtol=.02 if dtype==torch.bfloat16 else 7e-4)
    states={'layers.0.cache.conv':cache.conv_states[0].float().numpy(),
            'layers.0.cache.delta':cache.recurrent_states[0].float().numpy(),
            'layers.1.cache.kv':torch.stack([cache.key_cache[1],cache.value_cache[1]]).float().numpy()}
    return config,ref,ids,expected,states


def composite_reference(generation=False, tied=True, dtype=torch.float32):
    from transformers import Qwen3_5Config, Qwen3_5Model, Qwen3_5ForConditionalGeneration
    text, backbone, ids, mask, positions, _ = reference(dtype=dtype)
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=dict(
        depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
        patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
        out_hidden_size=16, num_position_embeddings=16), tie_word_embeddings=tied)
    config.text_config.tie_word_embeddings = tied
    config._attn_implementation = 'eager'
    cls = Qwen3_5ForConditionalGeneration if generation else Qwen3_5Model
    config.architectures = [cls.__name__]
    model = cls(config).to(dtype).eval()
    child = model.model if generation else model
    child.language_model.load_state_dict(backbone.state_dict())
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
        expected = (out.logits if generation else out.last_hidden_state).float().numpy()
    return config, model, ids, mask, positions, expected


@pytest.mark.parametrize('generation,tied', [(False,True),(True,True),(True,False)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_qwen35_composite_text_import_and_head(generation,tied,dtype):
    import polygrad as pg
    config, ref, ids, mask, positions, expected = composite_reference(generation,tied,dtype)
    cfg = config.to_dict(); cfg['modalities'] = ['text']
    # Safetensors omits the duplicate tied head in published checkpoints.
    weights = {k:v.contiguous() for k,v in ref.state_dict().items()
               if not (generation and tied and k == 'lm_head.weight')}
    inputs = {k:v.numpy().astype('int32') for k,v in zip(
        ['input_ids','attention_mask','position_ids'],[ids,mask,positions])}
    key = 'logits' if generation else 'last_hidden_state'
    with pg.create(device='CPU') as rt:
        model = rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)],
                                 max_batch=2,max_seq_len=5)
        try:
            got = model.forward(**inputs)[key]
            np.testing.assert_allclose(got,expected,atol=.04 if dtype==torch.bfloat16 else 5e-5,
                                       rtol=.02 if dtype==torch.bfloat16 else 5e-4)
            assert not any('visual.' in b['name'] for b in model.bindings())
            restored = rt.Model.load(model.save())
            try: np.testing.assert_array_equal(restored.forward(**inputs)[key],got)
            finally: restored.dispose()
        finally: model.dispose()


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('tied',[True,False])
def test_qwen35_composite_cached_logits_and_sampling(dtype,tied):
    import polygrad as pg
    config, ref, ids, _, _, _ = composite_reference(True,tied,dtype)
    ids = ids[:1]
    with torch.no_grad(): expected = ref(input_ids=ids,use_cache=False).logits.float().numpy()
    cfg = config.to_dict(); cfg.update(modalities=['text'],cache_capacity=8)
    weights = {k:v.contiguous().clone() for k,v in ref.state_dict().items()}
    with pg.create(device='CPU') as rt:
        model = rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)],max_seq_len=2)
        try:
            outputs=[]
            for pos in (0,2,4):
                entry = 'decode' if pos == 4 else 'prefill'
                n = 1 if pos == 4 else 2
                out = model.call(entry,{f'tokens_{entry}':ids[:,pos:pos+n].numpy().astype('int32')},
                                 controls={'start_pos':pos})[f'logits_{entry}']
                outputs.append(out.reshape(1,n,-1))
            np.testing.assert_allclose(np.concatenate(outputs,axis=1),expected,
                atol=7e-5 if dtype==torch.float32 else .04,rtol=7e-4 if dtype==torch.float32 else .02)
            states={b['name']:model.read_buffer(b['name']).copy() for b in model.bindings()
                    if '.cache.' in b['name']}
            restored=rt.Model.load(model.save())
            try:
                model.reset_transient()
                for candidate in (model,restored):
                    codes=[]
                    for pos in range(ids.shape[1]):
                        out=candidate.call('decode_sample',{
                            'tokens_decode':ids[:,pos:pos+1].numpy().astype('int32'),
                            'sampling.temperature':np.array([0.],np.float32)},controls={'start_pos':pos})
                        codes.append(int(out['sampling.decode_token'].reshape(-1)[0]))
                    np.testing.assert_array_equal(codes,expected.argmax(-1)[0])
                    for name,value in states.items():
                        actual=candidate.read_buffer(name)
                        if value.dtype==np.uint16:
                            if dtype==torch.bfloat16:
                                actual=(actual.astype(np.uint32)<<16).view(np.float32)
                                value=(value.astype(np.uint32)<<16).view(np.float32)
                            else:
                                actual=actual.view(np.float16).astype(np.float32)
                                value=value.view(np.float16).astype(np.float32)
                        np.testing.assert_allclose(actual,value,atol=3e-5 if dtype==torch.float32 else .04,
                                                   rtol=5e-4 if dtype==torch.float32 else .02)
            finally: restored.dispose()
        finally: model.dispose()


def test_qwen35_composite_selection_and_weight_errors():
    import polygrad as pg
    config,ref,*_ = composite_reference(True)
    cfg=config.to_dict()
    weights={k:v.clone() for k,v in ref.state_dict().items()}
    with pg.create(device='CPU') as rt:
        missing_vision=dict(cfg);missing_vision.pop('vision_config')
        with pytest.raises(RuntimeError,match='modalities'):
            rt.Model.from_hf(config_json=json.dumps(missing_vision),weight_bytes_list=[save(weights)])
        for selection in (None,[],['image'],['text','text'],['text','image']):
            bad=dict(cfg)
            if selection is not None: bad['modalities']=selection
            with pytest.raises(RuntimeError,match='modalities'):
                rt.Model.from_hf(config_json=json.dumps(bad),weight_bytes_list=[save(weights)])
        cfg['modalities']=['text']
        weights['lm_head.weight'][0,0]+=1
        with pytest.raises(RuntimeError,match='must match tied embeddings'):
            rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)])
        del weights['lm_head.weight']
        for key in ('model.language_model.unrelated.weight','visual.unrelated.weight'):
            bad=dict(weights);bad[key]=torch.ones(1)
            with pytest.raises(RuntimeError,match='unexpected weight'):
                rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(bad)])
        del weights['model.language_model.layers.0.linear_attn.dt_bias']
        with pytest.raises(RuntimeError,match='weight'):
            rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)])


@pytest.mark.parametrize('causal',[True,False])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_qwen35_text_prefill(causal,dtype):
    import polygrad as pg
    config,ref,ids,mask,positions,expected = reference(causal,dtype)
    payload = save({k:v.contiguous() for k,v in ref.state_dict().items()})
    with pg.create(device='CPU') as rt:
        model = rt.Model.from_hf(config_json=config.to_json_string(),weight_bytes_list=[payload],
                                 max_batch=2,max_seq_len=5)
        assert all(b['dtype']==str(dtype).removeprefix('torch.') for b in model.bindings()
                   if b['role']==pg.model.ROLE_PARAM)
        inputs = {k:v.numpy().astype('int32') for k,v in zip(
            ['input_ids','attention_mask','position_ids'],[ids,mask,positions])}
        got = model.forward(**inputs)['last_hidden_state']
        np.testing.assert_allclose(got,expected,atol=0.04 if dtype==torch.bfloat16 else 5e-5,rtol=0.02 if dtype==torch.bfloat16 else 5e-4)
        restored = rt.Model.load(model.save())
        np.testing.assert_array_equal(restored.forward(**inputs)['last_hidden_state'],got)
        restored.dispose(); model.dispose()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_qwen35_left_padding(dtype):
    import polygrad as pg
    config,ref,ids,mask,positions,expected=reference(dtype=dtype,left_padding=True)
    with pg.create(device='CPU') as rt:
        model=rt.Model.from_hf(config_json=config.to_json_string(),weight_bytes_list=[save(ref.state_dict())],
                              max_batch=2,max_seq_len=5)
        try:
            inputs={k:v.numpy().astype('int32') for k,v in zip(
                ['input_ids','attention_mask','position_ids'],[ids,mask,positions])}
            result=model.forward(**inputs)['last_hidden_state']
            np.testing.assert_allclose(result,expected,atol=5e-5 if dtype==torch.float32 else .04,
                                       rtol=5e-4 if dtype==torch.float32 else .02)
        finally:
            model.dispose()


def test_qwen35_linear_layers_ignore_full_attention_causal_flag():
    import polygrad as pg
    results=[]
    for causal in (True,False):
        config,ref,ids,mask,positions,expected = reference(causal,layers=['linear_attention']*2)
        with pg.create(device='CPU') as rt:
            model=rt.Model.from_hf(config_json=config.to_json_string(),
                weight_bytes_list=[save(ref.state_dict())],max_batch=2,max_seq_len=5)
            inputs={k:v.numpy().astype('int32') for k,v in zip(
                ['input_ids','attention_mask','position_ids'],[ids,mask,positions])}
            result=model.forward(**inputs)['last_hidden_state']
            np.testing.assert_allclose(result,expected,atol=5e-5,rtol=5e-4)
            results.append(result)
            model.dispose()
    np.testing.assert_array_equal(*results)


def test_qwen35_strict_weights_and_unsupported_configs():
    import polygrad as pg
    config,ref,*_ = reference()
    weights=ref.state_dict()
    bad=dict(weights); bad.pop('layers.0.linear_attn.dt_bias')
    extra=dict(weights); extra['unrelated.weight']=torch.ones(1)
    wrong=dict(weights); wrong['embed_tokens.weight']=wrong['embed_tokens.weight'].reshape(16,32)
    with pg.create(device='CPU') as rt:
        for state in (bad,extra,wrong):
            with pytest.raises(RuntimeError,match='weight'):
                rt.Model.from_hf(config_json=config.to_json_string(),weight_bytes_list=[save(state)])
        for key,value in [('layer_types',['full_attention']),('attention_bias',True),
                          ('vision_config',{}),('is_causal','false')]:
            cfg=config.to_dict(); cfg[key]=value
            with pytest.raises(RuntimeError):
                rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)])
        for extra in ({'cache_capacity':1}, {'cache_capacity':8,'is_causal':False},
                      {'cache_capacity':8,'layer_types':['linear_attention']*2}):
            cfg=config.to_dict(); cfg.update(extra)
            with pytest.raises(RuntimeError,match='cache_capacity'):
                rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(weights)],max_seq_len=2)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_qwen35_cached_prefill_decode_and_reset(dtype):
    import polygrad as pg
    config,ref,ids,expected,reference_states=cached_reference(dtype)
    cfg=config.to_dict(); cfg['cache_capacity']=8
    with pg.create(device='CPU') as rt:
        model=rt.Model.from_hf(config_json=json.dumps(cfg),weight_bytes_list=[save(ref.state_dict())],
                              max_batch=1,max_seq_len=2)
        try:
            results=[]
            for pos in (0,2,4,5):
                entry='prefill' if pos<4 else 'decode'
                length=2 if pos<4 else 1
                result=model.call(entry,{f'tokens_{entry}':ids[:,pos:pos+length].numpy().astype('int32')},
                                  controls={'start_pos':pos})
                results.append(result[f'hidden_{entry}'])
            np.testing.assert_allclose(np.concatenate(results,axis=1),expected,
                atol=.04 if dtype==torch.bfloat16 else 7e-5,rtol=.02 if dtype==torch.bfloat16 else 7e-4)
            cache_bindings={b['name']:b for b in model.bindings() if '.cache.' in b['name']}
            states={name:model.read_buffer(name).copy() for name in cache_bindings}
            assert len(states)==3
            for name,value in states.items():
                value=value.reshape(cache_bindings[name]['shape'])
                if value.dtype==np.uint16:
                    value=(value.astype(np.uint32)<<16).view(np.float32)
                if name.endswith('.kv'): value=value[:,:,:,:6,:]
                np.testing.assert_allclose(value,reference_states[name],
                    atol=.01 if dtype==torch.bfloat16 else 3e-5,rtol=.02 if dtype==torch.bfloat16 else 5e-4)
            with pytest.raises((RuntimeError,ValueError)):
                model.call('prefill',{'tokens_prefill':np.array([[1,2]],dtype='int32')},controls={'start_pos':7})
            for name,value in states.items(): np.testing.assert_array_equal(model.read_buffer(name),value)
            restored=rt.Model.load(model.save())
            try:
                for name in states: assert not np.any(restored.read_buffer(name))
                model.reset_transient()
                for name in states: assert not np.any(model.read_buffer(name))
                for candidate in (model,restored):
                    got=candidate.call('prefill',{'tokens_prefill':ids[:,:2].numpy().astype('int32')},controls={'start_pos':0})
                    np.testing.assert_array_equal(got['hidden_prefill'],results[0])
            finally: restored.dispose()
        finally: model.dispose()
