"""Reference/checkpoint probe. Keep weights, converted subsets and outputs on Eve.

Use separate processes for reference and Polygrad to bound peak memory. Example:
  python test/external/check_embeddinggemma2_pretrained.py fetch --checkpoint /path/on/eve/model
  python test/external/check_embeddinggemma2_pretrained.py reference --checkpoint /path/on/eve/model
  PYTHONPATH=py POLY_LIB=build/libpolygrad.so python test/external/check_embeddinggemma2_pretrained.py polygrad --checkpoint /path/on/eve/model --device CUDA
Reference requires Transformers revision 6d3802a45f50ce6f2262029926265629010b2217.
Run text and two-photo checks without downloading weights:
  POLY_DEV=CUDA make test-embeddinggemma2-pretrained HF_PYTHON=/path/to/python EMBEDDINGGEMMA2_CHECKPOINT=/path/to/checkpoint
Device selection: --device, then POLY_DEV, then DEV, otherwise CPU.
Small live-reference fixtures: make test-embeddinggemma2 HF_PYTHON=/path/to/python
Full-checkpoint browser probes are tracked separately from this native runner;
this target does not certify browser performance or memory use.
Use --audio with reference/polygrad to check a deterministic waveform through
the official audio processor and the pretrained audio/text path (not a speech
quality benchmark). The converted audio subset stays beside the checkpoint.
Image verification checks final embeddings end-to-end, then each intermediate
stage against HF on identical inputs. --strict-raw additionally applies the
text-only accumulated-state bound, known to fail even between HF CPU and CUDA.
Image timings include intermediate-output readbacks and are not inference benchmarks.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def fp32_vector_errors(actual, expected):
    actual, expected = np.asarray(actual,np.float64), np.asarray(expected,np.float64)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    delta=actual-expected
    length=np.maximum(np.linalg.norm(expected,axis=-1),1e-30)
    relative=np.linalg.norm(delta,axis=-1)/length
    rms=length/np.sqrt(expected.shape[-1])
    return float(relative.max()), float((np.abs(delta)/rms[...,None]).max())


def assert_fp32_vectors(actual, expected):
    """Normwise FP32 check for unnormalized, cancellation-sensitive activations.

    The recorded layer trace shows <=3.3e-6 relative L2 error before projection;
    the projection amplifies absolute error. A fixed 1e-4 absolute bound near
    zero also rejects PyTorch CPU versus CUDA. Bound every vector's error and
    every channel relative to its vector RMS instead of allowing 1e-3 relative
    error on large channels. Small frozen fixtures retain their elementwise
    tolerances; normalized public embeddings have a separate tighter check.
    """
    relative, channel_rms = fp32_vector_errors(actual, expected)
    assert relative < 1e-5, relative
    assert channel_rms < 5e-5, channel_rms


def assert_image_stages(reference, inputs, actual):
    """Check local errors without conflating them with propagation through 40 layers.

    Each hook bounds every vector's relative L2 error by 1e-5, then feeds the
    Polygrad value into the next reference stage. The
    separate end-to-end embedding assertion remains mandatory. Small fixtures
    also retain their end-to-end hidden-state assertions.
    """
    import torch
    hooks, checked = [], {}

    def check(name, value):
        expected = value.detach().cpu().numpy()
        candidate = actual[name]
        if name == 'image_hidden_states':
            candidate = candidate[actual['image_attention_mask'].astype(bool)]
        try:
            relative, channel_rms = fp32_vector_errors(candidate, expected)
            assert relative < 1e-5, relative
        except AssertionError as exc:
            raise AssertionError(f'{name}: {exc}') from exc
        # Report channel error too, but do not impose the text projection's
        # RMS-based bound on differently sized internal vectors. The normwise
        # bound covers every channel without privileging cancellation near zero.
        checked[name] = dict(relative_l2=relative,channel_rms=channel_rms)
        # Own the replay Tensor: HF decoder layers can mutate their output.
        return torch.tensor(candidate, device=value.device, dtype=value.dtype)

    def output(name):
        return lambda module, args, value: check(name, value)

    def text_input(module, args, kwargs):
        check('hidden_states.0', kwargs['inputs_embeds'])

    def image_mask(module, args, value):
        np.testing.assert_array_equal(value[1].cpu().numpy(), actual['image_attention_mask'])

    expected_names = {'vision_hidden_states.0','image_hidden_states','hidden_states.0','last_hidden_state'}
    try:
        hooks.append(reference.vision_tower.patch_embedder.register_forward_hook(output('vision_hidden_states.0')))
        hooks.append(reference.vision_tower.pooler.register_forward_hook(image_mask))
        for i, layer in enumerate(reference.vision_tower.encoder.layers):
            name=f'vision_hidden_states.{i+1}'
            expected_names.add(name)
            hooks.append(layer.register_forward_hook(output(name)))
        hooks.append(reference.embed_vision.register_forward_hook(output('image_hidden_states')))
        hooks.append(reference.language_model.register_forward_pre_hook(text_input,with_kwargs=True))
        for i, layer in enumerate(reference.language_model.layers):
            name=f'hidden_states.{i+1}'
            expected_names.add(name)
            hooks.append(layer.register_forward_hook(output(name)))
        with torch.no_grad():
            result=reference(**{k:torch.tensor(v) for k,v in inputs.items()})
        check('last_hidden_state', result.last_hidden_state)
        assert set(checked) == expected_names
        return checked
    finally:
        for hook in hooks:
            hook.remove()


def image_reference(path, device='CPU', trace=False, photo_name='flower.jpg'):
    import torch
    from PIL import Image
    from sklearn.datasets import load_sample_image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoProcessor, EmbeddingGemma2Config, EmbeddingGemma2Model
    torch.set_num_threads(4)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tag = 'image' if photo_name == 'flower.jpg' else 'image-china'
    photo = Image.fromarray(load_sample_image(photo_name))
    processed = processor(text=['title: none | text: '+processor.image_token], images=[photo],
                          images_kwargs={'max_soft_tokens':70}, return_tensors='pt')
    inputs = {k:processed[k] for k in ('input_ids','attention_mask','pixel_values','image_position_ids')}
    config = json.loads((path/'config.json').read_text())
    config['audio_config'] = None
    config['image_num_patches'] = inputs['pixel_values'].shape[1]
    config['max_seq_len'] = inputs['input_ids'].shape[1]
    (path/f'{tag}-config.json').write_text(json.dumps(config),encoding='utf-8')
    cfg = EmbeddingGemma2Config(**config)
    cfg._attn_implementation = 'eager'
    # AutoModel.from_config honors child dtype metadata. A later .float()
    # cannot recover constants already rounded during BF16 construction.
    cfg.dtype = cfg.text_config.dtype = cfg.vision_config.dtype = torch.float32
    model = EmbeddingGemma2Model(cfg).float().eval()
    torch.testing.assert_close(model.language_model.embed_tokens.embed_scale,
                               torch.tensor(cfg.text_config.hidden_size**0.5),rtol=0,atol=0)
    with safe_open(path/'model.safetensors',framework='pt') as f:
        weights = {k:f.get_tensor(k) for k in f.keys() if not k.startswith(('audio_tower.','embed_audio.'))}
    save_file(weights,path/'image.safetensors')
    model.load_state_dict(weights,strict=True)
    del weights
    model=model.to(device.lower())
    inputs={k:v.to(device.lower()) for k,v in inputs.items()}
    traced={}
    hooks=[]
    if trace:
        def capture(name):
            def hook(module,args,out): traced[name]=out.detach().cpu().numpy().copy()
            return hook
        hooks.append(model.vision_tower.patch_embedder.register_forward_hook(capture('vision_hidden_states.0')))
        hooks += [layer.register_forward_hook(capture(f'vision_hidden_states.{i+1}'))
                  for i,layer in enumerate(model.vision_tower.encoder.layers)]
        hooks += [layer.register_forward_hook(capture(f'hidden_states.{i+1}'))
                  for i,layer in enumerate(model.language_model.layers)]
    with torch.no_grad():
        start=time.perf_counter()
        outputs=model(**inputs)
        mask=inputs['attention_mask']
        pooled=(outputs.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        embedding=torch.nn.functional.normalize(pooled,dim=-1)
    suffix='' if device.upper()=='CPU' else '-'+device.lower()
    np.savez(path/f'reference-{tag}{suffix}.npz', **{k:v.cpu().numpy() for k,v in inputs.items()},
             last_hidden_state=outputs.last_hidden_state.cpu().numpy(),sentence_embedding=embedding.cpu().numpy(),
             image_hidden_states=outputs.image_hidden_states.cpu().numpy(),**traced)
    for hook in hooks: hook.remove()
    print(json.dumps(dict(reference_seconds=time.perf_counter()-start,
                          inputs={k:list(v.shape) for k,v in inputs.items()},
                          image='sklearn.datasets.load_sample_image '+photo_name)),flush=True)


def audio_probe(path, mode, device):
    if mode == 'reference':
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers import AutoProcessor, EmbeddingGemma2Config, EmbeddingGemma2Model
        torch.set_num_threads(4)
        processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        # A short chirp exercises the real log-mel frontend without downloading
        # a second artifact or claiming speech retrieval quality from synthetic data.
        rate=16000
        t=np.arange(rate//2,dtype=np.float64)/rate
        waveform=(0.2*np.sin(2*np.pi*(180*t+400*t*t))).astype(np.float32)
        processed=processor(text=['title: none | text: '+processor.audio_token],audio=[waveform],
                            audio_kwargs={'sampling_rate':rate},return_tensors='pt')
        inputs={k:processed[k] for k in ('input_ids','attention_mask','input_features','input_features_mask')}
        config=json.loads((path/'config.json').read_text())
        config['vision_config']=None
        config['audio_seq_len']=inputs['input_features'].shape[1]
        config['max_seq_len']=inputs['input_ids'].shape[1]
        (path/'audio-config.json').write_text(json.dumps(config),encoding='utf-8')
        cfg=EmbeddingGemma2Config(**config)
        cfg.dtype=cfg.text_config.dtype=cfg.audio_config.dtype=torch.float32
        cfg._attn_implementation='eager'
        cfg.audio_config._attn_implementation='sdpa'
        weights={}
        with safe_open(path/'model.safetensors',framework='pt') as f:
            for key in f.keys():
                if key.startswith(('language_model.','audio_tower.','embed_audio.')):
                    weights[key]=f.get_tensor(key)
        save_file(weights,path/'audio.safetensors')
        model=EmbeddingGemma2Model(cfg).float().eval()
        model.load_state_dict(weights,strict=True)
        del weights
        if device.upper()=='CUDA': model=model.cuda(); inputs={k:v.cuda() for k,v in inputs.items()}
        with torch.no_grad():
            start=time.perf_counter()
            result=model(**inputs)
            mask=inputs['attention_mask']
            pooled=(result.last_hidden_state*mask[...,None]).sum(1)/mask.sum(1)[:,None]
            normalized=torch.nn.functional.normalize(pooled,dim=-1)
        np.savez(path/'reference-audio.npz',**{k:v.cpu().numpy() for k,v in inputs.items()},
                 last_hidden_state=result.last_hidden_state.cpu().numpy(),
                 audio_hidden_states=result.audio_hidden_states.cpu().numpy(),
                 sentence_embedding=normalized.cpu().numpy())
        print(json.dumps(dict(reference_seconds=time.perf_counter()-start,
                              inputs={k:list(v.shape) for k,v in inputs.items()})),flush=True)
        return
    import polygrad as pg
    fixture=np.load(path/'reference-audio.npz')
    config=json.loads((path/'audio-config.json').read_text())
    inputs={k:fixture[k].astype(np.float32 if k=='input_features' else np.int32)
            for k in ('input_ids','attention_mask','input_features','input_features_mask')}
    with pg.create(device=device) as rt:
        start=time.perf_counter()
        model=pg.Model.from_hf(config_json=json.dumps(config),
                              weight_bytes_list=[(path/'audio.safetensors').read_bytes()],
                              max_batch=inputs['input_ids'].shape[0],max_seq_len=inputs['input_ids'].shape[1],runtime=rt)
        print(json.dumps(dict(load_seconds=time.perf_counter()-start)),flush=True)
        try:
            results=[]
            for run in range(2):
                start=time.perf_counter()
                result=model.forward(**inputs)
                np.savez(path/f'polygrad-audio-{device.lower()}.npz',**result)
                error=float(np.max(np.abs(result['sentence_embedding']-fixture['sentence_embedding'])))
                print(json.dumps(dict(run=run,seconds=time.perf_counter()-start,max_abs=error,
                                      text_errors=fp32_vector_errors(result['last_hidden_state'],fixture['last_hidden_state']),
                                      audio_errors=fp32_vector_errors(
                                          result['audio_hidden_states'][result['audio_attention_mask'].astype(bool)],
                                          fixture['audio_hidden_states']))),flush=True)
                results.append(result)
            # Report both runs and all error metrics even if an accumulated
            # raw-state bound fails; retain every assertion below unchanged.
            for result in results:
                np.testing.assert_allclose(result['sentence_embedding'],fixture['sentence_embedding'],atol=2e-6,rtol=2e-5)
                assert_fp32_vectors(result['last_hidden_state'],fixture['last_hidden_state'])
                selected=result['audio_hidden_states'][result['audio_attention_mask'].astype(bool)]
                assert_fp32_vectors(selected,fixture['audio_hidden_states'])
        finally: model.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fetch', 'reference', 'polygrad'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default=os.environ.get('POLY_DEV', os.environ.get('DEV', 'CPU')))
    parser.add_argument('--trace', action='store_true', help='record embedding and layer outputs')
    parser.add_argument('--image', action='store_true', help='exercise the pretrained image/text path')
    parser.add_argument('--audio', action='store_true', help='exercise the pretrained audio/text path')
    parser.add_argument('--photo', choices=['flower.jpg','china.jpg'], default='flower.jpg')
    parser.add_argument('--strict-raw', action='store_true', help='also enforce the text-only accumulated-state bound on images')
    args = parser.parse_args()
    if args.image and args.audio: parser.error('select --image or --audio')
    path = args.checkpoint
    if args.mode == 'fetch':
        from huggingface_hub import snapshot_download
        snapshot_download('google/embeddinggemma-2',
                          revision='914f7f89142e33e77833254d9c9b90c3cef7303b',
                          local_dir=path, allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                                                         'tokenizer_config.json','processor_config.json','preprocessor_config.json'])
        return
    config = json.loads((path / 'config.json').read_text())
    if args.audio:
        audio_probe(path,args.mode,args.device)
        return
    if args.mode == 'reference':
        if args.image:
            image_reference(path,args.device,args.trace,args.photo)
            return
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from tokenizers import Tokenizer
        from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel
        torch.set_num_threads(4)
        text_config = EmbeddingGemma2TextConfig(**config['text_config'])
        text_config._attn_implementation = 'eager'
        weights = {}
        with safe_open(path / 'model.safetensors', framework='pt') as f:
            for key in f.keys():
                if key.startswith('language_model.'):
                    weights[key.removeprefix('language_model.')] = f.get_tensor(key)
        # The subset retains original BF16 bytes. Polygrad's importer converts
        # to FP32, just as this reference does, without requantizing weights.
        save_file(weights, path / 'text.safetensors')
        model = EmbeddingGemma2TextModel(text_config).float().eval()
        model.load_state_dict(weights, strict=True)
        del weights
        traced = {}
        hooks = []
        if args.trace:
            def capture(name):
                def hook(module, inputs, output):
                    traced[name] = output.detach().cpu().numpy().copy()
                return hook
            hooks.append(model.embed_tokens.register_forward_hook(capture('hidden_states.0')))
            for i, layer in enumerate(model.layers):
                hooks.append(layer.register_forward_hook(capture(f'hidden_states.{i+1}')))
        tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        texts = ['task: search result | query: What causes the northern lights?',
                 'title: none | text: Charged particles from the sun cause the northern lights.',
                 'title: none | text: A recipe for baking sourdough bread.']
        encoded = tokenizer.encode_batch(texts)
        length = max(len(e.ids) for e in encoded)
        ids = np.zeros((len(texts), length), np.int32)
        mask = np.zeros_like(ids)
        for row, e in enumerate(encoded):
            ids[row,:len(e.ids)] = e.ids
            mask[row,:len(e.ids)] = 1
        with torch.no_grad():
            start = time.perf_counter()
            hidden = model(torch.tensor(ids), attention_mask=torch.tensor(mask)).last_hidden_state
            pooled = (hidden * torch.tensor(mask)[...,None]).sum(1) / torch.tensor(mask).sum(1)[:,None]
            expected = torch.nn.functional.normalize(pooled, dim=-1).numpy()
        np.savez(path / 'reference.npz', input_ids=ids, attention_mask=mask,
                 last_hidden_state=hidden.numpy(), sentence_embedding=expected, **traced)
        for hook in hooks:
            hook.remove()
        print(json.dumps(dict(reference_seconds=time.perf_counter()-start, shape=ids.shape,
                              similarities=(expected @ expected.T).tolist())), flush=True)
    else:
        import polygrad as pg
        tag = 'image' if args.photo == 'flower.jpg' else 'image-china'
        fixture = np.load(path / (f'reference-{tag}.npz' if args.image else 'reference.npz'))
        ids, mask = fixture['input_ids'], fixture['attention_mask']
        text_config = json.loads((path/f'{tag}-config.json').read_text()) if args.image else config['text_config']
        text_config = {**text_config,'output_hidden_states':args.trace or args.image}
        inputs = dict(input_ids=ids.astype(np.int32),attention_mask=mask.astype(np.int32))
        if args.image:
            inputs.update(pixel_values=fixture['pixel_values'].astype(np.float32),
                          image_position_ids=fixture['image_position_ids'].astype(np.int32))
        with pg.create(device=args.device) as rt:
            start = time.perf_counter()
            model = pg.Model.from_hf(config_json=json.dumps(text_config),
                                    weight_bytes_list=[(path / ('image.safetensors' if args.image else 'text.safetensors')).read_bytes()],
                                    max_batch=ids.shape[0], max_seq_len=ids.shape[1], runtime=rt)
            print(json.dumps(dict(load_seconds=time.perf_counter()-start)), flush=True)
            try:
                for run in range(2):
                    start = time.perf_counter()
                    result = model.forward(**inputs)
                    seconds = time.perf_counter()-start
                    if args.image:
                        np.savez(path / f'polygrad-{tag}-{args.device.lower()}.npz', **result)
                    if args.trace:
                        np.savez(path / f'polygrad-{args.device.lower()}-trace.npz', **result)
                        for name, value in result.items():
                            if name not in fixture: continue
                            ref = fixture[name]
                            if name == 'image_hidden_states':
                                value=value[result['image_attention_mask'].astype(bool)]
                            delta = value.astype(np.float64) - ref
                            relative = np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-30)
                            print(json.dumps(dict(output=name, max_abs=float(np.abs(delta).max()),
                                                  relative_l2=float(relative.max()))), flush=True)
                    expected, actual = fixture['sentence_embedding'], result['sentence_embedding']
                    error = float(np.max(np.abs(expected-actual)))
                    hidden_error = float(np.max(np.abs(fixture['last_hidden_state']-result['last_hidden_state'])))
                    cosine = np.sum(expected*actual, axis=-1)
                    print(json.dumps(dict(run=run, seconds=seconds, max_abs=error, hidden_max_abs=hidden_error,
                                          cosine=cosine.tolist(), similarities=(actual @ actual.T).tolist())), flush=True)
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
                    assert np.all(cosine > 0.99999)
                    if args.image:
                        valid=result['image_attention_mask'].astype(bool)
                        for name, value in [('image_hidden_states',result['image_hidden_states'][valid]),
                                            ('last_hidden_state',result['last_hidden_state'])]:
                            target=fixture[name]
                            relative=np.linalg.norm(value-target,axis=-1)/np.maximum(np.linalg.norm(target,axis=-1),1e-30)
                            print(json.dumps(dict(raw_output=name, accumulated_relative_l2=float(relative.max()))),flush=True)
                            if args.strict_raw:
                                assert_fp32_vectors(value,target)
                    else:
                        assert_fp32_vectors(result['last_hidden_state'], fixture['last_hidden_state'])
            finally:
                model.dispose()
        if args.image:
            # Release Polygrad's model/runtime before constructing the large HF
            # reference. Both use original checkpoint values promoted to FP32.
            import torch
            from safetensors.torch import load_file
            from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model
            torch.set_num_threads(4)
            cfg=EmbeddingGemma2Config(**text_config)
            cfg._attn_implementation='eager'
            cfg.dtype=cfg.text_config.dtype=cfg.vision_config.dtype=torch.float32
            reference=EmbeddingGemma2Model(cfg).float().eval()
            reference.load_state_dict(load_file(path/'image.safetensors'),strict=True)
            checked=assert_image_stages(reference,inputs,result)
            print(json.dumps(dict(stages_checked=len(checked),stage_errors=checked)),flush=True)


if __name__ == '__main__':
    main()
