"""
HuggingFace model loader for polygrad.

Loads config.json with safetensors or restricted PyTorch ZIP state dictionaries.
No file I/O happens in C -- this module reads files and passes byte buffers.

Usage:
    from polygrad.hf import load_hf, download_hf

    # From local directory
    inst = load_hf('/path/to/model')

    # Download + load
    path = download_hf('gpt2')
    inst = load_hf(path)
"""

import ctypes
import json
import os
import pathlib
from contextlib import ExitStack
import numpy as np
from . import _ffi
from .device import _device_id
from .model import Model, _import_context

_get_lib = _ffi.get_lib
_u8p = ctypes.POINTER(ctypes.c_uint8)


def load_hf(model_path, max_batch=1, max_seq_len=0, device=None, *, runtime=None):
    """Load a HuggingFace model from a local directory.

    Args:
        model_path: Directory containing config.json and a supported checkpoint.
        max_batch: Maximum batch size for the model instance.
        max_seq_len: Maximum sequence length (0 = use config default).

    Returns:
        Model: A PolyModel ready for forward pass.
    """
    model_path = pathlib.Path(model_path)

    # Read config.json
    config_path = model_path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {model_path}')
    config_bytes = config_path.read_bytes()

    # Prefer safetensors, including an authoritative shard index. Never fall
    # back after a missing shard or parse error in the preferred format.
    names = [p.relative_to(model_path).as_posix() for p in model_path.rglob('*') if p.is_file()]
    plan = plan_hf(names)
    if plan['index']:
        plan = plan_hf(names, (model_path / plan['index']).read_text(encoding='utf-8'))
    weight_files = [model_path / name for name in plan['files']]

    if plan['format'] == 'safetensors':
        with ExitStack() as stack:
            sources = [stack.enter_context(wf.open('rb')) for wf in weight_files]
            return load_hf_sources(config_bytes, sources, max_batch, max_seq_len, device, runtime=runtime)

    # Restricted PyTorch ZIP import retains its existing whole-file path.
    weight_data = []
    for wf in weight_files:
        weight_data.append(wf.read_bytes())

    return _load_from_bytes(config_bytes, weight_data, max_batch, max_seq_len, device, runtime)


def load_hf_sources(config_json, sources, max_batch=1, max_seq_len=0, device=None, *, runtime=None):
    """Import safetensors from immutable seekable binary files, one tensor at a time.

    Files stay caller-owned. Conversion may need tensor-sized scratch storage;
    device weights are uploaded without staging the entire checkpoint.
    """
    config = config_json.encode('utf-8') if isinstance(config_json, str) else bytes(config_json)
    sources = tuple(sources)
    if b'\0' in config:
        raise ValueError('checkpoint config contains NUL')
    lib, ctx = _get_lib(), _import_context(runtime)
    session = lib.poly_hf_import_new(ctx, config, max_batch, max_seq_len,
                                     _device_id(device) if device is not None else 0)

    def check(ok):
        if not ok:
            detail = lib.poly_import_last_error_message()
            raise RuntimeError('HF import failed: ' + (detail.decode('utf-8', 'replace') if detail else 'unknown error'))

    def read(source, offset, length):
        source.seek(offset)
        data = source.read(length)
        if len(data) != length:
            raise ValueError('short safetensors range')
        return data

    check(session)
    try:
        for source in sources:
            size = source.seek(0, os.SEEK_END)
            prefix = read(source, 0, 8)
            length = int.from_bytes(prefix, 'little')
            if length > 100_000_000 or length > size - 8:
                raise ValueError('invalid safetensors header size')
            header = prefix + read(source, 8, length)
            check(lib.poly_hf_import_add_file(session, ctypes.cast(ctypes.c_char_p(header), _u8p), len(header), size) == 0)
        check(lib.poly_hf_import_build(session) == 0)
        while (request := lib.poly_hf_import_request(session)):
            item = json.loads(request)
            data = read(sources[item['file']], item['offset'], item['length'])
            check(lib.poly_hf_import_feed(session, ctypes.cast(ctypes.c_char_p(data), _u8p), len(data)) == 0)
        model = lib.poly_hf_import_finish(session)
        check(model)
        return Model._from_handle(model, ctx)
    finally:
        lib.poly_hf_import_free(session)


def load_hf_bytes(config_json, weight_bytes_list, max_batch=1, max_seq_len=0, device=None, *, runtime=None):
    """Load from raw bytes (useful for non-filesystem sources).

    Args:
        config_json: config.json content as bytes or str.
        weight_bytes_list: Safetensors or supported PyTorch ZIP files as bytes.
        max_batch: Maximum batch size.
        max_seq_len: Maximum sequence length (0 = use config default).

    Returns:
        Model: A PolyModel ready for forward pass.
    """
    if isinstance(config_json, str):
        config_json = config_json.encode('utf-8')
    return _load_from_bytes(config_json, weight_bytes_list, max_batch, max_seq_len, device, runtime)


def download_hf(repo_id, cache_dir=None, *, revision=None):
    """Download a HuggingFace model to a local directory.

    Requires huggingface_hub: pip install huggingface_hub

    Args:
        repo_id: HuggingFace model repo (e.g. 'gpt2', 'hf-internal-testing/tiny-random-gpt2').
        cache_dir: Optional cache directory.
        revision: Optional pinned Hub commit or revision.

    Returns:
        str: Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import model_info, hf_hub_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required for download_hf. '
            'Install with: pip install huggingface_hub'
        )
    info = model_info(repo_id, revision=revision)
    names = [item.rfilename for item in info.siblings]
    # Resolve a moving branch once; metadata and payloads use the same commit.
    if not info.sha or len(info.sha) != 40 or any(c not in '0123456789abcdef' for c in info.sha):
        raise ValueError('Hub did not return a commit identifier')
    kwargs = dict(revision=info.sha, cache_dir=cache_dir)
    plan = plan_hf(names)
    if plan['index']:
        index = hf_hub_download(repo_id, plan['index'], **kwargs)
        plan = plan_hf(names, pathlib.Path(index).read_text(encoding='utf-8'))
    config = hf_hub_download(repo_id, 'config.json', **kwargs)
    for name in plan['files']:
        hf_hub_download(repo_id, name, **kwargs)
    return str(pathlib.Path(config).parent)


def plan_hf(files, index=None):
    """Select checkpoint filenames without I/O using Polygrad's shared C planner.

    ``files`` contains repository-relative names. If the result names an index
    and has an empty files list, read that index and pass its JSON as ``index``.
    The result contains ``format``, ``index`` and sorted unique ``files``.
    """
    if not isinstance(files, (list, tuple)) or any(not isinstance(x, str) for x in files):
        raise TypeError('files must be a sequence of filenames')
    listing = json.dumps(files).encode('utf-8')
    data = (index if isinstance(index, str) else json.dumps(index)).encode('utf-8') if index is not None else None
    if b'\0' in listing or (data is not None and b'\0' in data):
        raise ValueError('checkpoint metadata contains NUL')
    lib = _get_lib()
    size = lib.poly_hf_plan(listing, data, None, 0)
    if size < 0:
        raise ValueError(lib.poly_import_last_error_message().decode('utf-8'))
    out = ctypes.create_string_buffer(size)
    if lib.poly_hf_plan(listing, data, out, size) != size:
        raise RuntimeError('checkpoint planning failed')
    return json.loads(out.value)


def generate(instance, tokens, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressive text generation.

    The model runs at fixed max_seq_len (buffers pre-allocated). Input is padded
    to max_seq_len, and logits are extracted at the actual last token position.

    Args:
        instance: PolyModel with forward pass.
        tokens: Initial token IDs as numpy array of shape (1, seq_len).
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (1.0 = no change).
        top_k: If set, only sample from top-k logits.

    Returns:
        numpy array of shape (1, seq_len + max_new_tokens) with generated tokens.
    """
    tokens = np.asarray(tokens)
    if not np.issubdtype(tokens.dtype, np.integer):
        raise TypeError(f'tokens must have an integer dtype, got {tokens.dtype}')
    tokens = tokens.astype(np.int32, copy=False)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)

    vocab_size = _get_vocab_size(instance)
    max_seq_len = _get_max_seq_len(instance)

    for _ in range(max_new_tokens):
        actual_len = tokens.shape[1]
        if actual_len > max_seq_len:
            raise RuntimeError(
                f'Sequence length {actual_len} exceeds max_seq_len {max_seq_len}')

        # Pad input to max_seq_len
        x_padded = np.zeros((1, max_seq_len), dtype=np.int32)
        x_padded[0, :actual_len] = tokens[0]

        positions = np.arange(max_seq_len, dtype=np.int32).reshape(1, -1)
        outputs = instance.forward(x=x_padded, positions=positions)
        logits = outputs.get('output')
        if logits is None:
            raise RuntimeError('Model did not produce output buffer')

        # Reshape to (batch, max_seq_len, vocab)
        logits = logits.reshape(1, max_seq_len, vocab_size)

        # Get logits at actual last token position
        next_logits = logits[:, actual_len - 1, :]  # (1, vocab)

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            topk_indices = np.argpartition(next_logits, -top_k, axis=-1)[:, -top_k:]
            mask = np.full_like(next_logits, -1e9)
            np.put_along_axis(mask, topk_indices, np.take_along_axis(next_logits, topk_indices, axis=-1), axis=-1)
            next_logits = mask

        # Softmax
        next_logits = next_logits - next_logits.max(axis=-1, keepdims=True)
        probs = np.exp(next_logits)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Sample
        next_token = np.array(
            [[np.random.choice(probs.shape[-1], p=probs[0])]], dtype=np.int32
        )
        tokens = np.concatenate([tokens, next_token], axis=1)

    return tokens


def _load_from_bytes(config_bytes, weight_data_list, max_batch, max_seq_len, device=None, runtime=None):
    """Internal: call poly_hf_load with byte buffers."""
    if isinstance(config_bytes, str):
        config_bytes = config_bytes.encode('utf-8')

    n_files = len(weight_data_list)

    # Create ctypes arrays for weight files
    file_ptrs = (_u8p * n_files)()
    file_lens = (ctypes.c_int64 * n_files)()
    # C reads shards synchronously and copies decoded state into the Model.
    # Keep immutable bytes alive without duplicating whole checkpoint files;
    # mutable buffers still need a snapshot while ctypes releases the GIL.
    bufs = []

    for i, data in enumerate(weight_data_list):
        buf = (ctypes.c_char_p(data) if isinstance(data, bytes)
               else (ctypes.c_uint8 * len(data)).from_buffer_copy(data))
        bufs.append(buf)
        file_ptrs[i] = ctypes.cast(buf, _u8p)
        file_lens[i] = len(data)

    ctx = _import_context(runtime)
    ptr = _get_lib().poly_hf_load_into(
        ctx, config_bytes, len(config_bytes),
        file_ptrs, file_lens,
        n_files, max_batch, max_seq_len, _device_id(device) if device is not None else 0
    )

    if not ptr:
        detail = _get_lib().poly_import_last_error_message()
        raise RuntimeError('poly_hf_load returned NULL' +
                           (': ' + detail.decode('utf-8', 'replace') if detail else ''))

    return Model._from_handle(ptr, ctx)


def _get_vocab_size(instance):
    """Infer vocab size from wte.weight param shape."""
    for i in range(instance.param_count):
        if instance.param_name(i) == 'wte.weight':
            shape = instance.param_shape(i)
            if len(shape) >= 1:
                return shape[0]
    return 50257  # GPT-2 default


def _get_max_seq_len(instance):
    """Infer max_seq_len from x input buffer shape."""
    for i in range(instance.buf_count):
        if instance.buf_name(i) == 'x':
            shape = instance.buf_shape(i)
            if len(shape) >= 2:
                return shape[1]
    return 1024  # GPT-2 default
