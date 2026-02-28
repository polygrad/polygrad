"""
HuggingFace model loader for polygrad.

Loads models from config.json + safetensors files into PolyInstance.
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
import numpy as np
from . import _ffi
from .instance import Instance

_lib = _ffi._lib
_u8p = _ffi._u8p


def load_hf(model_path, max_batch=1, max_seq_len=0):
    """Load a HuggingFace model from a local directory.

    Args:
        model_path: Path to directory containing config.json and safetensors.
        max_batch: Maximum batch size for the model instance.
        max_seq_len: Maximum sequence length (0 = use config default).

    Returns:
        Instance: A PolyInstance ready for forward pass.
    """
    model_path = pathlib.Path(model_path)

    # Read config.json
    config_path = model_path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'config.json not found in {model_path}')
    config_bytes = config_path.read_bytes()

    # Find safetensors files
    weight_files = _find_safetensors(model_path)
    if not weight_files:
        raise FileNotFoundError(f'No safetensors files found in {model_path}')

    # Read all weight files into memory
    weight_data = []
    for wf in weight_files:
        weight_data.append(wf.read_bytes())

    return _load_from_bytes(config_bytes, weight_data, max_batch, max_seq_len)


def load_hf_bytes(config_json, weight_bytes_list, max_batch=1, max_seq_len=0):
    """Load from raw bytes (useful for non-filesystem sources).

    Args:
        config_json: config.json content as bytes or str.
        weight_bytes_list: List of safetensors file contents as bytes.
        max_batch: Maximum batch size.
        max_seq_len: Maximum sequence length (0 = use config default).

    Returns:
        Instance: A PolyInstance ready for forward pass.
    """
    if isinstance(config_json, str):
        config_json = config_json.encode('utf-8')
    return _load_from_bytes(config_json, weight_bytes_list, max_batch, max_seq_len)


def download_hf(repo_id, cache_dir=None):
    """Download a HuggingFace model to a local directory.

    Requires huggingface_hub: pip install huggingface_hub

    Args:
        repo_id: HuggingFace model repo (e.g. 'gpt2', 'hf-internal-testing/tiny-random-gpt2').
        cache_dir: Optional cache directory.

    Returns:
        str: Path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            'huggingface_hub is required for download_hf. '
            'Install with: pip install huggingface_hub'
        )
    kwargs = {'allow_patterns': ['*.json', '*.safetensors']}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    return snapshot_download(repo_id, **kwargs)


def generate(instance, tokens, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressive text generation.

    The model runs at fixed max_seq_len (buffers pre-allocated). Input is padded
    to max_seq_len, and logits are extracted at the actual last token position.

    Args:
        instance: PolyInstance with forward pass.
        tokens: Initial token IDs as numpy array of shape (1, seq_len).
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (1.0 = no change).
        top_k: If set, only sample from top-k logits.

    Returns:
        numpy array of shape (1, seq_len + max_new_tokens) with generated tokens.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
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
        x_padded = np.zeros((1, max_seq_len), dtype=np.float32)
        x_padded[0, :actual_len] = tokens[0]

        positions = np.arange(max_seq_len, dtype=np.float32).reshape(1, -1)
        arange = np.arange(max_seq_len, dtype=np.float32)

        outputs = instance.forward(x=x_padded, positions=positions, arange=arange)
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
        next_token = np.array([[np.random.choice(probs.shape[-1], p=probs[0])]], dtype=np.float32)
        tokens = np.concatenate([tokens, next_token], axis=1)

    return tokens


def _find_safetensors(model_path):
    """Find safetensors files, handling both single and sharded models."""
    model_path = pathlib.Path(model_path)

    # Check for index file (sharded model)
    index_path = model_path / 'model.safetensors.index.json'
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        # Get unique shard filenames
        shard_names = sorted(set(index.get('weight_map', {}).values()))
        return [model_path / name for name in shard_names if (model_path / name).exists()]

    # Single file
    single = model_path / 'model.safetensors'
    if single.exists():
        return [single]

    # Fallback: any safetensors file
    files = sorted(model_path.glob('*.safetensors'))
    return files


def _load_from_bytes(config_bytes, weight_data_list, max_batch, max_seq_len):
    """Internal: call poly_hf_load with byte buffers."""
    if isinstance(config_bytes, str):
        config_bytes = config_bytes.encode('utf-8')

    n_files = len(weight_data_list)

    # Create ctypes arrays for weight files
    file_ptrs = (_u8p * n_files)()
    file_lens = (ctypes.c_int64 * n_files)()
    # Keep references to prevent GC
    bufs = []

    for i, data in enumerate(weight_data_list):
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        bufs.append(buf)
        file_ptrs[i] = ctypes.cast(buf, _u8p)
        file_lens[i] = len(data)

    ptr = _lib.poly_hf_load(
        config_bytes, len(config_bytes),
        file_ptrs, file_lens,
        n_files, max_batch, max_seq_len
    )

    if not ptr:
        raise RuntimeError('poly_hf_load returned NULL')

    return Instance(ptr)


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
