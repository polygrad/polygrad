# Polygrad Python

The Python API for Polygrad, a tensor library and JIT compiler built around a
C11 port of tinygrad. It provides automatic differentiation, neural-network
layers, optimizers and NumPy data exchange. Models saved in Python can load in
JavaScript without their original Python classes.

[Project overview and shared C/runtime reference](https://github.com/polygrad/polygrad#readme) | [JavaScript guide](https://github.com/polygrad/polygrad/blob/main/js/README.md)

## Contents

- [Install](#install)
- [Quickstart: Tensors](#quickstart-tensors)
- [Quickstart: Models](#quickstart-models)
- [Working With Tensors](#working-with-tensors)
- [Training](#training)
- [Models](#models)
- [Devices And Runtimes](#devices-and-runtimes)
- [JIT And Compile](#jit-and-compile)
- [Custom Kernels](#custom-kernels)
- [Package Integration](#package-integration)
- [API Overview](#api-overview)
- [Troubleshooting](#troubleshooting)

## Install

```bash
pip install polygrad
```

Requirements:

- Linux
- Python 3.9 or newer (installed wheels tested on CPython 3.9-3.13)
- NumPy
- A C compiler on `PATH` for CPU execution: clang recommended, GCC fallback;
  generated float16 kernels require clang's `__fp16` support

CPython 3.9-3.13 on x86_64 Linux with glibc 2.27+ can install a prebuilt wheel.
Other Linux configurations build from source and require a C compiler and Python
development headers. Wheels do not remove the CPU backend's runtime compiler
requirement; use `DEV=X86` (x86_64) or `DEV=INTERP` to run without one.

Optional dependency for downloading Hugging Face checkpoints (loading local
config and weight files does not need it):

```bash
pip install huggingface_hub
```

For a source checkout, see [building from source](https://github.com/polygrad/polygrad#building-from-source).

## Quickstart: Tensors

```python
from polygrad import Tensor

a = Tensor.rand(3, 4)
b = Tensor.rand(4, 5)
c = (a @ b).softmax(-1)

print(c.numpy())
```

Autograd:

```python
from polygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])

loss = (x * x).sum()
loss.backward()

print(x.grad.numpy())  # [2. 4. 6.]
```

## Quickstart: Models

Fit `y = 3x + 2` in Python, save its graph and weights, then load it in
JavaScript without redefining the model.

```python
from polygrad import Model, Tensor

class Linear:
    def __init__(self):
        self.a = Tensor([0.0])
        self.b = Tensor([0.0])
    def __call__(self, x):
        return {"prediction": self.a * x + self.b}

model = Model(
    Linear(), inputs={"x": Tensor.empty(5)}, targets={"y": Tensor.empty(5)},
    loss=lambda outputs, y: (outputs["prediction"] - y).square().mean(),
)
try:
    model.fit({"x": [-2, -1, 0, 1, 2], "y": [-4, -1, 2, 5, 8]},
              epochs=100, optimizer="sgd", lr=0.1)
    model.save("linear.pgb", include_optimizer=False)
finally:
    model.dispose()
```

In Node, using a matching Polygrad package:

```javascript
const { Model } = require('polygrad')
const model = Model.load('linear.pgb')
try {
  const { prediction } = model.forward({x: new Float32Array([3, 4, 5, 6, 7])})
  console.log(Array.from(prediction)) // approximately [11, 14, 17, 20, 23]
} finally {
  model.dispose()
}
```

See [Models](#models) for capture, input shapes, minibatches and export options.

## Working With Tensors

### Creation

```python
from polygrad import Tensor

x = Tensor([1, 2, 3])
a = Tensor.zeros(2, 3)
b = Tensor.ones(2, 3)
c = Tensor.randn(2, 3)
d = Tensor.arange(0, 6).reshape(2, 3)
```

### Math And Indexing

```python
from polygrad import Tensor

x = Tensor.arange(0, 12).reshape(3, 4)
y = x.permute(1, 0).reshape(2, 6)
z = y.relu().sum(axis=1)
picked = x.gather(1, Tensor([[0, 2], [1, 3], [0, 1]], dtype="int32"))
```

Probability math uses shared C implementations: `x.erf()`, `x.erfc()`,
`x.erfinv()`, `x.ndtri()` (standard Normal quantile), `x.digamma()` and
`x.lgamma()` (log absolute Gamma). These support autograd and Model capture;
derivatives at domain endpoints and poles are not defined.
Use `x.log_ndtr()` for the standard Normal log-CDF and `(-x).log_ndtr()`
for its log survival probability; taking `log()` of a CDF can lose the tails.

### NumPy And Data Updates

```python
import numpy as np
from polygrad import Tensor

arr = np.array([1, 2, 3, 4], dtype=np.float32)
x = Tensor(arr).reshape(2, 2)
print((x * 2 + 1).numpy())
```

Polygrad tensors are lazy. Use `realize()` to execute and `numpy()` when host
readback is needed.

```python
import numpy as np
from polygrad import Tensor

x = Tensor.empty((4,), dtype="float32")
x.copy_from(np.array([1, 2, 3, 4], dtype=np.float32))

y = (x * 3 - 1).realize()
print(y.numpy())

x.update_from(np.array([5, 6, 7, 8], dtype=np.float32))
print((x + 1).realize().numpy())
```

Use `copy_from` or `update_from` for repeated loops that should preserve input
buffer identity for compiled replay.

### Linear Algebra

```python
from polygrad import Tensor

A = Tensor([[4.0, 2.0], [2.0, 5.0]])
b = Tensor([1.0, 3.0])
x = A.solve(b)

print(x.numpy())
```

Structured linalg methods are portable tensor-composed fallbacks tested against
NumPy and Torch. They do not add LAPACK or runtime library dependencies.
Current `lstsq` is solution-only for full-rank tall or square systems.

## Training

Use Tensor autograd when you want to control the training loop:

```python
from polygrad import Context, Tensor
from polygrad.nn import Linear, get_parameters
from polygrad.nn.optim import SGD

Tensor.manual_seed(42)
model = Linear(2, 1)
opt = SGD(get_parameters(model), lr=0.01)

with Context(TRAINING=1):
    for _ in range(100):
        opt.zero_grad()
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[5.0], [11.0]])
        loss = (model(x) - y).square().mean()
        loss.backward()
        opt.step()

print(loss.item())
```

### Layers And Optimizers

```python
from polygrad.nn import Linear, LayerNorm, RMSNorm, Embedding
from polygrad.nn import get_parameters
from polygrad.nn.optim import SGD, Adam, AdamW
```

Layers include `Linear`, `LayerNorm`, `LayerNorm2d`, `RMSNorm`, `Embedding`,
`Dropout`, `GroupNorm`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`,
`InstanceNorm`, `LSTMCell`, and `BatchNorm`. Optimizers include `SGD`, `Adam`, and
`AdamW`; each provides `step()` and `zero_grad()`.

`nn.gated_delta_rule(q, k, v, log_decay, beta, state=None)` returns the causal
attention output and final state without mutating the supplied state. It supports
concrete shapes and chunked prefill; accumulation and state use float32.

## Models

Start with [Quickstart: Models](#quickstart-models) for capture, fitting and export.
More runnable scripts are in [Python examples](https://github.com/polygrad/polygrad/tree/main/py/examples).

For stateful loops, `model.call('step', {}, controls={'iteration': i}, readback=False)`
executes without collecting outputs and returns `{}`. Read a named state/output
with `model.read_buffer(name)` when needed. The shared `models.Graph` description
supports fixed-shape state and explicit entrypoint writes; see the
[Graph schema](https://github.com/polygrad/polygrad#configuration-driven-models).

### Capture and input rules

- `controls` declares named integer Variables. Pass values through
  `model.call(..., controls={...})`. An entrypoint's `controls` mapping overrides
  shared declarations; `{}` declares none. Controls must occur in that entrypoint's
  graph. They are currently inference-only; combining them with a loss is rejected
  during construction. Pass training data through Tensor inputs instead.
- `params` defaults to named Tensor attributes of the supplied object;
  functions require explicit closure state. `model.summary()` inspects metadata without executing or
  reading weights.
- Two independently named lazy constants, such as `a = Tensor(2.0)` and
  `r = Tensor(2.0)`, can merge into one logical node and cause
  `ambiguous state occurrence` during capture. Reuse one Tensor if the names
  should share state. For independent state, construct one from host data, e.g.
  `a = Tensor(np.array(2.0, dtype=np.float32))`. Calling `.realize()` on the
  scalar constant alone is not sufficient.
- With `loss`, construction captures evaluation and training forwards against
  the same state. The callable runs twice during construction, never during
  execution. Without a loss, capture uses the current `TRAINING` value;
  changing it later does not recapture the graph.
- Inside an author function, `child.forward(x=x)` with Tensor inputs inlines
  an existing Model. Fixed Python loops share its weights; slicing and sampling
  can stay in the combined graph. Import children before capture, on the same
  runtime/device. Inference capture supports static value entrypoints and explicit
  state inputs/outputs; internal writes, integer controls and input checks are
  rejected. The parent owns one weight copy, independent of later child writes or disposal;
  both copies occupy memory until the child is disposed.
- Set the seed before construction. BatchNorm/RNG updates affect Model-owned
  state. Authoring Tensor roots and training mode are restored even on failure;
  arbitrary Python side effects are not. Capture rejects parameter/input
  assignments and effectful reads.
- Variable-size calls support bounded dimensions, including `[N, N]` and
  independent `[N, M]` axes. State shapes remain fixed.
  Save/load preserves the signature; storage currently reserves
  maximum capacity. Tensor results retain their invocation's values and shape
  across later calls and Model disposal, but require a live runtime.
- Flat arrays can infer one variable axis; multiple variable axes require a
  shaped array. Multidimensional NumPy arrays must match the
  declared shape, not merely its element count. The equivalent JS binding is
  `{data: typedArray, shape: [rows, columns]}`.

`model.place('CPU:1')` and `set_device_map(...)` accept the same exact native
CPU identities. Unsupported accelerator ordinals reject without changing the Model.

### Minibatches

- Use `model.fit(data, epochs=2, batch_size=32)` when every input/target's
  leading-axis bounds admit 32. Epochs visit samples in input order and return
  one loss per step. Omit `batch_size` to repeat one full batch.
- An incomplete final batch rejects before training unless `remainder='drop'`
  discards it or `remainder='keep'` processes the smaller extent permitted by
  every input's bounds. No padding or shuffling is implicit.
- Tensor datasets are sliced on-device without frontend host readback and may
  mix with host inputs.

### Tabular Models

For numerical tabular data, `rt.models.MLP`, `NAM` and `TabM` accept
`batch_size` as an upper bound; smaller batches work with `remainder='keep'`.
TabM trains individual members and averages them for regression. With
`loss='cross_entropy'`, its `output` is log mean member probabilities; apply
softmax to obtain probabilities. `model.call('members', x=x)['members']`
returns raw `[batch, members, outputs]` predictions.
Cross-entropy targets are class probabilities or one-hot rows, not integer IDs.

TabM's `dropout` (default `0`) applies only during training. Its seeded random
state advances for each actual batch and is included in saved models; inference
uses the same parameters without dropout. Optional `num_embeddings` accepts
`{'type': 'linear_relu', 'd_embedding': 16}` or
`{'type': 'piecewise_linear', 'd_embedding': 16, 'bins': edges}`.
Supply strictly increasing bin edges for each numerical feature, fitted on
training data. Piecewise embeddings use the reference's version B with an
optional `activation=True` (default `False`). `layers[0]` remains the number of
raw features. Categorical encoding and fitting bins are outside this template.

For quantile regression, set `loss='pinball'`, `quantiles=[0.1, 0.5, 0.9]`
and three output units (`layers[-1]`, or NAM's `n_outputs`). Targets have shape
`[batch, 1]`. Levels must be strictly increasing and between zero and one;
the loss averages samples, levels and (for TabM) members. It does not constrain
quantile crossing or perform calibration.

After configuring an optimizer, `model.set_learning_rate(lr)` changes its rate
without rebuilding the training graph or resetting moments. For separate rates
and weight decay, call `model.set_optimizer_groups([{'params': ['weight'],
'lr': 0.001, 'weight_decay': 0.01}, {'params': ['bias'], 'lr': 0.01}])` after
`set_optimizer` and before the first training step. Cover every parameter once;
list only one name for tied parameters. Omitted rates inherit the optimizer's
defaults. Repeat this call to change rates or regroup parameters without
resetting moments; use it instead of `set_learning_rate` when groups are enabled.
After loading a checkpoint, reapply the optimizer configuration and groups;
saved moments and random state resume from the checkpoint.
Training rejects a grouped checkpoint until its groups have been reapplied.

### Saving And Loading

`model.save()` returns bundle bytes; `model.save(path)` also writes a file.
`Model.load(bytes_or_path)` uses the default runtime, or pass `runtime=rt`.
Each load owns independent state, while tied weights within a Model stay tied.
Disposing a Model does not dispose its runtime.
Newer releases load older portable models; files saved by a newer release may
require that release or later. Check the changelog when exchanging models.

Bundles preserve auxiliary/RNG and optional optimizer state.
`model.import_weights(data)` requires every persistent parameter, with matching
names, shapes and dtypes. Missing or unknown names are errors; use `write_buffer`
for partial updates.
`include_optimizer=False` omits optimizer state, not training entrypoints.
To resume training, reapply the optimizer configuration after loading.
See [export products](https://github.com/polygrad/polygrad#export-products)
for graph, program and weight exports and version compatibility.

### ONNX Import

`Model.from_onnx(path_or_bytes, dimensions={'batch': 2})` imports a supported
ONNX graph as an ordinary inference Model. Omit `dimensions` for fixed shapes.
Use `model.call('forward', inputs)` with the graph's input names; outputs retain
their ONNX names. The imported Model supports placement and bundle save/load.

For external weights, pass `external_data={'weights.bin': weight_bytes}`;
locations are exact keys, not files fetched by the loader. Initializers are
copied into frozen state; this import does not create a training objective or
Transformer generation methods. Unsupported operators and shapes raise an
import error; see [ONNX import](https://github.com/polygrad/polygrad#onnx-import).
Constant-terminated loops with fixed-shape carried values are unrolled during
import; runtime-dependent loops and scan outputs are rejected.

### Pretrained models

<!-- readme-test: network -->
```python
from polygrad.hf import download_hf, load_hf, generate
import numpy as np

model_path = download_hf("hf-internal-testing/tiny-random-gpt2")
model = load_hf(model_path, max_batch=1, max_seq_len=16)
try:
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
    result = generate(model, tokens, max_new_tokens=2, temperature=1.0, top_k=10)
    print(result)
finally:
    model.dispose()
```

Local safetensors imports read and upload one tensor at a time. For other
seekable binary sources, use `polygrad.hf.load_hf_sources(config_json, sources,
max_seq_len=16)`. Sources remain caller-owned and must not change during import.
Staging is bounded by the largest tensor, with additional conversion or
tied-weight scratch; PyTorch ZIP imports still read whole files.

`load_hf` uses the same checkpoint importer as `Model.from_hf`, including the
vision and embedding models below, with F32/F16/BF16 safetensors or supported
PyTorch ZIP state dictionaries.
The `generate` helper above expects GPT-2 input/output names;
it is not a Llama generation API. Qwen loading uses the shared C/GGUF path.

### Hugging Face checkpoints

Local loading selects `model.safetensors`, then `pytorch_model.bin`; each
format's single file takes precedence over its `.index.json`. Other filenames
are not collected automatically. Missing shards and malformed files raise errors; a broken
safetensors checkpoint is never silently replaced with `.bin`. Pass
`revision="<commit>"` to `download_hf` for a pinned download. Only one weight
format is downloaded, without downloading unrelated precision variants.
`polygrad.hf.plan_hf(filenames, index=None)` exposes the same C selection used
by JavaScript: it returns `format`, `index` and sorted unique `files`. If it
selects an index with no files yet, read the index and call again with its JSON.
Planning does not validate weight contents. For arbitrary `.pt` filenames, pass the file contents
through `Model.from_hf(config_json=..., weight_bytes_list=[...])`.

The C reader accepts uncompressed ZIP/ZIP64 archives containing a flat mapping
of names to ordinary tensors (pickle protocols 2–5). It never executes Python
code. Supported storage types are float32/16, bfloat16, float64, signed integers
8/16/32/64, uint8 and bool; individual model builders may accept fewer types.
Storage offsets and nonnegative strides are checked. Non-contiguous views are
packed; shared checkpoint storage does not impose shared trainable Model state.

Pickled model objects, custom classes, nested training checkpoints, compressed
ZIP entries and pre-ZIP serialization are rejected. Limits include eight tensor
axes, 100,000 ZIP entries/tensors, a 64 MiB pickle index, a 64 MiB parser metadata
budget and 256 MiB of packed-view data per archive. Referenced entries must have
valid ZIP checksums. Downloads and tokenization remain outside the C loader.

`models.GPT2(config)` and `models.Llama(config)` build topology without pretrained
weights. They reject execution and saving until every required parameter has been
written; use a checkpoint loader for a ready-to-run model. Explicit zero weights
are valid. Wrong input dtypes are rejected, not silently narrowed.

`models.list()` (also `rt.models.list()`) reports construction and checkpoint
capabilities; see [supported models](https://github.com/polygrad/polygrad#supported-models).
Calling `models.Qwen3(...)` reports that it is import-only and points to
`Model.from_gguf(...)`; it does not construct an uninitialized model.

Whisper HF imports expose `encode(input_features)` followed by `prefill(tokens_prefill)`
or `decode(tokens_decode)`, called through `Model.call`. They accept batch-one
FP32 log-mel features `[1, num_mel_bins, 2*max_source_positions]` and int32 token
IDs. Decoder calls require `controls={'start_pos': position}`; `max_seq_len`
sets the maximum prefill chunk. `decode_sample` takes `tokens_decode` and
`sampling.temperature` and returns `sampling.decode_token`, sampling on-device.
Call `reset_transient()` and encode again for a new recording. Saved models load
with empty caches. Audio preprocessing, tokenization and transcription policy
(language/task tokens, suppression and timestamp rules) remain the caller's job;
this is not a replacement for Whisper's full `transcribe` pipeline.

`Qwen35Text` imports a `qwen3_5_text` configuration and text-backbone weights
through `Model.from_hf`. Its `forward` takes int32 `input_ids`, `attention_mask`
and `position_ids`, all shaped `[max_batch, max_seq_len]`, and returns float32
`last_hidden_state`. F32/F16/BF16 weight storage is preserved. Only full attention
obeys `is_causal`; linear attention stays causal. With `cache_capacity` in a causal
configuration, use `model.call('prefill', {'tokens_prefill': ids},
controls={'start_pos': offset})` or `decode` with `tokens_decode` for one token.
These return `hidden_prefill` or `hidden_decode`; prefill takes `max_seq_len`
tokens. Supply consecutive offsets; `reset_transient()` clears the caches, and
saved models load with empty caches. This is a text backbone, not a token
generator, vision model or complete sentence-embedding pipeline.

Composite `qwen3_5` support uses the same importer with `modalities: ['text']`
in their configuration. `Qwen3_5Model` returns hidden states;
`Qwen3_5ForConditionalGeneration` returns `logits`, `logits_prefill` and
`logits_decode` (the latter shaped `[batch, vocabulary]`). A cached single-batch
generator also exposes `decode_sample` with the same inputs as Whisper above.
Tied heads share embedding storage.

For images, select `modalities: ['text', 'image']` and set `image_grid_thw` in the
configuration to the processor's grid array. This specializes the graph to those
grids; a different grid requires another import. `forward_image` takes int32
`input_ids`, `attention_mask`, `image_position_ids` (`[3, batch, sequence]`) and
`image_grid_thw`, plus float32 packed `pixel_values` from the checkpoint's image
processor. It returns `logits_image` or `hidden_image`. Incorrect image-slot counts
and grids are rejected before execution. With caches, `prefill_image` takes the
same inputs except `attention_mask`, requires an unpadded prompt, and returns
`logits_prefill_image` (`[batch, vocabulary]`) or `hidden_prefill_image`.
Follow it with `decode`/`decode_sample` at `start_pos=max_seq_len`; rotary offsets
are retained internally. Reset caches before a new prompt. Tokenization, image
processing and multimodal position IDs remain the caller's responsibility.
Use the checkpoint's `generation_config.json` for stopping tokens; nested text
configuration defaults can differ.

[Late token embeddings](examples/late_embeddings.py) composes a backbone, a
checkpoint projection and normalization through the public Model API, with a
document-only token mask. The same recipe supports Perplexity's late-embedding
checkpoint; use its original tokenizer, prompts and skip-token IDs. The importer
does not automatically interpret Sentence Transformers pipeline metadata.

For cached Qwen3 generation, use `models.Transformer.from_gguf(data,
cache_capacity=4096, prefill_chunk_size=64)`, or the runtime-bound
`rt.models.Transformer.from_gguf(...)`. The loader validates the generation
contract; `Model.from_gguf` accepts the same cache options but stays generic.

Qwen3 GGUF imports accept only int32 token input `x` and return `output`:
`model.forward(x=token_ids)['output']`. Rotary tables are Model-owned state,
included in saved bundles. Older bundles retain their original signatures;
use `model.entrypoints()` to inspect them.

Llama supports dense Llama 2/base 3 and text-only 3.x `llama3` RoPE scaling.
Set `cache_capacity` and `prefill_chunk_size` in its configuration to enable
batch-one FP32 cached decoding and return a `models.Transformer` subclass.
After loading weights, `model.append_tokens(ids)`
accepts int32 `[N]` or `[1,N]` tokens and returns last-token logits `[1,vocab]`.
C handles variable-width chunks up to `prefill_chunk_size` and checks capacity
before writing the cache. `model.decode_position` reports the committed
count. `reset()` starts a new conversation without changing weights or rewinding
the sampling RNG. Saved bundles preserve RNG state but load with empty KV caches.
Use `models.Transformer.load(bundle)` to restore generation methods;
`Model.load(bundle)` stays generic. `model.generate(ids, temperature=0, max_tokens=32)`
yields token IDs sampled in C. `prefill_tokens(ids)` reuses a matching prompt
prefix; `rewind(position)` discards a suffix without clearing cache storage.
Load or adopt only trusted Transformer bundles: bounds checks do not verify
the causal, append-only cache semantics required by prefix reuse and rewind.
`polygrad.llm.model.Transformer` aliases `models.Transformer`; it does not
provide Tinygrad constructor or method compatibility.

After a failed append, cache-writing direct call, weight write or placement during
a conversation, reset before appending again. Read-only calls and rejected inputs
preserve the conversation. Borrowed C storage pointers must not mutate an active decoder.
Uncached `forward(tokens=...)` remains available. MoE, arbitrary RoPE schemes,
tokenizers and chat templates are not supplied by this loader.

The C Model GGUF loader converts F32/F16/BF16, I8/I16/I32 and complete
Q4_0/Q4_1/Q8_0/Q6_K blocks. Q4_K/Q5_K metadata is readable but weight conversion
is unsupported. This is narrower than Python's Tensor-level GGUF decoder;
decoding a file does not imply support for its model architecture.

### Vision models

`models.CLIP`, `models.ViT`, `models.DINOv2` and `models.DINOv3` use shared C
components. Construct from a configuration, write all required weights, or load
a Hugging Face directory with `Model.from_hf(path)`. Unloaded models reject
execution and saving. Their bundles load in JavaScript with `pg.Model.load(bytes)`.

| Model | Inputs | Forward outputs |
| --- | --- | --- |
| CLIP | `pixel_values`, `input_ids` | `image_embeds`, `text_embeds`, `logits_per_image`, `logits_per_text` |
| ViT | `pixel_values` | `last_hidden_state`, `pooler_output` (tanh pooler) |
| DINOv2 / DINOv3 | `pixel_values` | `last_hidden_state`, `pooler_output` (CLS token) |

Inputs are already normalized float32 images in `[batch, channels, height, width]`
layout. Use the checkpoint's image processor; the Model does not decode, resize
or normalize images. CLIP additionally takes int32, right-padded token IDs with
an EOS token. Its `encode_image` and `encode_text` entrypoints accept one modality
and return normalized embeddings. Both modalities use the configured batch size.
Normalization clamps the L2 norm to at least `1e-12`; zero projections return
zero embeddings rather than NaNs.

This implementation is fixed-square-resolution, unmasked inference. It supports
HF `CLIPModel`, `ViTModel`, `Dinov2Model` and `DINOv3ViTModel` weights, not
classification heads, DINOv2-with-registers, DINOv3 ConvNeXt, training augmentations
or dynamic input resolutions. DINOv2's optional `input_image_size` config selects
a fixed input resolution and bicubically interpolates the original checkpoint's
positions; `image_size` must retain the checkpoint resolution. DINOv3 includes register tokens
and patch-only 2D RoPE; `last_hidden_state` retains those prefix tokens. DINOv2
SwiGLU and DINOv3 gated MLP configurations are supported. JSON type tags are
`clip`, `vit`, `dinov2` and `dinov3_vit`.

For C-built model types and JSON-based `models.Sequential` / `models.Graph`, see
[shared JSON reference](https://github.com/polygrad/polygrad#configuration-driven-models).
These return the same Model type and use the same training and export APIs.
Sequential/Graph configurations support typed inputs, one bounded leading batch
dimension, and shared embedding, normalization, RoPE and attention components.
For example, `{"dtype":"int32","shape":[{"name":"batch","min":1,"max":32},16]}`
declares token batches of 1 to 32 rows without rebuilding the model.

### EmbeddingGemma 2

`models.EmbeddingGemma2Text` constructs the text backbone from a resolved HF
text config. For a full `embedding_gemma2` checkpoint, set
`"modalities": ["text"]` in its parsed configuration for text-only inference;
there is no need to remove the image/audio configuration. Pass the serialized
config and a list of safetensors file contents together as
`Model.from_hf(config_json=config_text, weight_bytes_list=weight_files)`.
These are bytes/text, not filenames; use `Model.from_hf(directory)` when the
checkpoint config needs no changes.
Omitting `modalities` includes every configured tower.
Computation and imported parameters use float32.

Pass fixed-shape int32 `input_ids` and int32 `attention_mask` (1 for tokens,
0 for padding) to `forward`.
Outputs are projected `last_hidden_state` and normalized `sentence_embedding`.
Both left and right padding are supported; masked tokens are excluded from
pooling. Tokenization and task prompts use the official HF tokenizer outside
Polygrad. Shortening an embedding requires L2 normalization after slicing.

For image inference, select `"modalities": ["text", "image"]`.
Set `image_num_patches` to the HF processor's patch capacity, then
pass its float32 `pixel_values` `[batch, patches, 3*patch_size*patch_size]`
and int32 `image_position_ids` `[batch, patches, 2]` alongside tokens and mask.
Use one nonempty image per batch item and processor-generated matching image
token slots; mismatches are rejected before inference, including after Model
save/load. Token IDs and patch positions must be host arrays, not Tensors.
These are flattened, unnormalized patches with `(x,y)` positions, not the
normalized NCHW images used by the vision models above. Padding positions are
`(-1,-1)`. Outputs also include padded `image_hidden_states` and their boolean
`image_attention_mask`.

For audio, select `"modalities": ["text", "audio"]`, set `audio_seq_len` to the processor's feature
frame count, and pass float32 `input_features` `[batch, frames, features]` plus
an int32 0/1 `input_features_mask` `[batch, frames]`. Supply matching audio-token
slots in `input_ids`; mismatches are rejected before inference. Token IDs and
feature masks must be host arrays, not Tensors. Outputs include
padded `audio_hidden_states` and `audio_attention_mask`.
Video and clipped vision projections are not implemented. Tokenization and preprocessing remain
external; no Hugging Face packages are required by Polygrad itself.
This is an embedding model, not a text generator. Set `output_hidden_states`
to `true` in the config to expose per-layer text/image/audio hidden states.

## Devices And Runtimes

Compiler settings use the process environment, shared by native runtimes.
Set them before running the program, for example
`REDUCEOP_SPLIT_THRESHOLD=1024 python train.py`. This uses the existing reduction
splitting rule; lower thresholds can trade extra kernels for shorter accumulation
chains, but provide no accuracy guarantee. The default threshold is `32768`.
Integer settings read by the shared C environment helper diagnose malformed or
out-of-range values and use their defaults; they do not raise Python exceptions.

Choose a device explicitly or through the environment:

```python
from polygrad import Device, Tensor

device = "cuda" if Device.cuda_available() else "cpu"
x = Tensor.rand(4, device=device)
y = x * 2

print(y.numpy())
```

Environment variables:

```bash
DEV=CPU python app.py
POLY_DUMP_KERNELS=1 python app.py
BEAM=4 python app.py
```

Device selection uses an explicit `device=`, then `POLY_DEV`, then `DEV`.
Names are case-insensitive; available native backends include CPU, CUDA, X86
and INTERP. Unsupported device ordinals and multi-device targets are rejected.

Explicit runtimes are optional. Tinygrad-style programs use `from polygrad import
Tensor, nn` and `from polygrad.nn.optim import Adam` without `create()` or a Model
wrapper. The Python package uses a module-level default C context. Caller-created tensors
share that context, so package functions should accept and return `Tensor`
objects rather than copying through NumPy unless readback is required.

For explicit context/device ownership, create a runtime:

```python
import polygrad

rt = polygrad.create(device="cpu")
try:
    layer = rt.nn.Linear(2, 1)
    x = rt.Tensor([[1.0, 2.0]])
    print(layer(x).numpy())
finally:
    rt.dispose()
```

Use explicit runtimes for isolation, device-specific package wiring, or tests
that need independent compiler caches.

Use `rt.nn` for layers, optimizers and state loaders, `rt.models` for C-backed
model types, and `rt.Model` for capture/import on that runtime. For example,
`rt.Model.load("model.pgb")` is equivalent to
`polygrad.Model.load("model.pgb", runtime=rt)`.
Create tensors and layers on the same runtime: mixing owners is rejected even
when their devices match; package-level constructors use the default runtime.

### Threads

Never use one runtime concurrently from multiple Python threads: native calls
release the GIL, and overlapping context mutations can crash the process.
The default runtime is process-global. For parallel work, create one runtime
per thread, construct through `rt.Tensor` and `rt.Model`, and keep its objects
in that thread, including cleanup.
Separate-runtime CPU, INTERP and X86 execution is tested; concurrent GPU
initialization is not validated. Alternatively, serialize all use and cleanup
of a shared runtime yourself. Global configuration changes still need coordination.
Call `collect()` and `dispose()` only when no other thread is using the runtime.

### Runtime Inspection

```python
import polygrad

print(polygrad.stats())
print(polygrad.can_run("add", shape=[1024]))
```

`can_run(...)` is conservative. For some compound op/shape queries it raises
when support cannot be proven statically.

## JIT And Compile

`@jit` follows tinygrad's raw Tensor JIT behavior. The first call runs normally,
the second call captures realized schedules, and later calls replay those
schedules with current input buffers.

```python
from polygrad import Tensor, jit

@jit
def step(x):
    return (x + 1).realize()

print(step(Tensor([1, 2, 3])).numpy())  # normal run
print(step(Tensor([4, 5, 6])).numpy())  # capture
print(step(Tensor([7, 8, 9])).numpy())  # replay
```

For embedding loops, `compile(...)` performs the same warmup and capture up
front and exposes an explicit callable:

```python
from polygrad import Tensor, compile

def step(x):
    return (x + 1).realize()

compiled = compile(step, [Tensor([1, 2, 3]).realize()])
out = compiled.run([Tensor([7, 8, 9]).realize()])

print(out.numpy())
print(compiled.stats())
compiled.dispose()
```

`polygrad.stats()` exposes shared C runtime counters. The tinygrad-compatible
top-level `GlobalCounters` provides `global_ops`, `global_mem`, `time_sum_s`,
`kernel_count`, `mem_used`, `mem_used_per_device`, and `reset()`; reset keeps
live allocation bytes. `Jit.stats()` and `CompiledCallable.stats()` expose
wrapper-level capture and replay counters.

`polygrad.can_run(op, dtype="float32", shape=..., shapes=..., device="auto")`
is an advisory backend capability probe.

Capture requires at least one Tensor argument. Host reads must stay outside the
captured function; otherwise their values would be fixed during capture.
Compiler settings such as BEAM and NOOPT are documented in the
[shared runtime reference](https://github.com/polygrad/polygrad#settings).

### Reusing Input Buffers

```python
import numpy as np
from polygrad import Tensor, compile

x = Tensor(np.array([1, 2, 3], dtype=np.float32)).realize()
f = compile(lambda x: x.square().sum().realize(), [x])

print(f.run([x]).item())
x.copy_from(np.array([4, 5, 6], dtype=np.float32))
print(f.run([x]).item())
f.dispose()
```

## Custom Kernels

`Tensor.custom_kernel(...)` mirrors tinygrad's alpha custom-kernel shape. The
kernel function receives placeholder UOps and returns a `SINK` body. Polygrad
wraps the body in `CALL`, returns `AFTER(...)` tensors, and keeps execution in
the normal schedule and runtime caches.

```python
from polygrad import Tensor
from polygrad.uop.ops import KernelInfo, UOp

def add_kernel(out, a, b):
    out, a, b = out.flatten(), a.flatten(), b.flatten()
    i = UOp.range(out.ctx, out.numel(), 0)
    return out[i].store(a[i] + b[i]).end(i).sink(
        arg=KernelInfo(name="custom_add_4")
    )

out = Tensor.empty((4,), dtype="float32")
y = out.custom_kernel(
    Tensor([1.0, 2.0, 3.0, 4.0]),
    Tensor([10.0, 20.0, 30.0, 40.0]),
    fxn=add_kernel,
)[0]
print(y.numpy())  # [11. 22. 33. 44.]
```

`KernelInfo` is required to mark the executable kernel boundary. Without it,
the body can leave the output unchanged, as in pinned tinygrad. Match each
stored value's dtype to its destination, using an explicit UOp `cast` when
needed. Current C and Wasm renderers reject mismatched vector stores with
`vector STORE dtype mismatch; cast the value to the destination dtype`.
Scalar C stores can convert numerically, but that is not a portable kernel
contract. INTERP follows Tinygrad's Python memoryview conversion rules. Cast
explicitly on every backend.

## Package Integration

Python packages should accept caller-created Polygrad tensors and return
Polygrad tensors:

```python
from polygrad import Tensor

def normalize(x: Tensor) -> Tensor:
    mean = x.mean(axis=-1, keepdim=True)
    scale = (x - mean).square().mean(axis=-1, keepdim=True).sqrt()
    return (x - mean) / scale.maximum(1e-6)
```

This keeps execution in the caller's Polygrad context and avoids unnecessary
NumPy readback. The denominator guard returns zeros for constant inputs.

### C-authored graphs

`rt.load_extension(path)` attaches a separately built C author to this runtime's
existing core. The author calls existing `poly_tensor_*` functions; it does not
link another Polygrad implementation. The installed package includes its author
SDK; authoring requires Python, Clang and a C compiler, not a Polygrad checkout:

```bash
python -m polygrad.sdk --source author.c --entry build_graph --inputs 1 --outputs 1 --output build/author
cc -std=c11 -O2 -shared -fPIC -I"$(python -m polygrad.sdk --include)" author.c build/author/extension.c -lm -o author.so
```

Clang reads the packaged public declarations; unsupported pointer signatures
reject. New authors use an extension ABI and declare required imports:
additive core changes do not invalidate them. Legacy authors still use the
original exact core-ABI check. An incompatible contract or missing import fails
at load time. Ordinary package installation does not require SDK build tools.

Stateful authors can add `--methods methods.json`. Method names select C
declarations and import roots; the JSON supplies ownership, not signatures.
The installed `_sdk/examples/` directory contains a complete example. Compile generated
`methods.c` **instead of** the original author source, alongside `extension.c`.
Call `author.methods.<name>(...)`; context arguments are supplied automatically.
Returned consumer objects expose `dispose()`, not pointer integers. Borrowed
strings last only for the call; returned strings are copied. Authors retain any
Tensor they keep and roll back partial changes on failure. Method authors require
construction contract 2; stateless
authors retain contract 1. Extension disposal closes objects in reverse creation
order, and runtime disposal closes extensions before Models and Tensors.

A `PolyModel *` method argument annotated `{"model": "borrowed"}` accepts a live
Model from this runtime. It lasts only for that synchronous call: authors must
not retain or free it. Native authors can import public Model call/read and
signature queries through the same resolver, keeping evaluation loops in C.
These native-only pointer/record imports reject on Wasm before instantiation;
this does not add asynchronous browser execution. Callers serialize the context.
Public record layouts, signatures and ownership remain extension ABI contracts.
Graph authors can query `poly_tensor_static_dim(ctx, tensor, axis)` without
realizing data. It returns an int64 extent (including zero), or -1 for an invalid
axis/shape or a varying symbolic extent. Axes are nonnegative. Unlike a borrowed
shape pointer, this result is safe across independent Wasm heaps. The reviewed
Tensor `stack` import accepts an array of borrowed Tensor handles and returns
one owned Tensor; its array transport follows the existing `cat` contract.
`poly_uop_dtype_id` is also available to authors as a scalar query. Dtype IDs
remain part of the extension ABI; the string-name query is native-only.

`model.set_metadata('owner', value)` and `model.get_metadata('owner')` store and
read detached application JSON; `None` removes the key. `transformer`, `polygrad`
and `polygrad.*` are reserved. Embedded NULs are rejected. Finite binary64 numbers
round-trip exactly; encode integers outside its exact range as strings. Metadata survives bundles,
not raw IR, and application edits do not invalidate execution caches.

For a runnable example from the checkout root, first run
`make build/extension/author.so`. The supplied `test/extension_fixture.c`
constructs a Gaussian log density and its gradient (mode `0`):

<!-- readme-test: extension -->
```python
import polygrad

with polygrad.create(device='CPU', logical='always') as rt:
    x = rt.Tensor([1., 2., 3.])
    author = rt.load_extension('./build/extension/author.so')
    try:
        logp, gradient = author.build([x, x, x], [0])
        author.dispose()  # Returned Tensors remain valid.
        try:
            assert logp.item() == -7.0
            assert gradient.numpy().tolist() == [-1., -2., -3.]
        finally:
            logp.dispose()
            gradient.dispose()
    finally:
        author.dispose()
        x.dispose()
```

The C entry returns `int` and takes `PolyCtx*`, a borrowed `PolyTensor**` input
array, optional `int`/`double` values, and an owned `PolyTensor**` output array.
It builds graphs only: no execution, retained host callbacks or private struct
access. It must release intermediate references and unwind captures on failure.
Seal results with `rt.Model.from_tensors` when persistence is needed. ABI
mismatches and inputs from another context reject. This loads trusted native
code, not sandboxed plugins. Linux is tested; other native platforms are untested.

## API Overview

`Tokenizer.from_gguf(bytes)` and `Tokenizer.from_json(bytes_or_text)` use the
shared C tokenizer. Instances expose `encode(text)`, `decode(ids)`, `vocab_size`,
`bos_id`, `eos_id`, and `free()`, and support `with` for cleanup. They own their
vocabulary independently of Tensor runtimes; `runtime.Tokenizer` is the same
class. JSON accepts the supported byte-level BPE pipeline. Unsupported pipelines
raise `ValueError`; `strict=False` permits skipping NFC only, with `UserWarning`.
Use Hugging Face tokenizers for other pipelines. No Unicode dependency is added.

`create(device='cpu', kernels=True)` accepts the same kernel option as `Runtime`.
Native CPU and WebGPU kernel providers remain opt-in. The Wasm execution backend
selects supported strict no-pack FP32 matmuls by default; `kernels=False` or
`POLY_KERNELS=0` disables selection. Other operations use shared optimization and
ordinary rendering. The Wasm striped sum order changes rounding under cancellation;
it does not use relaxed multiply-add.

| Area | Main APIs |
|---|---|
| Runtime | `polygrad.create`, `polygrad.stats`, `polygrad.can_run`, `Device` |
| Models | `Model`, `from_callable`, `from_tensors`, `fit`, `forward`, `save`, `load`, `summary`, `dispose`; `polygrad.models` factories |
| Tensor creation | `Tensor(data)`, `zeros`, `ones`, `full`, `rand`, `randn`, `randint`, `arange`, `linspace`, `eye`, `empty` |
| Tensor math | `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `min`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std` |
| Sampling | `logits.sample_logits(temperature=1, top_k=0, keepdim=False)` returns int32 indices along the last axis; finite logits, captured runtime RNG, all ties at the top-k threshold remain eligible (`0` keeps all). |
| Movement/indexing | `reshape`, `view`, `permute`, `transpose`, `expand`, `squeeze`, `unsqueeze`, `flatten`, `shrink`, `pad`, `flip`, `repeat`, `gather`, `take_along_axis`, `cat`, `stack`, `split`, `chunk` |
| Linalg | `matmul`, `dot`, `linear`, `qr`, `triangular_solve`, `solve_triangular`, `cholesky`, `cholesky_solve`, `solve`, `lstsq` |
| Data/readback | `realize`, `numpy`, `item`, `tolist`, `copy_from`, `update_from`, `to`, `cpu`, `cuda`, `detach`, `clone` |
| Compilation | `function`, `jit`, `compile`, `Tensor.custom_kernel` |
| Tokenization | `Tokenizer.from_json`, `Tokenizer.from_gguf`, `encode`, `decode` |
| Neural nets | `polygrad.nn` layers, `SGD`, `Adam`, `AdamW`, `get_parameters`, `get_state_dict` |

## Troubleshooting

- **`unknown type name '__fp16'`:** float16 CPU kernels require Clang.
  Install it and select `CC=clang`; check that `CC` is not forcing GCC.
- **`args mismatch in jit`:** the call must match the traced input shapes,
  dtypes and devices. An `int32` input cannot replace a `float32` sample.
- **Model format mismatch:** upgrade the reader to the version that saved the
  model or a newer release.
  See [bundle compatibility](https://github.com/polygrad/polygrad#export-products);
  do not edit artifact version fields to bypass validation.
- **CPU compilation cannot find a compiler:** install clang (recommended) or GCC,
  or try `DEV=X86`
  on a supported x86 machine or `DEV=INTERP` for interpreted execution.
- **"TRAINING must be enabled":** wrap optimizer steps in
  `with Context(TRAINING=1):`, as in [Training](#training).
- **Operands belong to different runtimes:** create layers and Tensors through
  the same runtime (`rt.nn`, `rt.Tensor`), or use the default API consistently.
- **Memory grows across many different graphs:** at an idle boundary, use
  `rt.clear_schedule_cache()` and `rt.collect()`. Live owners still retain their
  resources; later execution rebuilds cleared schedules. This is a memory-pressure
  tool, not a per-step requirement.

For contributor checks, see [tests](https://github.com/polygrad/polygrad#tests).

## License

MIT
