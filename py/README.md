# Polygrad Python

tinygrad-compatible Tensor API for Python. Thin ctypes wrapper around the C core — each method is one FFI call.

## Installation

```bash
# Build the C shared library first
make

# Install Python package (editable)
pip install -e py/

# Or set POLYGRAD_LIB to point to your libpolygrad.so
export POLYGRAD_LIB=/path/to/libpolygrad.so
```

Requires: Python >= 3.9, numpy.

## Quick Start

```python
from polygrad import Tensor

# Create tensors
a = Tensor.rand(3, 4)
b = Tensor.rand(4, 5)

# Matrix multiply + softmax
c = (a @ b).softmax(-1)
print(c.numpy())

# Autograd
x = Tensor([1.0, 2.0, 3.0])
x.requires_grad = True
loss = (x * x).sum()
loss.backward()
print(x.grad.numpy())  # [2.0, 4.0, 6.0]
```

## Training Example

```python
from polygrad import Tensor
from polygrad.nn import Linear, SGD, get_parameters

Tensor.manual_seed(42)
model = Linear(2, 1)
opt = SGD(get_parameters(model), lr=0.01)

for i in range(100):
    opt.zero_grad()
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[5.0], [11.0]])
    loss = (model(x) - target).square().mean()
    loss.backward()
    opt.step()

print(f"loss: {loss.item():.4f}")
```

## Tensor API

### Construction

| Method | Description |
|--------|-------------|
| `Tensor(data)` | From list, numpy array, or scalar |
| `Tensor.zeros(*shape)` | Tensor of zeros |
| `Tensor.ones(*shape)` | Tensor of ones |
| `Tensor.full(shape, val)` | Tensor filled with value |
| `Tensor.rand(*shape)` | Uniform random [0, 1) |
| `Tensor.randn(*shape)` | Standard normal |
| `Tensor.randint(low, high, shape)` | Random integers [low, high) |
| `Tensor.arange(stop, start=0, step=1)` | Arithmetic progression |
| `Tensor.linspace(start, stop, steps)` | Evenly spaced values |
| `Tensor.eye(n)` | Identity matrix |
| `Tensor.empty(*shape)` | Uninitialized tensor |
| `Tensor.manual_seed(seed)` | Set random seed |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | tuple | Dimension sizes |
| `ndim` | int | Number of dimensions |
| `dtype` | str | Always `'float32'` |
| `device` | str | Always `'CPU'` |
| `T` | Tensor | Transpose of last two dims |
| `requires_grad` | bool | Settable; enables autograd |
| `grad` | Tensor/None | Gradient after `.backward()` |

### Realization & Conversion

| Method | Returns | Description |
|--------|---------|-------------|
| `realize()` | Tensor | Execute lazy graph, return self |
| `numpy()` | ndarray | Realize and return numpy array |
| `item()` | float | Scalar value |
| `tolist()` | list | Nested Python list |
| `numel()` | int | Total elements |
| `size(dim=None)` | tuple/int | Shape or dimension size |
| `detach()` | Tensor | Copy without graph |
| `clone()` | Tensor | Copy preserving requires_grad |

### Arithmetic

```python
a + b, a - b, a * b, a / b, -a, a ** b
```

All support broadcasting and scalar operands.

### Comparisons

```python
a < b, a == b, a != b, a > b, a >= b, a <= b
```

Returns float tensor (1.0 = true, 0.0 = false).

### Element-wise Math

| Method | Description |
|--------|-------------|
| `exp()` | e^x |
| `log()` | ln(x) |
| `sqrt()` | Square root |
| `square()` | x^2 |
| `abs()` | Absolute value |
| `sign()` | Sign (-1, 0, +1) |
| `reciprocal()` | 1/x |
| `rsqrt()` | 1/sqrt(x) |
| `sin()`, `cos()`, `tan()` | Trigonometric |
| `ceil()`, `floor()`, `round()`, `trunc()` | Rounding |
| `isnan()`, `isinf()` | NaN/Inf detection |
| `exp2()`, `log2()` | Base-2 functions |
| `where(x, y)` | Conditional: self ? x : y |
| `maximum(other)` | Element-wise max |
| `minimum(other)` | Element-wise min |
| `clamp(min_=None, max_=None)` | Clamp to range |

### Activations

| Method | Description |
|--------|-------------|
| `relu()` | max(0, x) |
| `relu6()` | clamp(relu(x), 0, 6) |
| `leaky_relu(neg_slope=0.01)` | Leaky ReLU |
| `sigmoid()` | 1 / (1 + e^-x) |
| `tanh()` | Hyperbolic tangent |
| `gelu()` | Gaussian Error Linear Unit |
| `quick_gelu()` | Fast GELU approximation |
| `silu()` / `swish()` | x * sigmoid(x) |
| `elu(alpha=1.0)` | Exponential Linear Unit |
| `softplus(beta=1.0)` | log(1 + e^(beta*x)) / beta |
| `mish()` | x * tanh(softplus(x)) |
| `hardtanh(min_val=-1, max_val=1)` | Clamped linear |
| `hardswish()` | Hard swish |
| `hardsigmoid()` | Hard sigmoid |

### Reductions

| Method | Description |
|--------|-------------|
| `sum(axis=None, keepdim=False)` | Sum along axes |
| `max(axis=None, keepdim=False)` | Maximum along axes |
| `min(axis=None, keepdim=False)` | Minimum along axes |
| `mean(axis=None, keepdim=False)` | Mean along axes |
| `var(axis=None, keepdim=False, correction=1)` | Variance |
| `std(axis=None, keepdim=False, correction=1)` | Standard deviation |

### Movement / Shape

| Method | Description |
|--------|-------------|
| `reshape(*shape)` / `view(*shape)` | Reshape (supports -1) |
| `permute(*order)` | Permute dimensions |
| `transpose(dim0=-2, dim1=-1)` | Swap two dimensions |
| `expand(*shape)` | Broadcast to shape |
| `squeeze(dim=None)` | Remove size-1 dims |
| `unsqueeze(dim)` | Add size-1 dim |
| `flatten(start_dim=0, end_dim=-1)` | Flatten dim range |
| `unflatten(dim, sizes)` | Split dim into multiple |
| `shrink(arg)` | Slice: [(start, end), ...] |
| `pad(arg)` | Pad: [(before, after), ...] |
| `flip(axis)` | Reverse along axes |
| `repeat(*repeats)` | Tile tensor |

### Linear Algebra

| Method | Description |
|--------|-------------|
| `matmul(other)` / `dot(other)` / `@` | Matrix multiplication |
| `linear(weight, bias=None)` | x @ weight.T + bias |

### Normalization & Loss

| Method | Description |
|--------|-------------|
| `softmax(axis=-1)` | Softmax normalization |
| `log_softmax(axis=-1)` | Log-softmax |
| `layernorm(axis=-1, eps=1e-5)` | Layer normalization |
| `cross_entropy(target, axis=-1)` | Cross-entropy loss |
| `binary_crossentropy(target)` | Binary cross-entropy |

### Advanced Operations

| Method | Description |
|--------|-------------|
| `Tensor.einsum(formula, *operands)` | Einstein summation |
| `rearrange(formula, **kwargs)` | einops-style rearrange |
| `Tensor.cat(*tensors, dim=0)` | Concatenate along dim |
| `Tensor.stack(*tensors, dim=0)` | Stack along new dim |
| `split(sizes, dim=0)` | Split into chunks |
| `chunk(n, dim=0)` | Split into n chunks |
| `__getitem__` | Indexing: int, slice, None, Ellipsis |

### Autograd

```python
x = Tensor([1.0, 2.0])
x.requires_grad = True
loss = (x * x).sum()
loss.backward()
print(x.grad.numpy())  # [2.0, 4.0]
```

Call `backward()` on a scalar loss before calling `item()` or `numpy()` on the loss.

## nn Module

### Layers

```python
from polygrad.nn import Linear, LayerNorm, RMSNorm, Embedding, Dropout
```

| Class | Signature | Description |
|-------|-----------|-------------|
| `Linear(in_f, out_f, bias=True)` | y = x @ W.T + b | Fully connected layer |
| `LayerNorm(shape, eps=1e-5)` | (x - mean) / sqrt(var + eps) * w + b | Layer normalization |
| `RMSNorm(dim, eps=1e-5)` | x / rms(x) * w | Root mean square normalization |
| `Embedding(vocab, dim)` | Lookup table | Token embedding |
| `Dropout(p=0.5)` | Random zeroing | Training-only (controlled by `Tensor.training`) |
| `GroupNorm(groups, channels)` | Group normalization | Per-group normalization |

### Optimizers

```python
from polygrad.nn import SGD, Adam, AdamW, get_parameters
```

| Class | Signature |
|-------|-----------|
| `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)` |
| `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)` |
| `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` |

All optimizers have `step()` and `zero_grad()` methods.

### State Dict

```python
from polygrad.nn import get_parameters, get_state_dict, load_state_dict

params = get_parameters(model)       # List of Tensor
sd = get_state_dict(model)           # {'weight': Tensor, 'bias': Tensor, ...}
load_state_dict(model2, sd)          # Load params into another model
```

## Compiled Training Steps

Compile a training step into a reusable C program. The first call traces the computation graph; subsequent calls execute with zero scheduling overhead.

```python
from polygrad import Tensor
from polygrad.nn import Linear, SGD, get_parameters, compile_step

Tensor.manual_seed(42)
model = Linear(4, 1)
opt = SGD(get_parameters(model), lr=0.01)

# Sample inputs (shapes must match at runtime)
x = Tensor.rand(8, 4)
y = Tensor.rand(8, 1)

def train_step(model, opt, x, y):
    loss = (model(x) - y).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

# Compile: traces forward + backward + optimizer into one PolyStep
step = compile_step(train_step, model, opt, x, y)

# Run: executes compiled kernels with current buffer data
for i in range(100):
    x._data[:] = ...  # update input data in-place
    y._data[:] = ...
    step.run()
    print(f"step {i}: loss = {step.loss_value():.4f}")
```

`compile_step` returns a `CompiledTrainingStep` with:
- `run()` -- execute all compiled kernels (forward + backward + optimizer)
- `loss_value()` -- read the loss scalar from the output buffer
- `n_kernels` -- number of compiled kernels
- `n_intermediates` -- number of pre-allocated intermediate buffers

## How It Works

1. **Lazy evaluation**: Operations build a UOp graph in the C core. No computation happens until `realize()`, `numpy()`, `item()`, or `backward()`.
2. **One FFI call per op**: Each Tensor method calls one C function via ctypes. The C core handles all op composition (e.g., `gelu` = `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`).
3. **Realize boundaries**: Some ops (softmax, layernorm, var) insert implicit `.realize()` calls to create kernel boundaries for the scheduler.
4. **Autograd**: `backward()` calls C's `poly_grad()` for each parameter, then realizes the gradient tensors.

## Limitations

- float32 only
- CPU only (GPU backends planned)
- Conv2d and BatchNorm are stubs (forward raises NotImplementedError)

## Tests

```bash
python -m pytest py/tests/ -v   # 87 tests (tensor + nn + compiled step + GPT-2)
```
