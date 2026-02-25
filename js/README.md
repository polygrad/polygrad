---
title: JavaScript (Node)
nav_order: 3
---

# Polygrad JavaScript (Node.js)

tinygrad-compatible Tensor API for Node.js. Thin koffi FFI wrapper around the C core — each method is one native call.

## Installation

```bash
# Build the C shared library first
make

# Install JS dependencies
cd js && npm install

# Or set POLYGRAD_LIB to point to your libpolygrad.so
export POLYGRAD_LIB=/path/to/libpolygrad.so
```

Requires: Node.js >= 18, koffi (native FFI).

## Quick Start

```javascript
const { Tensor } = require('./js')

// Create tensors
const a = Tensor.rand(3, 4)
const b = Tensor.rand(4, 5)

// Matrix multiply + softmax
const c = a.matmul(b).softmax(-1)
console.log(c.toArray())

// Autograd
const x = new Tensor([1.0, 2.0, 3.0], { requiresGrad: true })
const loss = x.mul(x).sum()
loss.backward()
console.log(x.grad.toArray())  // Float32Array [2, 4, 6]
```

## Tensor API

### Construction

| Method | Description |
|--------|-------------|
| `new Tensor(data, opts?)` | From number, array, or Float32Array |
| `Tensor.zeros(...shape)` | Tensor of zeros |
| `Tensor.ones(...shape)` | Tensor of ones |
| `Tensor.full(shape, val)` | Filled tensor |
| `Tensor.rand(...shape)` | Uniform random [0, 1) |
| `Tensor.randn(...shape)` | Standard normal |
| `Tensor.randint(low, high, shape)` | Random integers |
| `Tensor.arange(stop, start?, step?)` | Range tensor |
| `Tensor.linspace(start, stop, steps)` | Evenly spaced values |
| `Tensor.eye(n)` | Identity matrix |
| `Tensor.empty(...shape)` | Uninitialized tensor |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `shape` | number[] | Dimension sizes |
| `ndim` | number | Number of dimensions |
| `dtype` | string | Always `'float32'` |
| `device` | string | Always `'CPU'` |
| `T` | Tensor | Transpose of last two dims |
| `requiresGrad` | boolean | Settable; enables autograd |
| `grad` | Tensor/null | Gradient after `.backward()` |

### Realization & Conversion

| Method | Returns | Description |
|--------|---------|-------------|
| `realize()` | Tensor | Execute lazy graph, return self |
| `toArray()` | Float32Array | Realize and return typed array |
| `item()` | number | Scalar value |
| `tolist()` | array | Nested JS arrays |
| `numel()` | number | Total elements |
| `size(dim?)` | number/number[] | Shape or dimension size |
| `detach()` | Tensor | Copy without graph |
| `clone()` | Tensor | Copy preserving requiresGrad |

### Arithmetic

| Method | Description |
|--------|-------------|
| `add(other)` | Addition (broadcasting) |
| `sub(other)` | Subtraction |
| `mul(other)` | Multiplication |
| `div(other)` | Division |
| `neg()` | Negation |
| `pow(other)` | Exponentiation |

All accept Tensor or number.

### Comparisons

| Method | Description |
|--------|-------------|
| `lt(other)` | Less than |
| `eq(other)` | Equal |
| `ne(other)` | Not equal |
| `gt(other)` | Greater than |
| `ge(other)` | Greater or equal |
| `le(other)` | Less or equal |

### Element-wise Math

| Method | Description |
|--------|-------------|
| `exp()`, `log()` | Natural exp/log |
| `exp2()`, `log2()` | Base-2 exp/log |
| `sqrt()`, `rsqrt()` | Square root, reciprocal sqrt |
| `square()`, `abs()`, `sign()` | Square, abs, sign |
| `reciprocal()` | 1/x |
| `sin()`, `cos()`, `tan()` | Trigonometric |
| `ceil()`, `floor()`, `round()`, `trunc()` | Rounding |
| `isnan()`, `isinf()` | NaN/Inf detection |
| `where(x, y)` | Conditional select |
| `maximum(other)`, `minimum(other)` | Element-wise max/min |
| `clamp(lo, hi)` | Clamp to range |

### Activations

| Method | Description |
|--------|-------------|
| `relu()` | max(0, x) |
| `relu6()` | clamp(relu(x), 0, 6) |
| `leakyRelu(negSlope=0.01)` | Leaky ReLU |
| `sigmoid()` | 1 / (1 + e^-x) |
| `tanh()` | Hyperbolic tangent |
| `gelu()` | GELU |
| `quickGelu()` | Fast GELU |
| `silu()` / `swish()` | x * sigmoid(x) |
| `elu(alpha=1.0)` | ELU |
| `softplus(beta=1.0)` | Softplus |
| `mish()` | Mish |
| `hardtanh(minVal, maxVal)` | Hard tanh |
| `hardswish()` | Hard swish |
| `hardsigmoid()` | Hard sigmoid |

### Reductions

| Method | Description |
|--------|-------------|
| `sum(axis?, keepdim?)` | Sum |
| `max({axis?, keepdim?})` | Maximum |
| `min({axis?, keepdim?})` | Minimum |
| `mean(axis?, keepdim?)` | Mean |
| `var(axis?, keepdim?, correction?)` | Variance |
| `std(axis?, keepdim?, correction?)` | Standard deviation |

### Movement / Shape

| Method | Description |
|--------|-------------|
| `reshape(...shape)` / `view(...)` | Reshape (supports -1) |
| `permute(...order)` | Permute dimensions |
| `transpose(dim0?, dim1?)` | Swap two dims |
| `expand(...shape)` | Broadcast to shape |
| `squeeze(dim?)` | Remove size-1 dims |
| `unsqueeze(dim)` | Add size-1 dim |
| `flatten(startDim?, endDim?)` | Flatten dim range |
| `unflatten(dim, sizes)` | Split one dim |
| `shrink(arg)` | Slice: [[s,e], ...] |
| `pad(arg)` | Pad: [[b,a], ...] |
| `flip(axis)` | Reverse along axes |
| `repeat(...repeats)` | Tile tensor |

### Linear Algebra

| Method | Description |
|--------|-------------|
| `dot(w)` / `matmul(w)` | Matrix multiply |
| `linear(weight, bias?)` | x @ weight.T + bias |

### Normalization & Loss

| Method | Description |
|--------|-------------|
| `softmax(axis=-1)` | Softmax |
| `logSoftmax(axis=-1)` | Log-softmax |
| `layernorm(axis?, eps?)` | Layer normalization |
| `crossEntropy(target, axis?)` | Cross-entropy loss |
| `binaryCrossEntropy(target)` | Binary cross-entropy |

### Advanced

| Method | Description |
|--------|-------------|
| `Tensor.einsum(formula, ...ops)` | Einstein summation |
| `rearrange(formula, kwargs?)` | einops-style rearrange |
| `Tensor.cat(...tensors, {dim})` | Concatenate |
| `Tensor.stack(...tensors, {dim})` | Stack |
| `split(sizes, dim?)` | Split into chunks |
| `chunk(n, dim?)` | Split into n chunks |
| `getitem(...idx)` | Indexing (int, [start,stop], null) |

### Autograd

```javascript
const x = new Tensor([1, 2, 3], { requiresGrad: true })
const loss = x.mul(x).sum()
loss.backward()
console.log(x.grad.toArray())  // Float32Array [2, 4, 6]
```

## Differences from Python API

| Python | JavaScript | Notes |
|--------|-----------|-------|
| `a + b` | `a.add(b)` | JS has no operator overloading |
| `a @ b` | `a.matmul(b)` | |
| `a.numpy()` | `a.toArray()` | Returns Float32Array |
| `requires_grad` | `requiresGrad` | camelCase |
| `leaky_relu()` | `leakyRelu()` | camelCase |
| `log_softmax()` | `logSoftmax()` | camelCase |

## How It Works

Same architecture as Python: each method calls one C function via koffi FFI. The C core handles all op composition. Lazy evaluation with explicit `realize()`.

## Tests

```bash
node js/test/test_tensor.js   # 62 tests
```
