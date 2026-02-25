# Polygrad R

Tensor API for R via `.Call()` FFI to the C core. R operator overloads for natural syntax.

## Installation

```bash
# Build the C shared library first
make

# Install R package
R CMD INSTALL r/
```

## Quick Start

```r
library(polygrad)

# Create tensors
a <- Tensor$new(c(1, 2, 3))
b <- Tensor$new(c(4, 5, 6))

# Arithmetic with R operators
c <- a + b
print(c$as_numeric())  # [5, 7, 9]

# Reduction
s <- c$sum_op()
print(s$item())  # 21

# Autograd
x <- Tensor$new(c(1, 2, 3))
x$requires_grad <- TRUE
loss <- (x * x)$sum_op()
loss$backward()
print(x$grad$as_numeric())  # [2, 4, 6]
```

## Tensor API

### Construction

| Method | Description |
|--------|-------------|
| `Tensor$new(data)` | From numeric vector/matrix/array |
| `tensor_zeros(...)` | Tensor of zeros |
| `tensor_ones(...)` | Tensor of ones |
| `tensor_full(shape, val)` | Filled tensor |
| `tensor_arange(stop, start, step)` | Range tensor |

### Operators

```r
a + b    # Addition
a - b    # Subtraction
a * b    # Multiplication
a / b    # Division
-a       # Negation
```

### Methods

| Method | Description |
|--------|-------------|
| `realize()` | Execute lazy graph |
| `as_numeric()` | Return as R numeric |
| `item()` | Scalar value |
| `add(other)`, `sub(other)`, `mul(other)`, `fdiv(other)` | Arithmetic |
| `neg()` | Negation |
| `exp2()`, `sqrt_op()` | Unary math |
| `reshape_op(shape)` | Reshape |
| `permute_op(order)` | Permute (0-indexed) |
| `flip_op(axes)` | Reverse along axes |
| `pad_op(pairs)` | Pad dimensions |
| `sum_op(axis?)` | Sum reduction |
| `backward()` | Compute gradients |

## Tests

```r
# Run R tests
cd r && Rscript -e "testthat::test_dir('tests')"
```
