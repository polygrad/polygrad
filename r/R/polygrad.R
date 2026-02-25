# polygrad.R — Lazy Tensor class backed by polygrad's C11 compiler core.
# CPU-only, float32-only for v0.

# Module-level default context
.polygrad_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Load libpolygrad.so
  polygrad_lib <- Sys.getenv("POLYGRAD_LIB", "")
  if (polygrad_lib == "") {
    polygrad_lib <- file.path(system.file(package = pkgname), "..", "..", "..", "build", "libpolygrad.so")
    polygrad_lib <- normalizePath(polygrad_lib, mustWork = FALSE)
  }
  if (file.exists(polygrad_lib)) {
    dyn.load(polygrad_lib)
  }

  # Build op name -> int mapping
  n_ops <- .Call(C_poly_op_count)
  ops <- list()
  for (i in seq_len(n_ops)) {
    name <- .Call(C_poly_op_name, as.integer(i - 1L))
    if (!is.null(name)) ops[[name]] <- as.integer(i - 1L)
  }
  .polygrad_env$OPS <- ops

  # Create default context
  .polygrad_env$ctx <- .Call(C_poly_ctx_new)
}

.onUnload <- function(libpath) {
  if (!is.null(.polygrad_env$ctx)) {
    .Call(C_poly_ctx_destroy, .polygrad_env$ctx)
    .polygrad_env$ctx <- NULL
  }
}

# ── Tensor class ─────────────────────────────────────────────────────

Tensor <- setRefClass("Tensor",
  fields = list(
    ctx    = "ANY",
    uop    = "ANY",
    buffer = "ANY",
    data   = "ANY",
    shape  = "integer",
    inputs = "list",
    requires_grad = "logical",
    grad   = "ANY"
  ),
  methods = list(
    initialize = function(data_in = NULL, requires_grad_in = FALSE,
                          ctx_in = NULL, uop_in = NULL, buffer_in = NULL,
                          data_raw = NULL, shape_in = NULL, inputs_in = NULL) {
      ctx    <<- if (!is.null(ctx_in)) ctx_in else .polygrad_env$ctx
      requires_grad <<- requires_grad_in
      grad   <<- NULL

      if (!is.null(uop_in)) {
        # Internal construction from ops
        uop    <<- uop_in
        buffer <<- buffer_in
        data   <<- data_raw
        shape  <<- as.integer(shape_in)
        inputs <<- if (!is.null(inputs_in)) inputs_in else list()
      } else {
        # User construction from data
        arr <- as.numeric(data_in)
        if (is.matrix(data_in)) {
          shp <- dim(data_in)
        } else if (is.array(data_in)) {
          shp <- dim(data_in)
        } else {
          shp <- length(arr)
        }
        shape  <<- as.integer(shp)
        data   <<- arr
        buffer <<- .Call(C_poly_buffer_f32, ctx, as.numeric(length(arr)))

        # For multi-dimensional, insert RESHAPE
        if (length(shape) > 1) {
          uop <<- .Call(C_poly_reshape, ctx, buffer, as.numeric(shape))
        } else {
          uop <<- buffer
        }
        inputs <<- list()
      }
    },

    is_leaf = function() {
      !is.null(data) && !is.null(buffer)
    },

    collect_leaves = function(seen = NULL) {
      if (is.null(seen)) seen <- new.env(parent = emptyenv())
      # Use the R external pointer address as a string key for dedup
      id <- capture.output(print(.self$uop))
      if (exists(id, envir = seen)) return(list())
      assign(id, TRUE, envir = seen)

      if (is_leaf()) return(list(.self))

      leaves <- list()
      for (inp in inputs) {
        leaves <- c(leaves, inp$collect_leaves(seen))
      }
      leaves
    },

    realize = function() {
      if (is_leaf()) return(invisible(NULL))

      leaves <- collect_leaves()
      numel <- as.integer(prod(shape))

      out_buf <- .Call(C_poly_buffer_f32, ctx, as.numeric(numel))
      store <- .Call(C_poly_store_val, ctx, out_buf, uop)
      sink <- .Call(C_poly_sink1, ctx, store)

      # Build binding lists
      in_bufs <- lapply(leaves, function(l) l$buffer)
      in_data <- lapply(leaves, function(l) l$data)

      result <- .Call(C_poly_realize_read, ctx, sink,
                      in_bufs, in_data, out_buf, numel)

      data   <<- result
      buffer <<- out_buf
      uop    <<- out_buf
      inputs <<- list()
    },

    as_numeric = function() {
      realize()
      if (length(shape) > 1) {
        array(data, dim = shape)
      } else {
        data
      }
    },

    item = function() {
      arr <- as_numeric()
      if (length(arr) != 1) stop("item() requires scalar tensor")
      as.numeric(arr[1])
    },

    # ── Internal helpers ──

    make_result = function(new_uop, new_shape, new_inputs) {
      rg <- any(sapply(new_inputs, function(t) t$requires_grad))
      Tensor$new(uop_in = new_uop, shape_in = new_shape,
                 inputs_in = new_inputs, ctx_in = ctx,
                 requires_grad_in = rg)
    },

    ensure_tensor = function(other) {
      if (is(other, "Tensor")) return(other)
      if (is.numeric(other) && length(other) == 1) {
        c_uop <- .Call(C_poly_const_float, ctx, as.numeric(other))
        return(Tensor$new(uop_in = c_uop, shape_in = integer(0),
                          inputs_in = list(), ctx_in = ctx))
      }
      stop("Cannot convert to Tensor")
    },

    broadcast_shape = function(other_shape) {
      a <- shape; b <- other_shape
      if (length(a) == 0) return(b)
      if (length(b) == 0) return(a)
      ndim <- max(length(a), length(b))
      a <- c(rep(1L, ndim - length(a)), a)
      b <- c(rep(1L, ndim - length(b)), b)
      res <- integer(ndim)
      for (i in seq_len(ndim)) {
        if (a[i] == b[i]) { res[i] <- a[i] }
        else if (a[i] == 1L) { res[i] <- b[i] }
        else if (b[i] == 1L) { res[i] <- a[i] }
        else stop(paste("Cannot broadcast shapes"))
      }
      res
    },

    # ── Element-wise ops ──

    add = function(other) {
      other <- ensure_tensor(other)
      u <- .Call(C_poly_alu2, ctx, .polygrad_env$OPS$ADD, uop, other$uop)
      make_result(u, broadcast_shape(other$shape), list(.self, other))
    },

    sub = function(other) {
      other <- ensure_tensor(other)
      u <- .Call(C_poly_alu2, ctx, .polygrad_env$OPS$SUB, uop, other$uop)
      make_result(u, broadcast_shape(other$shape), list(.self, other))
    },

    mul = function(other) {
      other <- ensure_tensor(other)
      u <- .Call(C_poly_alu2, ctx, .polygrad_env$OPS$MUL, uop, other$uop)
      make_result(u, broadcast_shape(other$shape), list(.self, other))
    },

    fdiv = function(other) {
      other <- ensure_tensor(other)
      u <- .Call(C_poly_alu2, ctx, .polygrad_env$OPS$FDIV, uop, other$uop)
      make_result(u, broadcast_shape(other$shape), list(.self, other))
    },

    neg = function() {
      u <- .Call(C_poly_alu1, ctx, .polygrad_env$OPS$NEG, uop)
      make_result(u, shape, list(.self))
    },

    exp2 = function() {
      u <- .Call(C_poly_alu1, ctx, .polygrad_env$OPS$EXP2, uop)
      make_result(u, shape, list(.self))
    },

    sqrt_op = function() {
      u <- .Call(C_poly_alu1, ctx, .polygrad_env$OPS$SQRT, uop)
      make_result(u, shape, list(.self))
    },

    # ── Movement ops ──

    reshape_op = function(new_shape) {
      u <- .Call(C_poly_reshape, ctx, uop, as.numeric(new_shape))
      make_result(u, as.integer(new_shape), list(.self))
    },

    permute_op = function(order) {
      u <- .Call(C_poly_permute, ctx, uop, as.numeric(order))
      new_shape <- shape[order + 1L]  # R is 1-indexed but perm is 0-indexed
      make_result(u, new_shape, list(.self))
    },

    flip_op = function(axes) {
      u <- .Call(C_poly_flip, ctx, uop, as.numeric(axes))
      make_result(u, shape, list(.self))
    },

    pad_op = function(pairs) {
      # pairs: list of c(before, after) per dimension
      flat <- as.numeric(unlist(pairs))
      ndim <- length(pairs)
      u <- .Call(C_poly_pad, ctx, uop, flat, as.integer(ndim))
      new_shape <- shape + sapply(pairs, sum)
      make_result(u, as.integer(new_shape), list(.self))
    },

    # ── Reduction ──

    sum_op = function(axis = NULL) {
      if (is.null(axis)) axis <- seq_len(length(shape)) - 1L
      u <- .Call(C_poly_reduce_axis, ctx, .polygrad_env$OPS$ADD, uop, as.numeric(axis))
      axis_set <- axis
      new_shape <- integer(0)
      for (i in seq_along(shape)) {
        if (!((i - 1L) %in% axis_set)) {
          new_shape <- c(new_shape, shape[i])
        }
      }
      make_result(u, new_shape, list(.self))
    },

    # ── Autograd ──

    backward = function() {
      all_leaves <- collect_leaves()
      grad_leaves <- Filter(function(l) l$requires_grad, all_leaves)
      if (length(grad_leaves) == 0) stop("No leaf tensors require grad")

      for (leaf in grad_leaves) {
        grad_uop <- .Call(C_poly_grad, ctx, uop, leaf$buffer)
        if (is.null(grad_uop)) stop("poly_grad returned NULL")

        numel <- as.integer(prod(leaf$shape))
        grad_buf <- .Call(C_poly_buffer_f32, ctx, as.numeric(numel))
        store <- .Call(C_poly_store_val, ctx, grad_buf, grad_uop)
        sink <- .Call(C_poly_sink1, ctx, store)

        in_bufs <- lapply(all_leaves, function(l) l$buffer)
        in_data <- lapply(all_leaves, function(l) l$data)

        result <- .Call(C_poly_realize_read, ctx, sink,
                        in_bufs, in_data, grad_buf, numel)

        leaf$grad <- Tensor$new(result, ctx_in = ctx)
      }
    },

    # ── Stubs ──
    matmul = function(other) stop("matmul not implemented"),
    to = function(device) stop("device transfer not implemented"),
    softmax = function() stop("softmax not implemented"),

    show = function() {
      cat(sprintf("Tensor(shape=(%s), dtype=float32, realized=%s)\n",
                  paste(shape, collapse = ","), is_leaf()))
    }
  )
)

# ── Operator overloads ───────────────────────────────────────────────

setMethod("+", signature("Tensor", "ANY"), function(e1, e2) e1$add(e2))
setMethod("+", signature("ANY", "Tensor"), function(e1, e2) e2$add(e1))
setMethod("-", signature("Tensor", "ANY"), function(e1, e2) e1$sub(e2))
setMethod("*", signature("Tensor", "ANY"), function(e1, e2) e1$mul(e2))
setMethod("*", signature("ANY", "Tensor"), function(e1, e2) e2$mul(e1))
setMethod("/", signature("Tensor", "ANY"), function(e1, e2) e1$fdiv(e2))
setMethod("-", signature("Tensor", "missing"), function(e1, e2) e1$neg())

# ── Static constructors ─────────────────────────────────────────────

tensor_zeros <- function(...) {
  shape <- c(...)
  Tensor$new(rep(0, prod(shape)))
}

tensor_ones <- function(...) {
  shape <- c(...)
  Tensor$new(rep(1, prod(shape)))
}

tensor_full <- function(shape, fill_value) {
  if (length(shape) == 1) shape <- c(shape)
  Tensor$new(rep(fill_value, prod(shape)))
}

tensor_arange <- function(stop, start = 0, step = 1) {
  Tensor$new(seq(start, stop - step, by = step))
}
