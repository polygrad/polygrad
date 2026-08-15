"""Tinygrad-compatible engine package paths backed by Polygrad's C runtime."""

from .jit import JitError, TinyJit

__all__ = ["JitError", "TinyJit"]
