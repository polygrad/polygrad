"""Device — tinygrad-compatible device."""

from . import _ffi


class Device:
    DEFAULT = "CPU"

    @staticmethod
    def cuda_available():
        """Check if CUDA is available."""
        return _ffi._has_cuda_ffi and _ffi._lib.poly_cuda_available()
