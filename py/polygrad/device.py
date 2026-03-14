"""Device -- tinygrad-compatible device."""

from . import _ffi


class Device:
    DEFAULT = 'CPU'

    def __class_getitem__(cls, key):
        return cls.canonicalize(key)

    @staticmethod
    def canonicalize(device):
        if device is None:
            return Device.DEFAULT
        dev = str(device).upper()
        if dev not in {'CPU', 'CUDA'}:
            raise ValueError(f'Unsupported device: {device!r}')
        return dev

    @staticmethod
    def set_default(device):
        Device.DEFAULT = Device.canonicalize(device)

    @staticmethod
    def cuda_available():
        """Check if CUDA is available."""
        return _ffi._has_cuda_ffi and _ffi.get_lib().poly_cuda_available()
