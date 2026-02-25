"""dtypes — tinygrad-compatible dtype constants."""


class dtypes:
    float32 = 'float32'
    float16 = 'float16'
    int32 = 'int32'
    int64 = 'int64'
    int8 = 'int8'
    uint8 = 'uint8'
    bool = 'bool'
    float = float32
    half = float16
    default_float = float32

    @staticmethod
    def is_float(d):
        return d in ('float32', 'float16')

    @staticmethod
    def is_int(d):
        return d in ('int8', 'int32', 'int64', 'uint8')
