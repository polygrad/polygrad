from ..tensor import Tensor
from .state import tar_extract


def cifar(device=None):
    tensors = tar_extract(Tensor.from_url(
        "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        gunzip=True,
    ))
    train = Tensor.cat(*[
        tensors[f"cifar-10-batches-bin/data_batch_{index}.bin"]
        .reshape(-1, 3073).to(device)
        for index in range(1, 6)
    ])
    test = tensors["cifar-10-batches-bin/test_batch.bin"].reshape(-1, 3073).to(device)
    return (
        train[:, 1:].reshape(-1, 3, 32, 32),
        train[:, 0],
        test[:, 1:].reshape(-1, 3, 32, 32),
        test[:, 0],
    )


__all__ = ["cifar"]
