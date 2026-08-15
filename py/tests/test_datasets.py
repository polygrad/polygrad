"""Parity tests for tinygrad's generic DISK/TensorIO/tar/CIFAR data path."""

import io
import tarfile

import pytest

from polygrad import Tensor, dtypes, nn
from polygrad.dtype import to_dtype
from polygrad.nn.state import TensorIO, tar_extract


CIFAR_PREFIX = "cifar-10-batches-bin/"
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
FASHION_MNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"


def _tar_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        directory = tarfile.TarInfo(CIFAR_PREFIX)
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _cifar_record(label, base):
    return bytes((label,)) + b"".join(
        bytes((base + channel,)) * 1024 for channel in range(3)
    )


def _cifar_entries():
    return [
        (CIFAR_PREFIX + "test_batch.bin", _cifar_record(8, 80) + _cifar_record(9, 90)),
        (CIFAR_PREFIX + "data_batch_5.bin", _cifar_record(5, 50)),
        (CIFAR_PREFIX + "ignored.bin", b"ignored regular member"),
        (CIFAR_PREFIX + "data_batch_2.bin", _cifar_record(2, 20)),
        (CIFAR_PREFIX + "data_batch_4.bin", _cifar_record(4, 40)),
        (CIFAR_PREFIX + "data_batch_1.bin", _cifar_record(1, 10)),
        (CIFAR_PREFIX + "data_batch_3.bin", _cifar_record(3, 30)),
    ]


def _write_tar(tmp_path, entries=None):
    path = tmp_path / "cifar.tar"
    path.write_bytes(_tar_bytes(_cifar_entries() if entries is None else entries))
    return path


def test_path_tensor_and_data_return_independent_copies(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(bytes((1, 2, 3, 4)))

    disk = Tensor(path)
    assert disk.shape == (4,)
    assert to_dtype(disk.dtype) == dtypes.uint8
    assert disk.device == f"DISK:{path.resolve()}"
    disk_data = disk.data()
    disk_data[0] = 9
    assert list(disk_data) == [9, 2, 3, 4]
    assert disk.tolist() == [1, 2, 3, 4]
    assert list(path.read_bytes()) == [1, 2, 3, 4]

    cpu = Tensor([1, 2, 3], dtype=dtypes.uint8)
    cpu_data = cpu.data()
    cpu_data[0] = 9
    assert list(cpu_data) == [9, 2, 3]
    assert cpu.tolist() == [1, 2, 3]

    with pytest.raises(FileNotFoundError):
        Tensor(tmp_path / "missing.bin")


def test_tensorio_read_seek_and_validation():
    stream = TensorIO(Tensor([10, 20, 30, 40], dtype=dtypes.uint8))
    assert stream.readable() and stream.seekable()
    assert stream.read(2) == bytes((10, 20))
    assert stream.seek(-1, io.SEEK_CUR) == 1
    assert stream.read(2) == bytes((20, 30))
    assert stream.seek(-1, io.SEEK_END) == 3
    assert stream.read() == bytes((40,))
    assert stream.seek(99) == 4
    assert stream.read(1) == b""

    with pytest.raises(ValueError, match="1d and of dtype uint8"):
        TensorIO(Tensor([[1, 2]], dtype=dtypes.uint8))
    with pytest.raises(ValueError, match="1d and of dtype uint8"):
        TensorIO(Tensor([1, 2], dtype=dtypes.int32))
    with pytest.raises(io.UnsupportedOperation, match="write not supported"):
        stream.write(b"x")


def test_tar_extract_returns_only_regular_lazy_member_views(tmp_path):
    entries = [("first.bin", b"abc"), ("nested/second.bin", bytes((1, 2, 3, 4)))]
    path = _write_tar(tmp_path, entries)
    members = tar_extract(path)

    assert set(members) == {"first.bin", "nested/second.bin"}
    assert all(tensor.device == f"DISK:{path.resolve()}" for tensor in members.values())
    assert members["first.bin"].tolist() == [97, 98, 99]
    assert members["nested/second.bin"].tolist() == [1, 2, 3, 4]


def test_mnist_matches_pinned_idx_headers_urls_and_device(monkeypatch):
    calls = []

    def fake_from_url(url, **kwargs):
        calls.append((url, kwargs))
        if "images" in url:
            return Tensor(
                list(range(16)) + [i % 256 for i in range(2 * 28 * 28)],
                dtype=dtypes.uint8,
            )
        return Tensor(list(range(8)) + [3, 7], dtype=dtypes.uint8)

    monkeypatch.setattr(Tensor, "from_url", staticmethod(fake_from_url))
    x_train, y_train, x_test, y_test = nn.datasets.mnist(
        device="INTERP", fashion=True,
    )

    assert calls == [
        (FASHION_MNIST_URL + "train-images-idx3-ubyte.gz", {"gunzip": True}),
        (FASHION_MNIST_URL + "train-labels-idx1-ubyte.gz", {"gunzip": True}),
        (FASHION_MNIST_URL + "t10k-images-idx3-ubyte.gz", {"gunzip": True}),
        (FASHION_MNIST_URL + "t10k-labels-idx1-ubyte.gz", {"gunzip": True}),
    ]
    assert (x_train.shape, y_train.shape, x_test.shape, y_test.shape) == (
        (2, 1, 28, 28), (2,), (2, 1, 28, 28), (2,),
    )
    assert all(
        to_dtype(tensor.dtype) == dtypes.uint8
        for tensor in (x_train, y_train, x_test, y_test)
    )
    assert all(
        tensor.device == "INTERP"
        for tensor in (x_train, y_train, x_test, y_test)
    )
    assert x_train.flatten()[:4].tolist() == [0, 1, 2, 3]
    assert y_train.tolist() == [3, 7]
    assert x_test.flatten()[:4].tolist() == [0, 1, 2, 3]
    assert y_test.tolist() == [3, 7]


def test_cifar_matches_pinned_member_order_shapes_values_and_device(tmp_path, monkeypatch):
    path = _write_tar(tmp_path)
    from_url_calls, to_calls = [], []
    original_to = Tensor.to

    def fake_from_url(url, gunzip=False, **kwargs):
        from_url_calls.append((url, gunzip, kwargs))
        return Tensor(path)

    def tracked_to(self, device, *args, **kwargs):
        to_calls.append((self.shape, device))
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(Tensor, "from_url", staticmethod(fake_from_url))
    monkeypatch.setattr(Tensor, "to", tracked_to)
    x_train, y_train, x_test, y_test = nn.datasets.cifar(device="INTERP")

    assert from_url_calls == [(CIFAR_URL, True, {})]
    assert to_calls == [((1, 3073), "INTERP") for _ in range(5)] + [
        ((2, 3073), "INTERP")
    ]
    assert (x_train.shape, y_train.shape, x_test.shape, y_test.shape) == (
        (5, 3, 32, 32), (5,), (2, 3, 32, 32), (2,),
    )
    assert all(
        to_dtype(tensor.dtype) == dtypes.uint8
        for tensor in (x_train, y_train, x_test, y_test)
    )
    assert all(tensor.device == "INTERP" for tensor in (x_train, y_train, x_test, y_test))
    assert y_train.tolist() == [1, 2, 3, 4, 5]
    assert x_train[:, :, 0, 0].tolist() == [
        [10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42], [50, 51, 52],
    ]
    assert x_train[:, :, 31, 31].tolist() == x_train[:, :, 0, 0].tolist()
    assert y_test.tolist() == [8, 9]
    assert x_test[:, :, 0, 0].tolist() == [[80, 81, 82], [90, 91, 92]]


def test_cifar_rejects_missing_or_partial_records(tmp_path, monkeypatch):
    payload = [_tar_bytes(_cifar_entries())]

    def fake_from_url(url, gunzip=False, **kwargs):
        path = tmp_path / "current.tar"
        path.write_bytes(payload[0])
        return Tensor(path)

    monkeypatch.setattr(Tensor, "from_url", staticmethod(fake_from_url))

    missing = [(name, data) for name, data in _cifar_entries() if not name.endswith("data_batch_3.bin")]
    payload[0] = _tar_bytes(missing)
    with pytest.raises(KeyError, match="data_batch_3.bin"):
        nn.datasets.cifar()

    partial = [
        (name, data[:-1] if name.endswith("data_batch_3.bin") else data)
        for name, data in _cifar_entries()
    ]
    payload[0] = _tar_bytes(partial)
    with pytest.raises(ValueError, match="size mismatch"):
        nn.datasets.cifar()
