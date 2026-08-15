"""nn.state — State dict utilities for polygrad (tinygrad-compatible)."""

import functools
import io
import json
import pickle
import pathlib
import struct
import tarfile
import zipfile
import zlib
from collections import OrderedDict

from ..dtype import dtypes
from ..helpers import DEBUG, Timing, argsort, prod, strides_for_shape, tqdm


# Literal pinned tinygrad/nn/state.py:35-40 safetensors dtype vocabulary.
safe_dtypes = {
    "BOOL": dtypes.bool,
    "I8": dtypes.int8,
    "U8": dtypes.uint8,
    "I16": dtypes.int16,
    "U16": dtypes.uint16,
    "I32": dtypes.int32,
    "U32": dtypes.uint32,
    "I64": dtypes.int64,
    "U64": dtypes.uint64,
    "F16": dtypes.float16,
    "BF16": dtypes.bfloat16,
    "F32": dtypes.float32,
    "F64": dtypes.float64,
}


class TensorIO(io.RawIOBase):
    def __init__(self, tensor):
        from ..dtype import dtypes, to_dtype

        if tensor.ndim != 1 or to_dtype(tensor.dtype) != dtypes.uint8:
            raise ValueError("Tensor must be 1d and of dtype uint8!")
        self._position, self._tensor = 0, tensor

    def readable(self):
        return True

    def read(self, size=-1):
        buf = super().read(size)
        if buf is None:
            raise ValueError("io.RawIOBase.read returned None")
        return buf

    def readinto(self, buffer):
        data = self._tensor[self._position : self._position + len(buffer)].data()
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self._position = min(
            len(self._tensor),
            max(
                0, [offset, self._position + offset, len(self._tensor) + offset][whence]
            ),
        )
        return self._position

    def __enter__(self):
        return self

    def write(self, value):
        raise io.UnsupportedOperation("TensorIO.write not supported")

    def writelines(self, lines):
        raise io.UnsupportedOperation("TensorIO.writelines not supported")


def accept_filename(func):
    @functools.wraps(func)
    def wrapper(filename):
        from ..tensor import Tensor

        return func(
            Tensor(pathlib.Path(filename))
            if not isinstance(filename, Tensor)
            else filename
        )

    return wrapper


@accept_filename
def safe_load_metadata(tensor):
    """Load the safetensors JSON header from a file-backed uint8 Tensor."""
    data_start = int.from_bytes(tensor[0:8].data(), "little") + 8
    return tensor, data_start, json.loads(tensor[8:data_start].data().tobytes())


def safe_load(filename):
    """Load safetensors as lazy slices of the file-backed source Tensor."""
    tensor, data_start, metadata = safe_load_metadata(filename)
    data = tensor[data_start:]
    return {
        key: data[value["data_offsets"][0] : value["data_offsets"][1]]
        .bitcast(safe_dtypes[value["dtype"]])
        .reshape(value["shape"])
        for key, value in metadata.items()
        if key != "__metadata__"
    }


@accept_filename
def zip_extract(tensor):
    """Return zip members as lazy uint8 views, inflating compressed entries."""
    from ..tensor import Tensor

    files = {}
    with zipfile.ZipFile(TensorIO(tensor), "r") as archive:
        # Pinned nn/state.py:164-181 reads the local header because the central
        # directory does not carry the variable extra-field offset.
        header_contents = [
            tensor[info.header_offset + 26 : info.header_offset + 30]
            .bitcast(dtypes.uint16)
            .to("CPU")
            for info in archive.filelist
        ]
        if header_contents:
            Tensor.realize(*header_contents)
        for info, header in zip(archive.filelist, header_contents):
            file_offset = info.header_offset + 30 + sum(header.tolist())
            files[info.filename] = tensor[
                file_offset : file_offset + info.compress_size
            ]
            if info.compress_type == zipfile.ZIP_STORED:
                continue
            if info.compress_type == zipfile.ZIP_DEFLATED:
                raw = zlib.decompress(files[info.filename].data(), -15)
                files[info.filename] = Tensor(list(raw), dtype=dtypes.uint8)
                continue
            raise NotImplementedError(f"compression {info.compress_type} not supported")
    return files


@accept_filename
def tar_extract(tensor):
    """Return regular tar members as lazy tensor views into the archive."""
    with tarfile.open(fileobj=TensorIO(tensor), mode="r") as tar:
        return {
            member.name: tensor[member.offset_data : member.offset_data + member.size]
            for member in tar
            if member.type == tarfile.REGTYPE
        }


@accept_filename
def torch_load(tensor):
    """Load PyTorch zip, tar, or legacy pickle state dictionaries."""
    # Mechanical port of pinned tinygrad/nn/state.py:203-294. Archive payloads
    # remain Tensor views until a non-contiguous stride requires materializing
    # and permuting the value on the default execution device.
    storage_source = {}
    lens = {}

    def _rebuild_tensor(storage, storage_offset, size, stride):
        return _rebuild_tensor_v2(storage, storage_offset, size, stride)

    def _rebuild_tensor_v2(
        storage,
        storage_offset,
        size,
        stride,
        requires_grad=None,
        backward_hooks=None,
        metadata=None,
    ):
        del requires_grad, backward_hooks, metadata
        lens[storage[2]] = storage[4] * storage[1].itemsize
        if storage[2] not in storage_source:
            return None
        byte_start = storage_offset * storage[1].itemsize
        byte_end = (storage_offset + prod(size)) * storage[1].itemsize
        ret = storage_source[storage[2]][byte_start:byte_end].bitcast(storage[1])

        shape_strides = [(sz, st) for sz, st in zip(size, stride) if sz != 1]
        permute_indexes = [
            len(shape_strides) - 1 - index
            for index in argsort([value[1] for value in shape_strides])
        ]
        if tuple(permute_indexes) != tuple(range(len(permute_indexes))):
            intermediate_shape = tuple(
                shape_strides[index][0] for index in argsort(permute_indexes)
            )
            assert tuple(
                shape_strides[index][1] for index in argsort(permute_indexes)
            ) == strides_for_shape(intermediate_shape), "nonpermutable strides"
            if DEBUG >= 3:
                print(
                    "WARNING: this torch load is slow. to permute "
                    f"{intermediate_shape} with {permute_indexes}"
                )
            assert storage[1] != dtypes.bfloat16, "can't permute BF16"
            ret = ret.to(None).reshape(intermediate_shape).permute(permute_indexes)
        return ret.reshape(size)

    class Parameter:
        def __setstate__(self, state):
            self.tensor = state[0]

    deserialized_objects = {}
    intercept = {
        "HalfStorage": dtypes.float16,
        "FloatStorage": dtypes.float32,
        "BFloat16Storage": dtypes.bfloat16,
        "IntStorage": dtypes.int32,
        "BoolStorage": dtypes.bool,
        "LongStorage": dtypes.int64,
        "_rebuild_tensor": _rebuild_tensor,
        "_rebuild_tensor_v2": _rebuild_tensor_v2,
        "FloatTensor": None,
        "Parameter": Parameter,
    }
    whitelist = {"torch", "collections", "numpy", "_codecs"}

    class Dummy:
        pass

    class TorchPickle(pickle.Unpickler):
        def find_class(self, module, name):
            module_root = module.split(".")[0]
            if module_root not in whitelist:
                if DEBUG >= 2:
                    print(f"WARNING: returning Dummy for {module} {name}")
                return Dummy
            return (
                intercept[name]
                if module_root == "torch"
                else super().find_class(module, name)
            )

        def persistent_load(self, persistent_id):
            return deserialized_objects.get(persistent_id, persistent_id)

    fobj = io.BufferedReader(TensorIO(tensor))

    def passthrough_reset(value):
        return fobj.seek(0, 0) or value

    if passthrough_reset(zipfile.is_zipfile(fobj)):
        files = zip_extract(tensor)
        base_name = next(iter(files)).split("/", 1)[0]
        storage_source = {
            filename.split("/")[-1]: data
            for filename, data in files.items()
            if filename.startswith(f"{base_name}/data/")
            and not filename.endswith(".pkl")
        }
        return TorchPickle(
            io.BufferedReader(TensorIO(files[f"{base_name}/data.pkl"]), 1_000_000)
        ).load()

    if passthrough_reset(tarfile.is_tarfile(fobj)):
        files = tar_extract(tensor)
        storage_file = io.BufferedReader(TensorIO(files["storages"]), 1_000_000)
        for _ in range(TorchPickle(storage_file).load()):
            (key, _, storage_type), size = (
                TorchPickle(storage_file).load(),
                struct.unpack("<q", storage_file.read(8))[0],
            )
            byte_offset = storage_file.tell()
            storage_source[key] = files["storages"][
                byte_offset : byte_offset + size * storage_type.itemsize
            ]
            storage_file.seek(size * storage_type.itemsize, 1)
        tensor_file = io.BufferedReader(TensorIO(files["tensors"]), 1_000_000)
        for _ in range(TorchPickle(tensor_file).load()):
            (key, storage_id, _), ndim, _ = (
                TorchPickle(tensor_file).load(),
                struct.unpack("<i", tensor_file.read(4))[0],
                tensor_file.read(4),
            )
            size = struct.unpack(f"<{ndim}q", tensor_file.read(8 * ndim))
            stride = struct.unpack(f"<{ndim}q", tensor_file.read(8 * ndim))
            storage_offset = struct.unpack("<q", tensor_file.read(8))[0]
            deserialized_objects[str(key)] = _rebuild_tensor_v2(
                (None, storage_type, storage_id, None, -1),
                storage_offset,
                size,
                stride,
            )
        pickle_data = TorchPickle(
            io.BufferedReader(TensorIO(files["pickle"]), 1_000_000)
        ).load()
        return {
            key: value.tensor if isinstance(value, Parameter) else value
            for key, value in pickle_data.items()
        }

    pkl = TorchPickle(fobj)
    _, _, _, rewind, _, ids, base_offset = (
        pkl.load(),
        pkl.load(),
        pkl.load(),
        fobj.tell(),
        pkl.load(),
        pkl.load(),
        fobj.tell(),
    )
    for storage_id in ids:
        storage_source[storage_id] = tensor[
            base_offset + 8 : base_offset + 8 + lens[storage_id]
        ]
        base_offset += 8 + lens[storage_id]
    fobj.seek(rewind)
    return TorchPickle(fobj).load()


def get_parameters(obj):
    """Return every Tensor in the state dict; optimizers partition parameters and buffers."""
    return list(get_state_dict(obj).values())


def get_state_dict(obj, prefix=""):
    """Get every named Tensor path, preserving aliases like pinned tinygrad."""
    from ..tensor import Tensor

    if isinstance(obj, Tensor):
        return {prefix.strip("."): obj}
    if hasattr(obj, "_asdict"):
        return get_state_dict(obj._asdict(), prefix)
    if isinstance(obj, OrderedDict):
        return get_state_dict(dict(obj), prefix)
    if hasattr(obj, "__dict__"):
        return get_state_dict(obj.__dict__, prefix)
    state = {}
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            state.update(get_state_dict(value, f"{prefix}{i}."))
    elif isinstance(obj, dict):
        for name, value in obj.items():
            state.update(get_state_dict(value, f"{prefix}{name}."))
    return state


def load_state_dict(model, state_dict, strict=True, verbose=True, consume=False, realize=True):
    """Load a state dict with pinned tinygrad's lazy replacement semantics."""
    # Imported here to avoid binding the package-level counter owner before
    # polygrad.__init__ finishes creating its default context.
    from .. import GlobalCounters

    start_mem_used = GlobalCounters.mem_used
    ret = []
    with Timing(
        "loaded weights in ",
        lambda et_ns: (
            f", {(used := GlobalCounters.mem_used - start_mem_used) / 1e9:.2f} GB "
            f"loaded at {used / et_ns:.2f} GB/s"
        ),
        enabled=verbose,
    ):
        model_state_dict = get_state_dict(model)
        if DEBUG >= 1 and len(state_dict) > len(model_state_dict):
            print(
                "WARNING: unused weights in state_dict",
                sorted(list(state_dict.keys() - model_state_dict.keys())),
            )
        for key, target in (progress := tqdm(
            model_state_dict.items(), disable=None if verbose else True
        )):
            progress.desc = (
                f"ram used: {GlobalCounters.mem_used / 1e9:5.2f} GB, {key:50s}: "
            )
            if key not in state_dict and not strict:
                if DEBUG >= 1:
                    print(f"WARNING: not loading {key}")
                continue
            source = state_dict[key]
            if target.shape != source.shape:
                if {(), (1,)} == {source.shape, target.shape}:
                    source = state_dict[key] = source.reshape(target.shape)
                else:
                    raise ValueError(
                        f"Shape mismatch in layer `{key}`: Expected shape {target.shape}, "
                        f"but found {source.shape} in state dict."
                    )
            if isinstance(target.device, tuple):
                target.replace(
                    source if isinstance(source.device, tuple)
                    else source.shard(target.device, target.uop.axis)
                )
            else:
                target.replace(source.to(target.device))
            if realize:
                target.realize()
            if consume:
                del state_dict[key]
            ret.append(target)
    return ret
