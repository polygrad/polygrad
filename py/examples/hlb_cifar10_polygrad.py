#!/usr/bin/env python3
"""
HLB-shaped CIFAR-10 training smoke for Polygrad.

This uses the same CIFAR-10 binary archive format as tinygrad's
examples/hlb_cifar10.py and keeps the same basic STEPS/BS/EVAL_BS workflow.
MODEL=mlp is the cheap default smoke. MODEL=speedyresnet exercises the HLB
model topology with Polygrad tensor ops; augmentation/JIT/EMA parity is tracked
separately in MEMORY_CIFAR.md.
"""

from __future__ import annotations

import os
import random
import tarfile
import time
from pathlib import Path

import numpy as np

import polygrad
from polygrad import Device, Tensor, Variable, dtypes
from polygrad.nn import Conv2d, Linear, SGD, get_parameters


CIFAR_MEAN = np.array(
    [0.4913997551666284, 0.48215855929893703, 0.4465309133731618],
    dtype=np.float32,
).reshape(1, 3, 1, 1)
CIFAR_STD = np.array(
    [0.24703225141799082, 0.24348516474564, 0.26158783926049628],
    dtype=np.float32,
).reshape(1, 3, 1, 1)


def getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def getenv_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cifar_archive() -> Path:
    default = repo_root() / "references" / "cifar-10-binary.tar.gz"
    return Path(os.environ.get("CIFAR10_BINARY", default)).expanduser().resolve()


def load_cifar10_binary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing CIFAR-10 binary archive: {path}\n"
            "Download https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz "
            "or set CIFAR10_BINARY=/path/to/cifar-10-binary.tar.gz"
        )

    def read_member(tf: tarfile.TarFile, name: str) -> tuple[np.ndarray, np.ndarray]:
        member = f"cifar-10-batches-bin/{name}"
        fh = tf.extractfile(member)
        if fh is None:
            raise FileNotFoundError(f"{member} not found in {path}")
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
        records = raw.reshape(-1, 3073)
        labels = records[:, 0].astype(np.int64)
        images = records[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32)
        return images, labels

    with tarfile.open(path, "r:gz") as tf:
        train_parts = [read_member(tf, f"data_batch_{i}.bin") for i in range(1, 6)]
        test_x, test_y = read_member(tf, "test_batch.bin")

    train_x = np.concatenate([x for x, _ in train_parts], axis=0)
    train_y = np.concatenate([y for _, y in train_parts], axis=0)
    return train_x, train_y, test_x, test_y


def preprocess(x: np.ndarray, flatten: bool) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x = (x - CIFAR_MEAN) / CIFAR_STD
    return x.reshape(x.shape[0], -1) if flatten else x


def one_hot(y: np.ndarray, classes: int = 10) -> np.ndarray:
    out = np.zeros((y.shape[0], classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
    return out


def pad_reflect(x: Tensor, size: int = 2) -> Tensor:
    x = x[..., :, 1:size + 1].flip(-1).cat(
        x,
        x[..., :, -(size + 1):-1].flip(-1),
        dim=-1,
    )
    x = x[..., 1:size + 1, :].flip(-2).cat(
        x,
        x[..., -(size + 1):-1, :].flip(-2),
        dim=-2,
    )
    return x


def make_square_mask(shape, mask_size: int) -> Tensor:
    bs, _, h, w = shape
    low_x = Tensor.randint(bs, low=0, high=w - mask_size).reshape(bs, 1, 1, 1)
    low_y = Tensor.randint(bs, low=0, high=h - mask_size).reshape(bs, 1, 1, 1)
    idx_x = Tensor.arange(w, dtype=dtypes.int32).reshape((1, 1, 1, w))
    idx_y = Tensor.arange(h, dtype=dtypes.int32).reshape((1, 1, h, 1))
    return (
        (idx_x >= low_x)
        * (idx_x < (low_x + mask_size))
        * (idx_y >= low_y)
        * (idx_y < (low_y + mask_size))
    )


def make_random_crop_indices(shape, crop_size: int):
    bs, _, h, w = shape
    low_x = Tensor.randint(bs, low=0, high=w - crop_size).reshape(bs, 1, 1, 1)
    low_y = Tensor.randint(bs, low=0, high=h - crop_size).reshape(bs, 1, 1, 1)
    idx_x = Tensor.arange(crop_size, dtype=dtypes.int32).reshape((1, 1, 1, crop_size))
    idx_y = Tensor.arange(crop_size, dtype=dtypes.int32).reshape((1, 1, crop_size, 1))
    return low_x, low_y, idx_x, idx_y


def random_crop(x: Tensor, crop_size: int = 32) -> Tensor:
    xs, ys, xi, yi = make_random_crop_indices(x.shape, crop_size)
    x = x.gather(-1, (xs + xi).expand(-1, 3, x.shape[2], -1))
    return x.gather(-2, (ys + yi).expand(-1, 3, crop_size, crop_size))


def cutmix(x: Tensor, y: Tensor, order: Tensor, mask_size: int = 3):
    mask = make_square_mask(x.shape, mask_size)
    x_patch, y_patch = x[order], y[order]
    mix_portion = float(mask_size ** 2) / float(x.shape[-2] * x.shape[-1])
    return mask.where(x_patch, x), mix_portion * y_patch + (1.0 - mix_portion) * y


def augmentations(
    x: Tensor,
    y: Tensor,
    *,
    random_crop_enabled: bool,
    random_flip_enabled: bool,
    cutmix_size: int,
):
    perms = Tensor.randperm(x.shape[0], device=x.device)
    if random_crop_enabled:
        x = random_crop(x, crop_size=32)
    if random_flip_enabled:
        x = (Tensor.rand(x.shape[0], 1, 1, 1) < 0.5).where(x.flip(-1), x).contiguous()
    x, y = x[perms], y[perms]
    return x, y, *cutmix(x, y, perms, mask_size=cutmix_size)


def cross_entropy(logits: Tensor, labels: Tensor, label_smoothing: float = 0.0) -> Tensor:
    if label_smoothing:
        labels = labels * (1.0 - label_smoothing) + label_smoothing / labels.shape[1]
    return -(labels * logits.log_softmax(axis=1)).sum(axis=1).mean()


class CIFARMLP:
    def __init__(self, hidden: int):
        self.fc1 = Linear(3072, hidden)
        self.fc2 = Linear(hidden, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], 3072)
        return self.fc2(self.fc1(x).relu())


def whitening_weights(x: np.ndarray, kernel_size: int = 2, limit: int = 0) -> Tensor:
    data = preprocess(x[:limit] if limit > 0 else x, flatten=False)
    h = w = kernel_size
    patches = np.lib.stride_tricks.sliding_window_view(data, window_shape=(h, w), axis=(2, 3))
    n, c, oh, ow, _, _ = patches.shape
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c, h, w)
    flat = patches.reshape(patches.shape[0], c * h * w)
    cov = (flat.T @ flat) / (flat.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov.astype(np.float32), UPLO="U")
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order].T.reshape(c * h * w, c, h, w)
    wgt = vecs / np.sqrt(vals + 1e-2)[:, None, None, None]
    return Tensor(wgt.astype(np.float32)).is_param_(False)


class UnsyncedBatchNorm:
    def __init__(self, channels: int, eps=1e-12, momentum=0.85, affine=True, track_running_stats=False):
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = Tensor.ones(channels, dtype="float32").is_param_(False) if affine else None
        self.bias = Tensor.zeros(channels, dtype="float32") if affine else None
        if self.bias is not None:
            self.bias.requires_grad = True
        self.running_mean = Tensor.zeros(1, channels, dtype="float32").is_param_(False)
        self.running_var = Tensor.ones(1, channels, dtype="float32").is_param_(False)

    def calc_stats(self, x: Tensor):
        if Tensor.training or not self.track_running_stats:
            batch_mean = x.mean(axis=(1, 3, 4))
            y = x - batch_mean.detach().reshape((batch_mean.shape[0], 1, -1, 1, 1))
            batch_var = (y * y).mean(axis=(1, 3, 4))
            return batch_mean, (batch_var + self.eps).rsqrt()
        return self.running_mean, (self.running_var.reshape(self.running_var.shape[0], 1, -1, 1, 1) + self.eps).rsqrt()

    def __call__(self, x: Tensor):
        xr = x.reshape(1, -1, *x.shape[1:]).float()
        mean, invstd = self.calc_stats(xr)
        weight = self.weight.reshape(1, -1) if self.weight is not None else None
        bias = self.bias.reshape(1, -1) if self.bias is not None else None
        return xr.batchnorm(weight, bias, mean, invstd, axis=(0, 2)).reshape(x.shape).cast(x.dtype)


class ConvGroup:
    def __init__(self, channels_in: int, channels_out: int):
        self.conv1 = Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
        self.conv2 = Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.norm1 = UnsyncedBatchNorm(channels_out)
        self.norm2 = UnsyncedBatchNorm(channels_out)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).max_pool2d(2).float()
        x = self.norm1(x).cast("float32").quick_gelu()
        residual = x
        x = self.conv2(x).float()
        x = self.norm2(x).cast("float32").quick_gelu()
        return x + residual


class SpeedyResNet:
    def __init__(self, whitening: Tensor, width_scale: float = 1.0):
        def ch(v: int) -> int:
            return max(1, int(round(v * width_scale)))

        c0, c1, c2, c3 = ch(32), ch(64), ch(256), ch(512)
        self.whitening = whitening
        self.net = [
            Conv2d(12, c0, kernel_size=1, bias=False),
            lambda x: x.quick_gelu(),
            ConvGroup(c0, c1),
            ConvGroup(c1, c2),
            ConvGroup(c2, c3),
            lambda x: x.max((2, 3)),
            Linear(c3, 10, bias=False),
            lambda x: x / 9.0,
        ]

    def __call__(self, x: Tensor, training=True) -> Tensor:
        def forward(z: Tensor) -> Tensor:
            return z.conv2d(self.whitening).pad((1, 0, 0, 1)).sequential(self.net)
        if training:
            return forward(x)
        return (forward(x) + forward(x[..., ::-1])) / 2.0


def batch_stream(
    rng: np.random.Generator,
    x: np.ndarray,
    y: np.ndarray,
    bs: int,
    flatten: bool,
):
    epoch = 0
    while True:
        st = time.perf_counter()
        order = rng.permutation(x.shape[0])
        dt_ms = (time.perf_counter() - st) * 1e3
        print(f"shuffling training dataset in {dt_ms:.2f} ms (epoch={epoch})")
        usable = (x.shape[0] // bs) * bs
        for i in range(0, usable, bs):
            idx = order[i:i + bs]
        yield preprocess(x[idx], flatten), one_hot(y[idx])
        epoch += 1


def prepare_tensor_dataset(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    pad_amount: int,
):
    transform = [
        lambda x: x.float() / 255.0,
        lambda x: x.reshape((-1, 3, 32, 32)) - Tensor(CIFAR_MEAN, device=x.device, dtype=x.dtype).reshape((1, 3, 1, 1)),
        lambda x: x / Tensor(CIFAR_STD, device=x.device, dtype=x.dtype).reshape((1, 3, 1, 1)),
    ]
    x_train = Tensor(train_x.astype(np.float32)).sequential(transform).realize()
    x_test = Tensor(test_x.astype(np.float32)).sequential(transform).realize()
    y_train = Tensor(train_y.astype(np.int32), dtype="int32").one_hot(10)
    y_test = Tensor(test_y.astype(np.int32), dtype="int32").one_hot(10)
    if pad_amount > 0:
        x_train = pad_reflect(x_train, size=pad_amount)
    return x_train, y_train.realize(), x_test, y_test.realize()


def fetch_batches(
    x_in: Tensor,
    y_in: Tensor,
    bs: int,
    *,
    is_train: bool,
    flatten: bool,
    random_crop_enabled: bool,
    random_flip_enabled: bool,
    cutmix_enabled: bool,
    cutmix_steps: int,
    cutmix_size: int,
):
    step, epoch = 0, 0
    while True:
        st = time.perf_counter()
        x, y = x_in, y_in
        if is_train:
            x, y, x_cm, y_cm = augmentations(
                x,
                y,
                random_crop_enabled=random_crop_enabled,
                random_flip_enabled=random_flip_enabled,
                cutmix_size=cutmix_size,
            )
            if cutmix_enabled and step >= cutmix_steps:
                x, y = x_cm, y_cm
        dt_ms = (time.perf_counter() - st) * 1e3
        print(f"shuffling {'training' if is_train else 'test'} dataset in {dt_ms:.2f} ms (epoch={epoch})")

        full_batches = (x.shape[0] // bs) * bs
        vi = Variable("i", 0, full_batches - bs)
        for i in range(0, full_batches, bs):
            step += 1
            vib = vi.bind(i)
            xb, yb = x[vib:vib + bs], y[vib:vib + bs]
            if flatten:
                xb = xb.reshape(xb.shape[0], -1)
            yield xb, yb
        epoch += 1
        if not is_train:
            break


def lr_for_step(step: int, steps: int, max_lr: float, final_ratio: float) -> float:
    if steps <= 1:
        return max_lr
    frac = step / (steps - 1)
    return max_lr * (1.0 - frac * (1.0 - final_ratio))


def stat_delta(before: dict, after: dict, key: str) -> int:
    return int(after.get(key, 0) - before.get(key, 0))


def eval_model(model, x_test: Tensor, y_test: Tensor, bs: int, flatten: bool, limit: int):
    total = min(limit, x_test.shape[0]) if limit > 0 else x_test.shape[0]
    total = (total // bs) * bs
    correct = 0
    losses: list[float] = []
    st = time.perf_counter()
    batches = fetch_batches(
        x_test[:total],
        y_test[:total],
        bs,
        is_train=False,
        flatten=flatten,
        random_crop_enabled=False,
        random_flip_enabled=False,
        cutmix_enabled=False,
        cutmix_steps=0,
        cutmix_size=1,
    )
    with Tensor.train(False):
        for _ in range(0, total, bs):
            x_t, y_t = next(batches)
            logits = model(x_t)
            loss = cross_entropy(logits, y_t)
            logits_np, labels_np, loss_np = Tensor.numpy_many(logits, y_t, loss)
            correct += int((logits_np.argmax(axis=1) == labels_np.argmax(axis=1)).sum())
            losses.append(float(np.asarray(loss_np).reshape(())))
    dt_ms = (time.perf_counter() - st) * 1e3
    val_loss = float(np.mean(losses)) if losses else float("nan")
    return correct, total, val_loss, dt_ms


def configure_device() -> str:
    requested = os.environ.get("DEVICE") or os.environ.get("POLY_DEVICE") or "CPU"
    requested = requested.upper()
    if requested == "AUTO":
        requested = "CUDA" if Device.cuda_available() else "CPU"
    if requested == "CUDA" and not Device.cuda_available():
        print("CUDA requested but unavailable; falling back to CPU")
        requested = "CPU"
    Device.set_default(requested)
    return requested


def main() -> None:
    seed = getenv_int("SEED", 201)
    steps = getenv_int("STEPS", 5)
    bs = getenv_int("BS", 64)
    eval_bs = getenv_int("EVAL_BS", bs)
    eval_limit = getenv_int("EVAL_LIMIT", 1024)
    lr = getenv_float("LR", 0.02)
    final_lr_ratio = getenv_float("FINAL_LR_RATIO", 0.025)
    momentum = getenv_float("MOMENTUM", 0.85)
    label_smoothing = getenv_float("LABEL_SMOOTHING", 0.0)
    requested_model = os.environ.get("MODEL", "mlp").lower()
    hidden = getenv_int("HIDDEN", 128)
    width_scale = getenv_float("WIDTH_SCALE", 1.0)
    whiten_limit = getenv_int("WHITEN_LIMIT", 0)
    train_eval = getenv_bool("TRAIN_EVAL", True)
    pad_amount = getenv_int("PAD_AMOUNT", 2)
    random_crop_enabled = getenv_bool("RANDOM_CROP", True)
    random_flip_enabled = getenv_bool("RANDOM_FLIP", True)
    cutmix_enabled = getenv_bool("CUTMIX", True)
    cutmix_size = getenv_int("CUTMIX_SIZE", 3)
    cutmix_steps = getenv_int("CUTMIX_STEPS", 4992)

    if bs <= 0 or eval_bs <= 0:
        raise ValueError("BS and EVAL_BS must be positive")

    random.seed(seed)
    np.random.seed(seed)
    Tensor.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = configure_device()

    train_x, train_y, test_x, test_y = load_cifar10_binary(cifar_archive())
    if requested_model == "mlp":
        flatten = True
        model = CIFARMLP(hidden)
        input_shape = (bs, 3072)
    elif requested_model in {"speedyresnet", "cnn"}:
        flatten = False
        model = SpeedyResNet(whitening_weights(train_x, limit=whiten_limit), width_scale=width_scale)
        input_shape = (bs, 3, 32, 32)
    else:
        raise ValueError("MODEL must be 'mlp' or 'speedyresnet'")

    x_train, y_train, x_test, y_test = prepare_tensor_dataset(
        train_x,
        train_y,
        test_x,
        test_y,
        pad_amount=pad_amount,
    )

    params = get_parameters(model)
    opt = SGD(params, lr=lr, momentum=momentum)
    batches = fetch_batches(
        x_train,
        y_train,
        bs,
        is_train=True,
        flatten=flatten,
        random_crop_enabled=random_crop_enabled,
        random_flip_enabled=random_flip_enabled,
        cutmix_enabled=cutmix_enabled,
        cutmix_steps=cutmix_steps,
        cutmix_size=cutmix_size,
    )

    print(
        f"polygrad cifar10 model={requested_model} device={device} "
        f"steps={steps} bs={bs} eval_bs={eval_bs}"
    )

    for step in range(steps):
        xb, yb = next(batches)
        opt.lr = lr_for_step(step, steps, lr, final_lr_ratio)
        before = polygrad.stats()
        st = time.perf_counter()
        with Tensor.train(True):
            opt.zero_grad()
            logits = model(xb)
            loss = cross_entropy(logits, yb, label_smoothing=label_smoothing)
            loss.backward()
            opt.step()
            loss_value = float(loss.item())
        elapsed_ms = (time.perf_counter() - st) * 1e3
        after = polygrad.stats()
        launches = stat_delta(before, after, "launch_count")
        schedules = stat_delta(before, after, "schedule_cache_misses")
        print(
            f"{step:3d} {elapsed_ms:8.2f} ms run, "
            f"{loss_value:8.4f} loss, {opt.lr:.6f} LR, "
            f"{launches} launches, {schedules} schedule misses"
        )

    if train_eval:
        correct, total, val_loss, eval_ms = eval_model(model, x_test, y_test, eval_bs, flatten, eval_limit)
        acc = 100.0 * correct / total if total else 0.0
        print(f"eval {correct:8d}/{total:<8d} {acc:5.2f}%, {val_loss:7.4f} val_loss STEPS={steps} (in {eval_ms:.2f} ms)")


if __name__ == "__main__":
    main()
