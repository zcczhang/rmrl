from typing import List, Optional
from typing import Union

import numpy as np
import torch
import tree

__all__ = [
    "partition_inds",
    "any_stack",
    "torch_dtype",
    "torch_device",
    "torch_dtype_size",
    "any_to_torch_tensor",
    "any_to_numpy",
    "any_to_primitive",
]


def partition_inds(n: int, num_parts: int):
    return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(np.int32)


def any_stack(xs: List, *, dim: int = 0):
    """Works for both torch Tensor and numpy array."""

    def _any_stack_helper(*xs):
        x = xs[0]
        if isinstance(x, np.ndarray):
            return np.stack(xs, axis=dim)
        elif torch.is_tensor(x):
            return torch.stack(xs, dim=dim)
        elif isinstance(x, float):
            # special treatment for float, defaults to float32
            return np.array(xs, dtype=np.float32)
        else:
            return np.array(xs)

    return tree.map_structure(_any_stack_helper, *xs)


# ==== convert utils ====

_TORCH_DTYPE_TABLE = {
    torch.bool: 1,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.short: 2,
    torch.int32: 4,
    torch.int: 4,
    torch.int64: 8,
    torch.long: 8,
    torch.float16: 2,
    torch.half: 2,
    torch.float32: 4,
    torch.float: 4,
    torch.float64: 8,
    torch.double: 8,
}


def torch_dtype(dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        try:
            dtype = getattr(torch, dtype)
        except AttributeError:
            raise ValueError(f'"{dtype}" is not a valid torch dtype')
        assert isinstance(
            dtype, torch.dtype
        ), f"dtype {dtype} is not a valid torch tensor type"
        return dtype
    else:
        raise NotImplementedError(f"{dtype} not supported")


def torch_device(device: Union[str, int, None]) -> Optional[torch.device]:
    """
    Args:
        device:
            - "auto": use current torch context device, same as `.to('cuda')`
            - int: negative for CPU, otherwise GPU index
    """
    if device is None:
        return None
    elif device == "auto":
        return torch.device("cuda")
    elif isinstance(device, int) and device < 0:
        return torch.device("cpu")
    else:
        return torch.device(device)


def torch_dtype_size(dtype: Union[str, torch.dtype]) -> int:
    return _TORCH_DTYPE_TABLE[torch_dtype(dtype)]


def _convert_then_transfer(x, dtype, device, copy, non_blocking):
    x = x.to(dtype=dtype, copy=copy, non_blocking=non_blocking)
    return x.to(device=device, copy=False, non_blocking=non_blocking)


def _transfer_then_convert(x, dtype, device, copy, non_blocking):
    x = x.to(device=device, copy=copy, non_blocking=non_blocking)
    return x.to(dtype=dtype, copy=False, non_blocking=non_blocking)


def any_to_torch_tensor(
    x,
    dtype: Union[str, torch.dtype, None] = None,
    device: Union[str, int, torch.device, None] = None,
    copy=False,
    non_blocking=False,
    smart_optimize: bool = True,
):
    dtype = torch_dtype(dtype)
    device = torch_device(device)

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        # x is a primitive python sequence
        x = torch.tensor(x, dtype=dtype)
        copy = False

    # This step does not create any copy.
    # If x is a numpy array, simply wraps it in Tensor. If it's already a Tensor, do nothing.
    x = torch.as_tensor(x)
    # avoid passing None to .to(), PyTorch 1.4 bug
    dtype = dtype or x.dtype
    device = device or x.device

    if not smart_optimize:
        # do a single stage type conversion and transfer
        return x.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking)

    # we have two choices: (1) convert dtype and then transfer to GPU
    # (2) transfer to GPU and then convert dtype
    # because CPU-to-GPU memory transfer is the bottleneck, we will reduce it as
    # much as possible by sending the smaller dtype

    src_dtype_size = torch_dtype_size(x.dtype)

    # destination dtype size
    if dtype is None:
        dest_dtype_size = src_dtype_size
    else:
        dest_dtype_size = torch_dtype_size(dtype)

    if x.dtype != dtype or x.device != device:
        # a copy will always be performed, no need to force copy again
        copy = False

    if src_dtype_size > dest_dtype_size:
        # better to do conversion on one device (e.g. CPU) and then transfer to another
        return _convert_then_transfer(x, dtype, device, copy, non_blocking)
    elif src_dtype_size == dest_dtype_size:
        # when equal, we prefer to do the conversion on whichever device that's GPU
        if x.device.type == "cuda":
            return _convert_then_transfer(x, dtype, device, copy, non_blocking)
        else:
            return _transfer_then_convert(x, dtype, device, copy, non_blocking)
    else:
        # better to transfer data across device first, and then do conversion
        return _transfer_then_convert(x, dtype, device, copy, non_blocking)


def any_to_numpy(
    x,
    dtype: Union[str, np.dtype, None] = None,
    copy: bool = False,
    non_blocking: bool = False,
    smart_optimize: bool = True,
):
    if isinstance(x, torch.Tensor):
        x = any_to_torch_tensor(
            x,
            dtype=dtype,
            device="cpu",
            copy=copy,
            non_blocking=non_blocking,
            smart_optimize=smart_optimize,
        )
        return x.detach().numpy()
    else:
        # primitive python sequence or ndarray
        return np.array(x, dtype=dtype, copy=copy)


def any_to_primitive(x):
    if isinstance(x, (np.ndarray, np.number, torch.Tensor)):
        return x.tolist()
    else:
        return x
