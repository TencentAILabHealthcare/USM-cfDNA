# Copyright (c) 2024, Tencent Inc. All rights reserved.

import gzip
import io
import json
import logging
import mimetypes
import os
import shutil
import warnings
from functools import partial, wraps
from shutil import which
from typing import Any, Callable

import numpy as np
import torch


def open_file(filename, mode="r"):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode=mode)
    else:
        return open(filename, mode=mode)


def to(tensor,
       device=None,
       dtype=None,
       non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to(t,
                                  device=device,
                                  dtype=dtype,
                                  non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to(value,
                                device=device,
                                dtype=dtype,
                                non_blocking=non_blocking)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device=device,
                         dtype=dtype,
                         non_blocking=non_blocking)
    elif isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, dtype=dtype, device=device)
    else:
        return tensor


def to_device(tensor, device, non_blocking=False):
    return to(tensor, device, non_blocking=non_blocking)


def to_cuda(tensor, non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cuda(t, non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cuda(value, non_blocking=non_blocking)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cuda(non_blocking=non_blocking)
    else:
        return tensor


def to_cpu(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cpu(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cpu(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    else:
        return tensor


def to_numpy(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_numpy(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_numpy(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor


def to_list(tensor):
    if isinstance(tensor, (torch.Tensor, np.ndarray)):
        return tensor.tolist()

    return tensor


def record_stream(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(record_stream(t))
        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = record_stream(value)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.record_stream(torch.cuda.current_stream())
    else:
        return tensor


def rank_0_only(fn: Callable) -> Callable:
    """Call a function only on rank 0 in distributed settings.

    Meant to be used as an decorator.

    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                return fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)

        return None

    return wrapped_fn


print_rank_0 = rank_0_only(partial(print, flush=True))
debug_rank_0 = rank_0_only(logging.debug)
info_rank_0 = rank_0_only(logging.info)
warn_rank_0 = rank_0_only(warnings.warn)


def cat_files(files, output, end="\n"):
    files = list(files)
    n_files = len(files)
    if n_files == 0:
        return

    is_fio = isinstance(output, io.IOBase)
    if not is_fio:
        out = open(output, mode="wb")
    else:
        out = output

    for i, path in enumerate(files):
        with open(path, mode="rb") as f:
            shutil.copyfileobj(f, out)
            if i < n_files - 1 and end:
                out.write(end.encode())

    if not is_fio:
        out.close()


def jloads(jline, mode="r"):
    if os.path.isfile(jline):
        data = []
        with open(jline, mode=mode, encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    return json.loads(jline)


def jload(f, mode="r", object_pairs_hook=None):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        if f.endswith(".json.gz"):
            f = gzip.open(f, mode="rt")
        else:
            f = open(f, mode=mode)
    jdict = json.load(f, object_pairs_hook=object_pairs_hook)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A file handle or string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    is_fio = isinstance(f, io.IOBase)
    if not is_fio:
        folder = os.path.dirname(f)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        f = open(f, mode=mode)

    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)

    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")

    if not is_fio:
        f.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_cache_dir():
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', "tgnn")
    os.makedirs(cache_dir, exist_ok=True)

    return cache_dir


def unzip(filename):
    with gzip.GzipFile(filename) as gz:
        return gz.read()


def is_plain_text(fn):
    return mimetypes.guess_type(fn)[0] == 'text/plain'


def is_tool(name):
    return which(name) is not None


def get_file_size(filename):
    assert os.path.exists(filename), f"not exist file: {filename}"
    return os.path.getsize(filename)


def set_file_timestamp(filename, access_time, modif_time=None):
    modif_time = access_time if modif_time is None else modif_time
    os.utime(filename, (access_time, modif_time))


def get_file_timestamp(filename):
    return os.path.getmtime(filename)
