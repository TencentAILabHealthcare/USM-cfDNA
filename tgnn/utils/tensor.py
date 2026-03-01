# Copyright (c) 2024, Tencent Inc. All rights reserved.

from collections.abc import Mapping
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from .type import is_tensor

def flatten_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


def to_size(batch_x, target_size, mode='nearest'):
    if isinstance(batch_x, (list, tuple)):
        new_tensors = []
        for t in batch_x:
            new_tensors.append(to_size(t, target_size, mode))

        return new_tensors
    elif isinstance(batch_x, dict):
        new_dict = {}
        for name, value in batch_x.items():
            new_dict[name] = to_size(value, target_size, mode)

        return new_dict
    elif isinstance(batch_x, torch.Tensor):
        batch_x = F.interpolate(batch_x, target_size, mode=mode)

        return batch_x
    else:
        # TODO: add numpy array resize
        return batch_x


def clone(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(clone(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = clone(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.clone()
    else:
        return np.copy(tensor)


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


tensor_dict_map = partial(dict_map, leaf_type=torch.Tensor)


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Not supported type: {type(tree)}")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def masked_mean(mask, value, dim=None, keepdim=False, eps=1e-8):
    if mask is None:
        return value.mean(dim=dim, keepdim=keepdim)
    mask = mask.expand_as(value)
    x = torch.sum(mask * value, dim=dim, keepdim=keepdim) / (eps + torch.sum(mask, dim=dim, keepdim=keepdim))
    # avoid dtype promotion
    if x.dtype != value.dtype:
        x = x.to(value.dtype)
    return x


def collate_dense_tensors(
        samples: List[torch.Tensor],
        pad_v: float = 0,
        max_shape: Tuple = None
) -> torch.Tensor:
    """collate batch tensor
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()

    # assert all tensor have same dim
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )

    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    if max_shape is None:
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]

    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]  # get view of tensor[i], not copy a new tenor
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t

    return result


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad, )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))



def log(x):
    if is_tensor(x):
        return torch.log(x)
    else:
        return np.log(x)


def log10(x):
    if is_tensor(x):
        return torch.log10(x)
    else:
        return np.log10(x)


def amin(x, dim=None, keepdim=False):
    if is_tensor(x):
        return torch.amin(x, dim=dim, keepdim=keepdim)
    else:
        return np.amin(x, axis=dim, keepdims=keepdim)


def clip(x, min=None, max=None):
    if is_tensor(x):
        return torch.clip(x, min, max)
    else:
        return np.clip(x, min, max)


def sort(x, dim=None, descending=False):
    if is_tensor(x):
        return torch.sort(x, dim=dim, descending=descending)[0]
    else:
        out = np.sort(-x if descending else x, axis=dim)
        return -out if descending else out


def maximum(x1, x2):
    assert type(x1) == type(x2)
    if is_tensor(x1):
        return torch.maximum(x1, x2)
    else:
        return np.maximum(x1, x2)