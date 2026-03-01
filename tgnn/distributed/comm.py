# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import os
import socket

import numpy as np
import torch
import torch.distributed as dist

from . import parallel_state


def cpu_count():
    return torch.multiprocessing.cpu_count()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False

    return True


def is_main_tenser_rank():
    if not is_dist_avail_and_initialized():
        return True

    return parallel_state.get_tensor_model_parallel_rank() == 0


def is_main_data_rank():
    if not is_dist_avail_and_initialized():
        return True

    return parallel_state.get_data_parallel_rank() == 0


def get_world_size(group=None) -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size(group=group)

def get_sequence_parallel_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return parallel_state.get_sequence_parallel_world_size()


def get_sequence_parallel_group():
    if not is_dist_avail_and_initialized():
        return 0

    return parallel_state.get_sequence_parallel_group()


def get_sequence_parallel_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return parallel_state.get_sequence_parallel_rank()

def get_data_parallel_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return parallel_state.get_data_parallel_world_size()


def get_dataloader_local_world_size():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return 1

    return worker_info.num_workers


def get_dataloader_local_rank():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return 0

    return worker_info.id


def get_dataloader_world_size():
    return get_data_parallel_world_size() * get_dataloader_local_world_size()


def get_dataloader_rank():
    local_world_size = get_dataloader_local_world_size()
    local_rank = get_dataloader_local_rank()
    worker_id = get_data_parallel_rank() * local_world_size + local_rank
    return worker_id


def get_data_parallel_group():
    if not is_dist_avail_and_initialized():
        return None

    return parallel_state.get_data_parallel_group()


def get_data_parallel_rank():
    if not is_dist_avail_and_initialized():
        return 0

    return parallel_state.get_data_parallel_rank()


def get_pipeline_parallel_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return parallel_state.get_pipeline_model_parallel_world_size()


def get_tensor_parallel_group():
    if not is_dist_avail_and_initialized():
        return None

    return parallel_state.get_tensor_model_parallel_group()


def get_tensor_parallel_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return parallel_state.get_tensor_model_parallel_world_size()


def get_tensor_parallel_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return parallel_state.get_tensor_model_parallel_rank()


def get_rank(group=None):
    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank(group=group)


def get_local_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0

    return int(os.getenv('LOCAL_RANK', '0'))


def is_parallel_world() -> bool:
    return get_world_size() > 1


def is_first_rank(group=None) -> bool:
    return get_rank(group=group) == 0


# code compat
is_main_process = is_first_rank


def is_last_rank(group=None) -> bool:
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)
    return rank == (world_size - 1)


def synchronize(group=None):
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    world_size = get_world_size(group=group)
    if world_size == 1:
        return
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def all_reduce(x, op=dist.ReduceOp.SUM, group=None):
    if get_world_size() < 2:
        return x
    group = group or parallel_state.get_data_parallel_group()
    return dist.all_reduce(x, op=op, group=group)


def all_reduce_mean(x, group=None):
    all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    world_size = get_world_size(group)
    return x / world_size


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    world_size = get_world_size(group=group)
    if world_size == 1:
        return [data]

    rank = get_rank(group)
    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed(group=None):
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints, group=group)

    return all_ints[0]


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_num_devices():
    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES', None)
    if gpu_list is not None:
        return len(gpu_list.split(','))
    else:
        devices_list_info = os.popen("nvidia-smi -L")
        devices_list_info = devices_list_info.read().strip().split("\n")
        return len(devices_list_info)


def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return torch.cuda.get_device_capability()[0] >= 7
    except:
        return False


def destroy_process_group(group=None):
    dist.destroy_process_group(group=group)


def get_host_name():
    return socket.gethostname()


def all_reduce_dict(input_dict, op=dist.ReduceOp.SUM, group=None):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size(group=group)
    if world_size < 2:
        return input_dict

    reduced_dict = {}
    with torch.no_grad():
        # sort the keys so that they are consistent across processes
        for name, value in sorted(input_dict.items()):
            reduced_dict[name] = all_reduce(value, op=op, group=group)

    return reduced_dict


def is_rank_0(group=None):
    """Check whether it is rank 0."""
    return get_rank(group=group) == 0


def is_local_rank_0():
    """Check whether it is local rank 0."""
    return get_local_rank() == 0
