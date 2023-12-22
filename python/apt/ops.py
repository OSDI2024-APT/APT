import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Union, Tuple, Optional
import argparse
import dgl
import time
import os
from dataclasses import dataclass
import threading
import queue
import psutil


GB_TO_BYTES = 1024 * 1024 * 1024
BYTES_PER_ELEMENT = 4


# -- utils --
def show_process_memory_usage(tag: str) -> None:
    process = psutil.Process(os.getpid())
    print(f"[Note]{tag} memory usage:{process.memory_info().rss / 1024**2}MB")


def get_tensor_mem_usage_in_gb(ts: torch.Tensor):
    return ts.numel() * ts.element_size() / GB_TO_BYTES


def get_tensor_info_str(ts):
    return f"shape:{ts.shape}\t max: {torch.max(ts)}\t min: {torch.min(ts)}"


def get_total_mem_usage_in_gb():
    symem = psutil.virtual_memory()
    total = symem[0] / GB_TO_BYTES
    uses = symem[3] / GB_TO_BYTES
    return f"Mem usage: {uses} / {total}GB"


def nccl_get_unique_id() -> torch.Tensor:
    return torch.ops.apt.nccl_get_unique_id()


def allgather(a: torch.Tensor, comm_type: int = 0) -> torch.Tensor:
    return torch.ops.apt.allgather(a, comm_type)


def cache_feats_shared(
    num_total_nodes: int,
    localnode_feats: torch.Tensor,
    cached_feats: torch.Tensor,
    cached_idx: torch.Tensor,
    localnode_idx: Optional[torch.Tensor] = None,
    feat_dim_offset: int = 0,
) -> None:
    torch.ops.apt.cache_feats_shared(
        num_total_nodes,
        localnode_feats,
        cached_feats,
        cached_idx,
        localnode_idx,
        feat_dim_offset,
    )


def mix_cache_graphs(
    global_indptr: torch.Tensor,
    global_indices: torch.Tensor,
    num_cached_nodes: int = 0,
    cached_node_idx: torch.Tensor = torch.tensor([], dtype=torch.long),
    cached_indptr: torch.Tensor = torch.empty(0, dtype=torch.long),
    cached_indices: torch.Tensor = torch.empty(0, dtype=torch.long),
):
    torch.ops.apt.mix_cache_graphs(
        num_cached_nodes,
        cached_node_idx,
        cached_indptr,
        cached_indices,
        global_indptr,
        global_indices,
    )


def preparation(args: argparse.ArgumentParser, indptr: torch.Tensor, indices: torch.Tensor) -> None:
    min_vids_list = args.min_vids
    device = args.device
    min_vids = torch.LongTensor(min_vids_list).to(device)
    # register min_vids
    register_min_vids(min_vids)
    mix_cache_graphs(indptr, indices)


def load_partition(
    args: argparse.ArgumentParser,
    shared_tensor_list: List[torch.Tensor],
) -> None:
    (
        localnode_feats_idx,
        localnode_feats,
        cache_feat_node_idx,
        *rest,
    ) = shared_tensor_list

    rank = args.rank
    device = args.device
    num_localnode_feats, feat_dim = localnode_feats.shape
    num_total_nodes = args.min_vids[-1]
    # cache_mask = torch.zeros(num_total_nodes,).bool()
    # local_nodes_id = torch.arange(num_local_nodes)
    # global_nodes_id = local_nodes_id + min_vids_list[rank]
    # total_node_id = torch.arange(num_total_nodes)

    # cache feat
    # num_cached_feat_elements = num_cached_feat_nodes * args.input_dim
    # map cache_feat_node_idx to pos
    localnode_feat_pos = torch.zeros(num_total_nodes, dtype=torch.long)
    localnode_feat_pos[localnode_feats_idx] = torch.arange(num_localnode_feats)
    cache_feat_node_pos = localnode_feat_pos[cache_feat_node_idx]

    if args.system == "NFP":
        # single machine scenario

        cached_feats = localnode_feats[
            cache_feat_node_pos,
            args.cumsum_feat_dim[rank] : args.cumsum_feat_dim[rank + 1],
        ].to(device)
        feat_dim_offset = args.cumsum_feat_dim[rank]

    else:
        cached_feats = localnode_feats[cache_feat_node_pos].to(device)
        feat_dim_offset = 0
    cache_feats_shared(
        num_total_nodes=num_total_nodes,
        localnode_feats=localnode_feats,
        cached_feats=cached_feats,
        cached_idx=cache_feat_node_idx,
        localnode_idx=localnode_feats_idx,
        feat_dim_offset=feat_dim_offset,
    )
    dist.barrier()


def register_min_vids(shuffle_min_vids: torch.Tensor, shuffle_id_offset: int = 0) -> None:
    torch.ops.apt.register_min_vids(shuffle_min_vids, shuffle_id_offset)


def register_multi_machines_scheme(args: argparse.Namespace) -> Optional[torch.Tensor]:
    gpu_remote_worker_map = torch.tensor(args.remote_worker_map).to(f"cuda:{args.local_rank}")
    remote_worker_id = torch.tensor(args.remote_worker_id)
    torch.ops.apt.register_multi_machines_scheme(gpu_remote_worker_map, remote_worker_id)


def _load_subtensor(
    args: argparse.Namespace,
    seeds: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.apt.load_subtensor(seeds)


def load_subtensor(args, sample_result):
    if args.system == "DNP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]perm, [4]send_offset, [5]recv_offset, [6]inverse_idx
        fsi = NPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            num_dst=None,
            permutation=sample_result[3],
            send_offset=sample_result[4].to("cpu"),
            recv_offset=sample_result[5].to("cpu"),
            inverse_idx=sample_result[6],
        )
        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    elif args.system == "SNP":
        if args.model == "GAT":
            # [0]input_nodes, [1]seeds, [2]blocks, [3]perm, [4]send_offset, [5]recv_offset
            fsi = NPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                num_dst=sample_result[2][0].number_of_dst_nodes(),
                permutation=sample_result[3],
                send_offset=sample_result[4].to("cpu"),
                recv_offset=sample_result[5].to("cpu"),
                inverse_idx=None,
            )
        elif args.shuffle_with_dst:
            # [0]input_nodes [1]seeds, [2]blocks [3]send_size [4]recv_size
            send_sizes = sample_result[3].to("cpu")
            recv_sizes = sample_result[4].to("cpu")
            num_send_dst = send_sizes[0::3].sum().item()
            num_recv_dst = recv_sizes[0::3].sum().item()
            num_dst = [num_send_dst, num_recv_dst]
            total_send_size = num_send_dst + send_sizes[2::3].sum().item()
            total_recv_size = num_recv_dst + recv_sizes[2::3].sum().item()

            fsi = SPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                send_sizes=send_sizes,
                recv_sizes=recv_sizes,
                num_dst=num_dst,
                total_send_size=total_send_size,
                total_recv_size=total_recv_size,
                shuffle_with_dst=args.shuffle_with_dst,
            )
        else:
            # elif args.model == "GCN" or args.model == "SAGE":
            # [0]input_nodes [1] seeds, [2]blocks [3]send_size [4]recv_size
            send_sizes = sample_result[3].to("cpu")
            recv_sizes = sample_result[4].to("cpu")
            num_dst = sample_result[2][1].number_of_src_nodes()
            total_send_size = send_sizes[1::2].sum().item()
            total_recv_size = recv_sizes[1::2].sum().item()

            fsi = SPFeatureShuffleInfo(
                feat_dim=args.num_hidden,
                send_sizes=send_sizes,
                recv_sizes=recv_sizes,
                num_dst=num_dst,
                total_send_size=total_send_size,
                total_recv_size=total_recv_size,
                shuffle_with_dst=args.shuffle_with_dst,
            )

        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    elif args.system == "GDP":
        # [0]input_nodes, [1]seeds, [2]blocks
        return sample_result[2], _load_subtensor(args, sample_result[0])
    elif args.system == "NFP":
        # [0]input_nodes, [1]seeds, [2]blocks, [3]send_size, [4]recv_size
        fsi = MPFeatureShuffleInfo(
            feat_dim=args.num_hidden,
            send_size=sample_result[3].to("cpu"),
            recv_size=sample_result[4].to("cpu"),
        )
        return (
            sample_result[2],
            _load_subtensor(args, sample_result[0]),
            fsi,
        )
    else:
        raise NotImplementedError


def shuffle_seeds(
    seeds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.apt.shuffle_seeds(seeds)


@dataclass
class NPFeatureShuffleInfo(object):
    feat_dim: int
    num_dst: int
    send_offset: List[int]
    recv_offset: List[int]
    permutation: torch.Tensor
    inverse_idx: torch.Tensor


class NPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsi: NPFeatureShuffleInfo, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = feat_shuffle(
            input_tensor,
            fsi.send_offset,
            fsi.recv_offset,
            fsi.permutation,
            fsi.feat_dim,
            1,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: NPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = feat_shuffle(
            grad_output_tensor,
            fsi.recv_offset,
            fsi.send_offset,
            fsi.permutation,
            fsi.feat_dim,
            0,
        )
        return (None, shuffle_grad)


class SPFeatureShuffleGAT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsi: NPFeatureShuffleInfo, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = feat_shuffle(
            input_tensor,
            fsi.send_offset,
            fsi.recv_offset,
            fsi.permutation,
            fsi.feat_dim,
            1,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: NPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = feat_shuffle(
            grad_output_tensor,
            fsi.recv_offset,
            fsi.send_offset,
            fsi.permutation,
            fsi.feat_dim,
            0,
        )
        return (None, shuffle_grad)


@dataclass
class SPFeatureShuffleInfo(object):
    feat_dim: int
    send_sizes: torch.Tensor
    recv_sizes: torch.Tensor
    # int or tuple(int,int)
    num_dst: Union[int, Tuple[int, int]]
    total_send_size: int
    total_recv_size: int
    shuffle_with_dst: int = 0


class SPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fsi: SPFeatureShuffleInfo,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = sp_feat_shuffle(
            input_tensor,
            fsi.recv_sizes,
            fsi.send_sizes,
            fsi.total_send_size,
            fsi.feat_dim,
            fsi.shuffle_with_dst,
        )
        return shuffle_result

    @staticmethod
    def backward(
        ctx,
        grad_output_tensor: torch.Tensor,
    ) -> torch.Tensor:
        fsi: SPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = sp_feat_shuffle(
            grad_output_tensor,
            fsi.send_sizes,
            fsi.recv_sizes,
            fsi.total_recv_size,
            fsi.feat_dim,
            fsi.shuffle_with_dst,
        )
        return (None, shuffle_grad)


@dataclass
class MPFeatureShuffleInfo(object):
    feat_dim: int
    send_size: torch.Tensor
    recv_size: torch.Tensor


class MPFeatureShuffle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fsi: MPFeatureShuffleInfo,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fsi = fsi
        shuffle_result = mp_feat_shuffle_fwd(
            input_tensor,
            fsi.recv_size,
            fsi.send_size,
            fsi.feat_dim,
        )
        return shuffle_result

    @staticmethod
    def backward(ctx, grad_output_tensor: torch.Tensor) -> torch.Tensor:
        fsi: MPFeatureShuffleInfo = ctx.fsi
        shuffle_grad = mp_feat_shuffle_bwd(
            grad_output_tensor / fsi.recv_size.numel(),
            fsi.send_size,
            fsi.recv_size,
            fsi.feat_dim,
        )
        return (None, shuffle_grad)


def feat_shuffle(
    inputs: torch.Tensor,
    send_offset: torch.Tensor,
    recv_offset: torch.Tensor,
    permutation: torch.Tensor,
    feat_dim: int,
    fwd_flag: int,
):
    return torch.ops.apt.feat_shuffle(inputs, send_offset, recv_offset, permutation, feat_dim, fwd_flag)


def sp_feat_shuffle(
    input: torch.Tensor,
    send_sizes: torch.Tensor,
    recv_sizes: torch.Tensor,
    total_recv_size: int,
    feat_dim: int,
    shuffle_with_dst: int,
):
    return torch.ops.apt.sp_feat_shuffle(input, send_sizes, recv_sizes, total_recv_size, feat_dim, shuffle_with_dst)


def mp_feat_shuffle_fwd(
    input: torch.Tensor,
    send_size: torch.Tensor,
    recv_size: torch.Tensor,
    feat_dim: int,
):
    return torch.ops.apt.mp_feat_shuffle_fwd(input, send_size, recv_size, feat_dim)


def mp_feat_shuffle_bwd(
    input: torch.Tensor,
    send_size: torch.Tensor,
    recv_size: torch.Tensor,
    feat_dim: int,
):
    return torch.ops.apt.mp_feat_shuffle_bwd(input, send_size, recv_size, feat_dim)
