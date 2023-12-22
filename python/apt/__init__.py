from .ops import *
from .sageconv import *
from .planner import *
import os


def _load_apt_library():
    package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    so_path = os.path.join(package_path, "libapt.so")
    try:
        torch.classes.load_library(so_path)
        print(f"[Note]load so from {so_path}")
    except Exception:
        raise ImportError("Cannot load APT C++ library")


def _init_broadcast(rank, local_rank, world_size, node_size, device, num_nccl_comms):
    print(f"[Note]No#[{rank}\t {local_rank}\t {world_size}] device:{device}")
    nccl_unique_id_list = []
    for i in range(num_nccl_comms):
        nccl_unique_id = torch.ops.apt.nccl_get_unique_id().to(device)
        dist.broadcast(nccl_unique_id, 0)
        nccl_unique_id = nccl_unique_id.to("cpu")
        nccl_unique_id_list.append(nccl_unique_id)

    nccl_unique_id_list = torch.vstack(nccl_unique_id_list)
    torch.ops.apt.init(
        rank,
        local_rank,
        world_size,
        nccl_unique_id_list,
        node_size,
    )


def init(rank, local_rank, world_size, node_size, num_nccl_comms=1, device=None, init_mp=True):
    _load_apt_library()
    if init_mp:
        _init_broadcast(rank, local_rank, world_size, node_size, device, num_nccl_comms)
