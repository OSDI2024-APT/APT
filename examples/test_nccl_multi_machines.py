import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse
import numpy as np
import utils


def run_unidirection(rank, local_rank, world_size, backend, data_size):
    def run_once():
        if rank == 0:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:0")

            t1_time = time.time()
            dist.send(tensor=tensor, dst=1)
            # Wait for all data to be sent
            torch.cuda.synchronize(device="cuda:0")
            return time.time() - t1_time
        else:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:0")

            t1_time = time.time()
            dist.recv(tensor=tensor, src=0)
            torch.cuda.synchronize(device="cuda:0")
            return time.time() - t1_time

    # execute a few rounds of warmup
    warmup_time = 0.0
    for _ in range(2):
        warmup_time += run_once()
    # measure runtime
    benchmark_time = []
    for _ in range(10):
        benchmark_time.append(run_once())

    print(
        f"Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / (1000 * 1000)} MB | Warmup: {(warmup_time):.3f} s | Max: {np.max(benchmark_time):.5f} s | Min: {np.min(benchmark_time):.5f} s | Avg: {np.mean(benchmark_time):.5f} s"
    )


def run_bidirection(rank, local_rank, world_size, backend, data_size):
    def run_once():
        input = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        output = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        group_size = world_size // 2
        if rank < group_size:
            input_splits = [0] * group_size + [data_size // group_size] * group_size
        else:
            input_splits = [data_size // group_size] * group_size + [0] * group_size
        output_splits = input_splits

        t1_time = time.time()

        dist.all_to_all_single(output, input, output_splits, input_splits)
        # Wait for all data to be sent
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        return 1000.0 * (time.time() - t1_time)

    # execute a few rounds of warmup
    warmup_time = 0.0
    for _ in range(2):
        warmup_time += run_once()
    # measure runtime
    benchmark_time = []
    for _ in range(10):
        benchmark_time.append(run_once())

    print(
        f"Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / (1000 * 1000)} MB | Warmup: {(warmup_time):.3f} ms | Max: {np.max(benchmark_time):.5f} ms | Min: {np.min(benchmark_time):.5f} ms | Avg: {np.mean(benchmark_time):.5f} ms"
    )


def run_bidirection_mixed(rank, local_rank, world_size, backend, data_size):
    def run_once():
        input = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{local_rank}")
        output = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{local_rank}")

        t1_time = time.time()
        dist.all_to_all_single(output, input)
        # Wait for all data to be sent
        torch.cuda.synchronize(device=f"cuda:{local_rank}")
        return 1000.0 * (time.time() - t1_time)

    # execute a few rounds of warmup
    warmup_time = 0.0
    for _ in range(1):
        warmup_time += run_once()
    # measure runtime
    benchmark_time = []
    for _ in range(10):
        benchmark_time.append(run_once())

    print(
        f"Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / (1000 * 1000)} MB | Warmup: {(warmup_time):.3f} ms | Max: {np.max(benchmark_time):.5f} ms | Min: {np.min(benchmark_time):.5f} ms | Avg: {np.mean(benchmark_time):.5f} ms"
    )


def init_process(rank, local_rank, size, fn, backend, data_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "172.31.40.2"
    os.environ["MASTER_PORT"] = "29503"

    dist.init_process_group(backend, rank=rank, world_size=size)
    print(f"[Note]Rk#{rank} / {size} Initialize complete, local_rank:{local_rank} backend:{backend}")
    fn(rank, local_rank, size, backend, data_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, required=True, help="node in multi-machine")
    args = parser.parse_args()

    mp.set_start_method("spawn")

    world_size = 16
    num_workers_per_node = 4

    func = run_bidirection_mixed
    # func = run_bidirection

    print(f"World size: {world_size} | Node: {args.node}")

    data_size_list = [10**i * world_size for i in range(3, 9)]
    for data_size in data_size_list:
        # for data_size in [1000000]:
        processes = []
        for rank in range(num_workers_per_node):
            p = mp.Process(
                target=init_process,
                args=(
                    rank + args.node * num_workers_per_node,
                    rank,
                    world_size,
                    func,
                    "nccl",
                    data_size,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
