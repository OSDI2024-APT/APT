import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np


def run_unidirection(rank, world_size, backend, data_size):
    def run_once():
        if rank == 0:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:0")

            t1_time = time.time()
            dist.send(tensor=tensor, dst=1)
            # Wait for all data to be sent
            torch.cuda.synchronize(device="cuda:0")
            return time.time() - t1_time
        else:
            tensor = torch.randn(data_size, dtype=torch.float32, device="cuda:1")

            t1_time = time.time()
            dist.recv(tensor=tensor, src=0)
            torch.cuda.synchronize(device="cuda:1")
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


def run_bidirection(rank, world_size, backend, data_size_list):
    def run_once():
        input = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{rank}")
        output = torch.randn(data_size, dtype=torch.float32, device=f"cuda:{rank}")

        t1_time = time.time()
        dist.all_to_all_single(output, input)
        # Wait for all data to be sent
        torch.cuda.synchronize(device=f"cuda:{rank}")
        return 1000.0 * (time.time() - t1_time)

    x = []
    y = []
    for data_size in data_size_list:
        # execute a few rounds of warmup
        warmup_time = 0.0
        for _ in range(2):
            warmup_time += run_once()
        # measure runtime
        benchmark_time = []

        for _ in range(10):
            benchmark_time.append(run_once())
            maxx = np.max(benchmark_time)
            minn = np.min(benchmark_time)
            avg = np.mean(benchmark_time)

        x.append([data_size * 2 * (world_size - 1) / world_size, 1])
        y.append(avg)

        if rank == 0:
            print(
                f"[Note]run_bidirection Rank: {rank} | Backend: {backend} | Data Vol.: {(data_size * 4) / (1000 * 1000)} MB | Warmup: {(warmup_time):.3f} ms | Max: {maxx:.5f} ms | Min: {minn:.5f} s | Avg: {avg:.5f} ms"
            )

    if rank == 0:
        curve_fit(x, y)


def init_process(rank, size, fn, backend, data_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, backend, data_size)


def test_pcie_bandwidth(data_size_list=[1000000, 10000000, 100000000, 500000000, 1000000000]):
    def run_once():
        test_tensor = torch.randn(data_size, dtype=torch.float32).pin_memory()
        # test tensor copy to gpu time
        time_start = time.time()
        gpu_tensor = test_tensor.cuda()
        torch.cuda.synchronize()
        time_end = time.time()
        return 1000.0 * (time_end - time_start)

    x = []
    y = []
    for data_size in data_size_list:
        # execute a few rounds of warmup
        warmup_time = 0.0
        for _ in range(2):
            warmup_time += run_once()
        # measure runtime
        benchmark_time = []
        for _ in range(10):
            benchmark_time.append(run_once())

        maxx = np.max(benchmark_time)
        minn = np.min(benchmark_time)
        avg = np.mean(benchmark_time)

        x.append([data_size, 1])
        y.append(avg)

        print(
            f"Data Vol.: {(data_size * 4) / (1000 * 1000)} MB | Warmup: {(warmup_time):.3f} s | Max: {maxx:.5f} ms | Min: {minn:.5f} ms | Avg: {avg:.5f} ms"
        )
    curve_fit(x, y)


def print_list_percentile(lists):
    sorted_list = sorted(lists)
    length = len(sorted_list)
    perc = [sorted_list[min(length - 1, int(0.1 * i * length))] for i in range(1, 10)]
    print(f"[Note] perc:{perc}")


def curve_fit(x, y):
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x, y)
    print(f"[Note]reg.coef_:{reg.coef_}")
    y_pred = reg.predict(x)
    err_list = []
    for y1, y2 in zip(y_pred, y):
        err = abs(y1 - y2) / y2
        print(f"[Note]y1:{y1}\ty2:{y2}\terr:{err}")


if __name__ == "__main__":
    data_size_list = [10**i for i in range(0, 9)]
    test_pcie_bandwidth(data_size_list=data_size_list)
    exit(0)
    mp.set_start_method("spawn")

    world_size = 8
    func = run_bidirection

    print(f"World size: {world_size}")

    # for data_size in [1000000, 10000000, 100000000, 500000000, 1000000000]:
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, func, "nccl", data_size_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
