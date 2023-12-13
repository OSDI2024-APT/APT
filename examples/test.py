import npc
import torch
import dgl
import utils
import time
import threading
from typing import List
import torch.distributed as dist
import torch.multiprocessing as mp


def check_tensor(ts: torch.Tensor):
    print(f"[Note]ts shape:{ts.shape}\t dtype:{ts.dtype}")


def get_time():
    torch.cuda.synchronize()
    dist.barrier()
    return time.time()


def get_time_straggler():
    torch.cuda.synchronize()
    t1 = time.time()
    dist.barrier()
    t2 = time.time()
    return t1, t2


def test_c_test_func():
    npc.init(rank=1, world_size=1, shared_queue=None, init_mp=False)
    test_tensor = torch.arange(8).reshape(-1, 2)
    print(f"[Note]test_tensor:{test_tensor}")

    sorted_idx = torch.LongTensor([3, 2, 0, 1])
    print(f"[Note]Python test_tensor:{test_tensor}\t sorted_idx:{sorted_idx}")
    npc.test(test_tensor, sorted_idx)


def test_cache_graphs(rank, world_size, args):
    local_rank = rank
    device = torch.device(f"cuda:{local_rank}")
    utils.setup(rank=rank, local_rank=local_rank, world_size=world_size, args=args)

    npc.init(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        init_mp=True,
    )
    """
    sample_graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
    indptr, indices, edge_ids = sample_graph.adj_sparse("csr")
    num_local_nodes = -1
    num_graph_nodes = 4
    num_cached_nodes = 4
    sorted_idx = torch.arange(num_graph_nodes)

    npc.cache_graphs(
        num_local_nodes, num_graph_nodes, num_cached_nodes, sorted_idx, indptr, indices
    )
    """


def run_mp_test(world_size=4):
    args = utils.init_args()
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        test_cache_graphs,
        args=(
            world_size,
            args,
        ),
        nprocs=world_size,
        join=True,
    )
    # npc.init(rank=1, world_size=1, shared_queue=None, init_mp=False)


# test gloo
# -- gloo all-to-all #--


def thread_send(
    rank: int,
    to_send_rank: List[int],
    input_list: List[torch.Tensor],
    output_list: List[torch.Tensor],
):
    for dst in to_send_rank:
        if dst != rank:
            dist.send(tensor=input_list[dst], dst=dst)


def thread_recv(
    rank: int,
    to_recv_rank: List[int],
    input_list: List[torch.Tensor],
    output_list: List[torch.Tensor],
):
    for src in to_recv_rank:
        if src != rank:
            dist.recv(tensor=output_list[src], src=src)
        else:
            output_list[src].copy_(input_list[src])


def thread_send_v2(to_send_rank: List[int], input_list: List[torch.Tensor], offset=0):
    for i, dst in enumerate(to_send_rank):
        dist.send(tensor=input_list[i + offset], dst=dst)


def thread_recv_v2(to_recv_rank: List[int], output_list: List[torch.Tensor], offset=0):
    for i, src in enumerate(to_recv_rank):
        dist.recv(tensor=output_list[i + offset], src=src)


def gloo_all_to_all_without_size(
    rank: int,
    input_tensor: torch.Tensor,
    send_sizes: torch.Tensor,
    comm_rank_lists: List[int],
) -> List[torch.Tensor]:
    # gloo all to all recv size
    num_peers = len(comm_rank_lists)
    recv_sizes = [torch.empty(1, dtype=torch.long) for _ in range(num_peers)]
    send_sizes_split = list(torch.split(send_sizes, 1))
    send_sizes_list = send_sizes.tolist()
    print(f"[Note]Rk#{rank} num_peers:{comm_rank_lists} send_sizes:{send_sizes_split}\t")

    # 1 send thread & recv thread to communicate recv sizes
    send_size_thread = threading.Thread(target=thread_send_v2, args=(comm_rank_lists, send_sizes_split))
    send_size_thread.start()
    recv_size_thread = threading.Thread(target=thread_recv_v2, args=(comm_rank_lists, recv_sizes))
    recv_size_thread.start()

    send_size_thread.join()
    recv_size_thread.join()

    dist.barrier()
    print(f"[Note]Rk#{rank}\t recv_sizes:{recv_sizes}")

    recv_sizes_list = [rs.item() for rs in recv_sizes]

    # one thread for each rank to send & recv
    output_tensor_list = [torch.empty(recv_sizes_list[i], dtype=torch.long) for i in range(num_peers)]
    send_thread_list = []
    recv_thread_list = []
    # input_tensor_split = torch.split(input_tensor, 1)
    input_tensor_split = list(torch.split(input_tensor, send_sizes.tolist()))
    print(f"[Note]Rk#{rank} input_tensor_split:{input_tensor_split}")
    for i in range(num_peers):
        send_thread = threading.Thread(target=thread_send_v2, args=([comm_rank_lists[i]], input_tensor_split, i))
        send_thread.start()
        send_thread_list.append(send_thread)

        recv_thread = threading.Thread(target=thread_recv_v2, args=([comm_rank_lists[i]], output_tensor_list, i))
        recv_thread.start()
        recv_thread_list.append(recv_thread)

    for i in range(num_peers):
        send_thread_list[i].join()
        recv_thread_list[i].join()

    print(f"[Note]Rk#{rank} output_tensor_list:{output_tensor_list}")
    # send back (,2)
    recv_subtensor = torch.cat(output_tensor_list, dim=0).repeat_interleave(2).reshape(-1, 2)
    recv_subtensor_list = recv_subtensor.split(recv_sizes_list)
    final_subtensor_list = [torch.empty((send_sizes_list[i], 2), dtype=torch.long) for i in range(num_peers)]
    print(f"[Note]Rk#{rank} recv_subtensor:{recv_subtensor}")
    for i in range(num_peers):
        send_thread = threading.Thread(target=thread_send_v2, args=([comm_rank_lists[i]], recv_subtensor_list, i))
        send_thread.start()
        send_thread_list[i] = send_thread
        recv_thread = threading.Thread(target=thread_recv_v2, args=([comm_rank_lists[i]], final_subtensor_list, i))
        recv_thread.start()
        recv_thread_list[i] = recv_thread

    for i in range(num_peers):
        send_thread_list[i].join()
        recv_thread_list[i].join()

    return final_subtensor_list


def gloo_all_to_all(
    rank: int,
    world_size: int,
    input_list: List[torch.Tensor],
    output_list: List[torch.Tensor],
    to_send_rank_lists: List[int],
    to_recv_rank_lists: List[int],
    # num_send_threads=1,
    # num_recv_threads=1,
):
    # print(f"[Note]input_list:{input_list}")
    # print(f"[Note]output_list:{output_list}")
    assert len(input_list) == len(output_list)

    send_thread_list = []
    recv_thread_list = []
    num_send_threads = len(to_send_rank_lists)
    num_recv_threads = len(to_recv_rank_lists)
    for to_send_rank in to_send_rank_lists:
        send_thread = threading.Thread(
            target=thread_send,
            args=(
                rank,
                to_send_rank,
                input_list,
                output_list,
            ),
        )
        send_thread.start()
        send_thread_list.append(send_thread)

    for to_recv_rank in to_recv_rank_lists:
        recv_thread = threading.Thread(
            target=thread_recv,
            args=(
                rank,
                to_recv_rank,
                input_list,
                output_list,
            ),
        )
        recv_thread.start()
        recv_thread_list.append(recv_thread)

    for i in range(num_send_threads):
        send_thread_list[i].join()

    for i in range(num_recv_threads):
        recv_thread_list[i].join()


def test_gloo_all_to_all_correctness(rank, world_size, remote_worker_id):
    # all to all with known
    """
    test_size = 10
    send_tensor_list = [
        torch.arange(test_size, dtype=torch.float32) + rank * 10
        for rank in range(world_size)
    ]
    recv_tensor_list = [
        torch.empty(10, dtype=torch.float32) for rank in range(world_size)
    ]
    to_send_rank_lists = [[i] for i in range(world_size)]
    to_recv_rank_lists = [[i] for i in range(world_size)]
    print(f"[Note]Rk#{rank}: send_tensor_list:{send_tensor_list}")
    gloo_all_to_all(
        rank,
        world_size,
        send_tensor_list,
        recv_tensor_list,
        to_send_rank_lists,
        to_recv_rank_lists,
    )
    print(f"[Note]Rk#{rank}: recv_tensor_listt:{recv_tensor_list}")
    """
    # all to all with unknown size
    test_size = torch.arange(world_size) + 1
    to_send_size = test_size[remote_worker_id]
    total_size = torch.sum(to_send_size).item()
    input_tensor = torch.arange(total_size, dtype=torch.long) + rank * 10
    print(f"[Note]Rk#{rank} / {world_size}: input_tensor:{input_tensor}")

    output_tensor = gloo_all_to_all_without_size(rank, input_tensor, send_sizes=to_send_size, comm_rank_lists=remote_worker_id)
    print(f"[Note]Rk#{rank} / {world_size}: output_tensor:{output_tensor}")


def test_two_backend_bandwidth(args, rank, local_rank, device, world_size, run_times=20, warmup_times=5):
    print(f"[Note]test two backend bandwidth Rank#{rank}\t device:{device}")
    test_size_list = [10**i for i in range(3, 9)]
    # test_size_list = [10**8]
    for test_size in test_size_list:
        # gpu nccl
        nccl_time_list = []
        nccl_wait_time_list = []
        for test_time in range(run_times):
            send_tensor_list = [torch.rand(test_size, dtype=torch.float32, device=device) for rank in range(world_size)]
            recv_tensor_list = [torch.rand(test_size, dtype=torch.float32, device=device) for rank in range(world_size)]
            start_time = get_time()
            dist.all_to_all(recv_tensor_list, send_tensor_list)
            # dist.all_to_all_single(output=recv_tensor, input=send_tensor)
            # end_time = get_time()
            be, end_time = get_time_straggler()
            elapsed_time = round((end_time - start_time) * 1000.0, 4)
            waiting_time = round((end_time - be) * 1000.0, 4)
            # print(f"[Note]elapsed_time:{elapsed_time}")
            if test_time >= warmup_times:
                nccl_time_list.append(elapsed_time)
                nccl_wait_time_list.append(waiting_time)
            time.sleep(1)
        nccl_max = max(nccl_time_list)
        nccl_min = min(nccl_time_list)
        nccl_avg = sum(nccl_time_list) / len(nccl_time_list)
        print(
            f"[Note]Nccl size:{test_size}\t max:{nccl_max}\t min:{nccl_min}\t avg:{nccl_avg}\t wait: max:{max(nccl_wait_time_list)}\t min:{min(nccl_wait_time_list)}\t avg:{sum(nccl_wait_time_list) / len(nccl_wait_time_list)}"
        )

        # cpu gloo one thread for each rank to send & recv
        """
        remote_worker_id = args.remote_worker_id[1:]
        input_split_sizes = [0 for _ in range(world_size)]
        output_split_sizes = [0 for _ in range(world_size)]
        num_remote_workers = len(remote_worker_id)
        for r in remote_worker_id:
            input_split_sizes[r] = test_size
            output_split_sizes[r] = test_size
        gloo_time_list = []
        gloo_wait_time_list = []
        for test_time in range(run_times):
            send_tensor = torch.rand(
                test_size * num_remote_workers, dtype=torch.float32
            )
            recv_tensor = torch.empty(
                test_size * num_remote_workers, dtype=torch.float32
            )
            start_time = get_time()
            dist.all_to_all_single(
                output=recv_tensor,
                input=send_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
            )

            be, end_time = get_time_straggler()
            elapsed_time = round((end_time - start_time) * 1000.0, 4)
            waiting_time = round((end_time - be) * 1000.0, 4)
            if test_time >= warmup_times:
                gloo_time_list.append(elapsed_time)
                gloo_wait_time_list.append(waiting_time)
            time.sleep(10)

        gloo_max = max(gloo_time_list)
        gloo_min = min(gloo_time_list)
        gloo_avg = sum(gloo_time_list) / len(gloo_time_list)
        print(
            f"[Note]Gloo Size:{test_size}\t max:{gloo_max}\t min:{gloo_min}\t avg:{gloo_avg}\t wait: max:{max(gloo_wait_time_list)}\t min:{min(gloo_wait_time_list)}\t avg:{sum(gloo_wait_time_list) / len(gloo_wait_time_list)}"
        )
        """

        # record mini-batches stage time
        """
                    record_val = [
                        ms_sampling_time,
                        # t0 - bt0,
                        ms_loading_time,
                        # t1 - bt1,
                        ms_training_time,
                        # t2 - bt2,
                    ]
                    record_list.append(record_val)
        """
        """
                    if record_flag and args.model != "GAT" and not args.debug:
                        # V_shuffle, V_reshuffle, cache_miss, ms_loading_time
                        if args.system == "DP":
                            ms_record = [0, 0]
                        elif args.system == "NP":
                            send_offset = sample_result[4].tolist()
                            recv_offset = sample_result[5].tolist()
                            total_send = send_offset[world_size - 1] - send_offset[rank] + (send_offset[rank - 1] if rank > 0 else 0)
                            total_recv = recv_offset[world_size - 1] - recv_offset[rank] + (recv_offset[rank - 1] if rank > 0 else 0)
                            ms_record = [total_send + total_recv, 2 * args.num_hidden * (total_send + total_recv)]
                        elif args.system == "SP":
                            send_sizes = sample_result[3].tolist()
                            recv_sizes = sample_result[4].tolist()
                            total_shuffle = 0
                            total_reshuffle = 0
                            for r in range(world_size):
                                if r != rank:
                                    total_shuffle += send_sizes[2 * r] + recv_sizes[2 * r]
                                    total_reshuffle += 2 * args.num_hidden * (send_sizes[2 * r + 1] + recv_sizes[2 * r + 1])
                            ms_record = [total_shuffle, total_reshuffle]

                        elif args.system == "MP":
                            # total_shuffle = sum(recv_frontier_size) + recv_frontier_size[rank] * (world_size-2) + sum(recv_coo_size) + recv_coo_size[rank] * (world_size-2)
                            recv_frontier_size = sample_result[2][0][2]
                            recv_coo_size = sample_result[2][0][3]
                            total_shuffle = (
                                recv_frontier_size[rank].item() * (world_size - 2)
                                + sum(recv_frontier_size).item()
                                + recv_coo_size[rank].item() * (world_size - 2)
                                + sum(recv_coo_size).item()
                            )

                            send_size = sample_result[3]  # shape: [1]
                            recv_size = sample_result[4]  # shape: [world_size]

                            # 2 * means forward + backward
                            total_reshuffle = 2 * args.num_hidden * (send_size.item() * (world_size - 2) + sum(recv_size).item())
                            ms_record = [total_shuffle, total_reshuffle]

                        cache_miss = sample_result[0].numel() - torch.sum(cache_mask[sample_result[0].cpu()]).item()
                        ms_record.extend([cache_miss, ms_sampling_time, ms_loading_time, ms_training_time])

                        for i in range(6):
                            record_list[i].append(ms_record[i])
        """
        """
        # write mean of record to csv_path
        if record_flag and args.model != "GAT" and not args.debug:
            record_path = f"./logs/costmodel/Nov20_sm_all.csv"
            avg_record = [round(sum(record_list[i]) / len(record_list[i]), 2) for i in range(6)]
            input_tensor = torch.tensor(avg_record, device=device)
            output_list = [torch.empty(6, device=device) for _ in range(world_size)] if rank == 0 else None
            dist.gather(input_tensor, output_list, 0)
            if rank == 0:
                print(f"[Note] Write records to {record_path}")
                with open(record_path, "a") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    for r in range(world_size):
                        tag = f"{args.tag}_{args.system}_rk#{r}"
                        write_list = [tag] + output_list[r].tolist()
                        writer.writerow(write_list)
        """


# -- end of gloo all-to-all #--

if __name__ == "__main__":
    run_mp_test()
    exit(0)
    npc.init(0, 0, 1, None, False)
    s1 = torch.cuda.Stream()

    x = torch.arange(10).reshape(5, 2)
    x.share_memory_()
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0)
    print(x.is_shared())
    print(x.is_pinned())
