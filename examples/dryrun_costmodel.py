import atexit
import dgl
import npc
import torch
import time
import os

import torch.multiprocessing as mp
import torch.distributed as dist
import utils

WARMUP_EPOCHS = 1
DRYRUN_EPOCHS = 1
BYTES_PER_ELEMENT = 4
BW_PCIE = 5594.79
BW_GPU = 5782.60
# TODO set bw bandwidth
BW_NETWORK = 4616.41
BW_PURE_NETWORK = 3831.46


def get_time():
    torch.cuda.synchronize()
    dist.barrier()
    return time.time()


def run(rank, local_rank, world_size, args, shared_tensor_list):
    # init
    device = torch.device(f"cuda:{local_rank}")
    args.rank = rank
    args.local_rank = local_rank
    args.device = device
    backend = "NCCL"
    utils.setup(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        args=args,
        backend=backend,
    )
    node_size = world_size
    if args.nproc_per_node != -1 and args.hybrid:
        node_size = args.nproc_per_node
    print(f"[Note]Node_size: {node_size}")
    npc.init(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        node_size=node_size,
        num_nccl_comms=1,
        device=device,
        init_mp=True,
    )

    # init freq
    gpu_ori_sorted_idx, gpu_npc_sorted_idx, gpu_sp_sorted_idx, cpu_ori_sorted_idx, cpu_npc_sorted_idx = load_dryrun_result(args)

    train_mask = shared_tensor_list.pop(0)

    min_vids = args.min_vids
    num_total_nodes = min_vids[-1]

    for ts in shared_tensor_list:
        utils.pin_tensor(ts)

    npc.load_partition_dryrun(args, rank, device, shared_tensor_list)

    # rebalance train_idx
    train_idx = torch.masked_select(torch.arange(num_total_nodes), train_mask)
    num_train_nids_per_rank = train_idx.numel() // world_size
    local_train_idx = train_idx[rank * num_train_nids_per_rank : (rank + 1) * num_train_nids_per_rank]
    local_train_idx = local_train_idx.to(device)

    fake_graph = dgl.rand_graph(1, 1)
    # build sampler
    sampler = npc.AllPSNeighborSampler(
        rank=rank,
        world_size=world_size,
        fanouts=args.fan_out,
        model=args.model,
        num_total_nodes=num_total_nodes,
        shuffle_with_dst=args.shuffle_with_dst,
    )
    # build dataloader
    dataloader = dgl.dataloading.DataLoader(
        graph=fake_graph,
        indices=local_train_idx,
        graph_sampler=sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        use_uva=True,
    )
    dp_seeds_freq = torch.zeros((num_total_nodes,), dtype=torch.int32)
    np_seeds_freq = torch.zeros((num_total_nodes,), dtype=torch.int32)
    sp_seeds_freq = torch.zeros((num_total_nodes,), dtype=torch.int32)
    mp_seeds_freq = torch.zeros((num_total_nodes,), dtype=torch.int32)
    dp_reshuffle_vol = 0
    np_reshuffle_vol = 0
    sp_reshuffle_vol = 0
    mp_reshuffle_vol = 0
    num_record_steps = 0
    # start training
    warmup_start = get_time()
    for epoch in range(WARMUP_EPOCHS):
        print(f"[Note] Warmup epoch:{epoch}")
        for step, sample_result in enumerate(dataloader):
            pass
    warmup_end = get_time()
    sampler.clear_time_list()
    dryrun_start = get_time()
    for epoch in range(DRYRUN_EPOCHS):
        print(f"[Note] Dryrun epoch:{epoch}")
        for step, sample_result in enumerate(dataloader):
            num_record_steps += 1
            """
            print(f"[Note]sample_result type:{type(sample_result)}\t len: {len(sample_result)}")
            for i, sub_sample in enumerate(sample_result):
                print(f"[Note]sample{i}: type:{type(sub_sample)}\t len: {len(sub_sample)}")
            """
            # decompose sample result of each strategies
            dp_sample_result, np_sample_result, sp_sample_result, mp_sample_result = sample_result
            # dp
            (dp_seeds,) = dp_sample_result
            # np
            np_seeds, np_send_offset, np_recv_offset = np_sample_result
            # sp
            sp_seeds, sp_send_sizes, sp_recv_sizes = sp_sample_result
            # mp
            mp_seeds, mp_send_size, mp_recv_size = mp_sample_result

            # update freq
            dp_seeds_freq[dp_seeds.cpu()] += 1
            np_seeds_freq[np_seeds.cpu()] += 1
            sp_seeds_freq[sp_seeds.cpu()] += 1
            mp_seeds_freq[mp_seeds.cpu()] += 1

            # reshuffle vol
            dp_reshuffle_vol += get_dp_reshuffle_vol()
            np_reshuffle_vol += get_np_reshuffle_vol(args, rank, np_send_offset, np_recv_offset)
            sp_reshuffle_vol += get_sp_reshuffle_vol(args, rank, sp_send_sizes, sp_recv_sizes)
            mp_reshuffle_vol += get_mp_reshuffle_vol(args, rank, mp_send_size, mp_recv_size)
            # print(f"[Note]mp_send_size:{mp_send_size}\tmp_recv_size:{mp_recv_size}")
    dryrun_end = get_time()
    warmup_time = warmup_end - warmup_start
    dryrun_time = dryrun_end - dryrun_start
    log_info = f"warmup_time:{warmup_time:.4f}\t warmup_epoch:{WARMUP_EPOCHS}\t dryrun_time:{dryrun_time:.4f}\t  dryrun_epoch:{DRYRUN_EPOCHS}\t num_record_steps:{num_record_steps}"
    with open("dryrun_time.txt", "a") as f:
        f.write(log_info + "\n")
    print(f"[Note]warmup_time:{warmup_end - warmup_start:.4f}\t dryrun_time:{dryrun_end - dryrun_start:.4f}")
    """
    # special magic for DP :)
    # allreduce dp_seeds_freq
    made_dp_seeds_freq = dp_seeds_freq.to(device)
    dist.all_reduce(made_dp_seeds_freq, op=dist.ReduceOp.SUM)
    made_dp_seeds_freq = made_dp_seeds_freq.cpu()
    """

    sample_time_list = sampler.get_time_list()
    num_sampling_time, sampling_algo_time, other_overhead = sample_time_list
    print(f"[Note]num_sampling_time:{num_sampling_time}\t sampling_algo_time:{sampling_algo_time}\t other_overhead:{other_overhead}")
    dp_sampling_time = sum(sampling_algo_time) + other_overhead[0]
    np_sampling_time = sum(sampling_algo_time[:-1]) + other_overhead[1]
    sp_sampling_time = sum(sampling_algo_time) + other_overhead[2]
    mp_sampling_time = sum(sampling_algo_time) + other_overhead[3]
    # print sampling time :4f
    print(
        f"[Note]dp_sampling_time:{dp_sampling_time:.4f}\t np_sampling_time:{np_sampling_time:.4f}\t sp_sampling_time:{sp_sampling_time:.4f}\t mp_sampling_time:{mp_sampling_time:.4f}"
    )

    # cache miss
    # predefined_cache_nodes = int(0.1 * num_total_nodes)

    """
    dp_cache_miss_list = [
        get_cache_miss(dp_seeds_freq, num_cache_nodes, args.input_dim, made_dp_seeds_freq) for num_cache_nodes in predefine_num_cache_nodes_list
    ]
    np_cache_miss_list = [get_cache_miss(np_seeds_freq, num_cache_nodes, args.input_dim) for num_cache_nodes in predefine_num_cache_nodes_list]
    sp_cache_miss_list = [get_cache_miss(sp_seeds_freq, num_cache_nodes, args.input_dim) for num_cache_nodes in predefine_num_cache_nodes_list]
    mp_cache_miss_list = [
        get_cache_miss(mp_seeds_freq, num_cache_nodes * world_size, args.mp_input_dim_list[rank])
        for num_cache_nodes in predefine_num_cache_nodes_list
    ]
    # print cache miss
    # print(f"[Note]dp_cache_miss:{dp_cache_miss}\t np_cache_miss:{np_cache_miss}\t sp_cache_miss:{sp_cache_miss}\t mp_cache_miss:{mp_cache_miss}")
    dp_cache_miss_time = [cachemiss_vol_to_time(dp_cache_miss) for dp_cache_miss in dp_cache_miss_list]
    np_cache_miss_time = [cachemiss_vol_to_time(np_cache_miss) for np_cache_miss in np_cache_miss_list]
    sp_cache_miss_time = [cachemiss_vol_to_time(sp_cache_miss) for sp_cache_miss in sp_cache_miss_list]
    mp_cache_miss_time = [cachemiss_vol_to_time(mp_cache_miss) for mp_cache_miss in mp_cache_miss_list]
    """
    if args.nproc_per_node == -1:
        predefine_num_cache_nodes_list = [int(i * 1024 * 1024 * 1024 / (args.input_dim * BYTES_PER_ELEMENT)) for i in range(9)]
        dp_cache_miss_time = [
            get_featload_time(args, dp_seeds_freq, num_cache_nodes, args.input_dim, gpu_ori_sorted_idx)
            for num_cache_nodes in predefine_num_cache_nodes_list
        ]
        np_cache_miss_time = [
            get_featload_time(args, np_seeds_freq, num_cache_nodes, args.input_dim, gpu_npc_sorted_idx)
            for num_cache_nodes in predefine_num_cache_nodes_list
        ]
        sp_cache_miss_time = [
            get_featload_time(args, sp_seeds_freq, num_cache_nodes, args.input_dim, gpu_sp_sorted_idx)
            for num_cache_nodes in predefine_num_cache_nodes_list
        ]
        mp_cache_miss_time = [
            get_featload_time(args, mp_seeds_freq, num_cache_nodes * world_size, args.mp_input_dim_list[rank], gpu_ori_sorted_idx)
            for num_cache_nodes in predefine_num_cache_nodes_list
        ]
    else:
        # NOTE: multi-machine we set 4GB gpu cache
        num_gpu_cache_nodes = int(4 * 1024 * 1024 * 1024 / (args.input_dim * BYTES_PER_ELEMENT))

        predefine_num_cache_nodes_list = [int(i * 1024 * 1024 * 1024 / (args.input_dim * BYTES_PER_ELEMENT)) for i in args.nl]
        print(f"[Note]predefine_num_cache_nodes_list:{predefine_num_cache_nodes_list}")

        dp_cache_miss_time = [
            get_featload_time(args, dp_seeds_freq, num_gpu_cache_nodes, args.input_dim, gpu_ori_sorted_idx, cpu_ori_sorted_idx, cpu_num_cache_nodes)
            for cpu_num_cache_nodes in predefine_num_cache_nodes_list
        ]
        print(f"[Note]dp_cache_miss_time:{dp_cache_miss_time}")
        np_cache_miss_time = [
            get_featload_time(args, np_seeds_freq, num_gpu_cache_nodes, args.input_dim, gpu_npc_sorted_idx, cpu_npc_sorted_idx, cpu_num_cache_nodes)
            for cpu_num_cache_nodes in predefine_num_cache_nodes_list
        ]
        print(f"[Note]np_cache_miss_time:{np_cache_miss_time}")
        sp_cache_miss_time = [get_featload_time_multi_machine_spmp(args, sp_seeds_freq, num_gpu_cache_nodes, args.input_dim, gpu_sp_sorted_idx)]
        print(f"[Note]sp_cache_miss_time:{sp_cache_miss_time}")
        mp_cache_miss_time = [
            get_featload_time_multi_machine_spmp(
                args, mp_seeds_freq, num_gpu_cache_nodes * world_size, args.mp_input_dim_list[rank], gpu_ori_sorted_idx
            )
        ]
        print(f"[Note]mp_cache_miss_time:{mp_cache_miss_time}")
    # print cache miss time :4f
    # print(f"[Note]dp_cache_miss_time:{dp_cache_miss_time:.4f}\t np_cache_miss_time:{np_cache_miss_time:.4f}\t sp_cache_miss_time:{sp_cache_miss_time:.4f}\t mp_cache_miss_time:{mp_cache_miss_time:.4f}")
    # reshuffle
    # print(f"[Note]dp_reshuffle_vol:{dp_reshuffle_vol}\t np_reshuffle_vol:{np_reshuffle_vol}\t sp_reshuffle_vol:{sp_reshuffle_vol}\t mp_reshuffle_vol:{mp_reshuffle_vol}")
    dp_reshuffle_time = reshuffle_vol_to_time(args, dp_reshuffle_vol)
    np_reshuffle_time = reshuffle_vol_to_time(args, np_reshuffle_vol)
    sp_reshuffle_time = reshuffle_vol_to_time(args, sp_reshuffle_vol)
    mp_reshuffle_time = reshuffle_vol_to_time(args, mp_reshuffle_vol)
    # print reshuffle time :4f
    # print(f"[Note]dp_reshuffle_time:{dp_reshuffle_time:.4f}\t np_reshuffle_time:{np_reshuffle_time:.4f}\t sp_reshuffle_time:{sp_reshuffle_time:.4f}\t mp_reshuffle_time:{mp_reshuffle_time:.4f}")
    # pack value together
    dp_val = [dp_sampling_time] + dp_cache_miss_time + [dp_reshuffle_time]
    np_val = [np_sampling_time] + np_cache_miss_time + [np_reshuffle_time]
    sp_val = [sp_sampling_time] + sp_cache_miss_time + [sp_reshuffle_time]
    mp_val = [mp_sampling_time] + mp_cache_miss_time + [mp_reshuffle_time]
    dp_sizes = len(dp_val)
    np_szies = len(np_val)
    sp_sizes = len(sp_val)
    mp_sizes = len(mp_val)

    all_ps_val = dp_val + np_val + sp_val + mp_val
    all_ps_sizes = len(all_ps_val)
    print(f"[Note]all_ps_sizes:{all_ps_sizes}")
    print(f"[Note]dp_sizes:{dp_sizes}\t np_szies:{np_szies}\t sp_sizes:{sp_sizes}\t mp_sizes:{mp_sizes}")
    to_send_tensor = torch.tensor(
        all_ps_val,
        dtype=torch.float32,
        device=device,
    )

    # avg by DRYRUN_EPOCHS
    to_send_tensor /= DRYRUN_EPOCHS

    dist.reduce(to_send_tensor, dst=0, op=dist.ReduceOp.MAX)
    if rank == 0:
        ret = to_send_tensor.tolist()
        ret = [val * 1000.0 for val in ret]
        print(f"[Note]ret shape: {len(ret)}")
        print(f"[Note]ret:{ret}")

        dataset_name = args.configs_path.split("/")[-2]
        save_dir_prefix = f"./costmodel/micro_exp_{DRYRUN_EPOCHS}"
        os.makedirs(save_dir_prefix, exist_ok=True)
        output_file_path = os.path.join(save_dir_prefix, f"dryrun_{dataset_name}_{args.model}_{args.fan_out}.txt")
        # write ret to output_file_path
        with open(output_file_path, "w") as f:
            for val in ret:
                f.write(f"{val:.2f},")
            f.write("\n")

        # print to check
        print(f"[Note]dp_val:{ret[0:dp_sizes]}")
        print(f"[Note]np_val:{ret[dp_sizes:dp_sizes+np_szies]}")
        print(f"[Note]sp_val:{ret[dp_sizes+np_szies:dp_sizes+np_szies+sp_sizes]}")
        print(f"[Note]mp_val:{ret[dp_sizes+np_szies+sp_sizes:]}")

    dist.barrier()


def get_cache_miss(seeds_freq, num_cache_nodes, input_dim, made_seeds_freq=None):
    if made_seeds_freq is None:
        made_seeds_freq = seeds_freq
    num_cache_nodes = min(num_cache_nodes, seeds_freq.numel())
    sorted_freq, sorted_idx = torch.sort(made_seeds_freq, descending=True)
    total_access = seeds_freq.sum()
    total_cached = seeds_freq[sorted_idx[:num_cache_nodes]].sum()
    total_miss = total_access - total_cached
    return total_miss * input_dim


def get_featload_time_multi_machine_spmp(
    args, seeds_freq, num_gpu_cache_nodes, input_dim, gpu_sorted_idx=None, cpu_sorted_idx=None, num_cpu_cache_nodes=0
):
    if gpu_sorted_idx is None:
        gpu_sorted_idx = torch.sort(seeds_freq, descending=True)[1]

    num_gpu_cache_nodes = min(num_gpu_cache_nodes, seeds_freq.numel())

    total_access = seeds_freq.sum()
    total_cached = seeds_freq[gpu_sorted_idx[:num_gpu_cache_nodes]].sum()
    total_miss = total_access - total_cached
    return cachemiss_vol_to_time(total_miss * input_dim)


def get_featload_time(args, seeds_freq, num_gpu_cache_nodes, input_dim, gpu_sorted_idx=None, cpu_sorted_idx=None, num_cpu_cache_nodes=0):
    if args.nproc_per_node == -1:
        if gpu_sorted_idx is None:
            gpu_sorted_idx = torch.sort(seeds_freq, descending=True)[1]

        num_gpu_cache_nodes = min(num_gpu_cache_nodes, seeds_freq.numel())

        total_access = seeds_freq.sum()
        total_cached = seeds_freq[gpu_sorted_idx[:num_gpu_cache_nodes]].sum()
        total_miss = total_access - total_cached
        return cachemiss_vol_to_time(total_miss * input_dim)
    else:
        assert cpu_sorted_idx is not None
        num_cpu_cache_nodes = min(num_cpu_cache_nodes, seeds_freq.numel())
        total_access = seeds_freq.sum()
        total_cpu_access = seeds_freq[cpu_sorted_idx[:num_cpu_cache_nodes]].sum()
        total_cpu_missed = total_access - total_cpu_access
        if gpu_sorted_idx is None:
            gpu_sorted_idx = torch.sort(seeds_freq, descending=True)[1]

        # select gpu_sorted_idx that in cpu_sorted_idx
        vis_mask = torch.zeros(seeds_freq.numel(), dtype=torch.bool)
        vis_mask[cpu_sorted_idx[:num_cpu_cache_nodes]] = True
        gpu_sorted_idx = torch.masked_select(gpu_sorted_idx, vis_mask[gpu_sorted_idx])

        num_gpu_cache_nodes = min(num_gpu_cache_nodes, seeds_freq.numel())
        # remove remote cpu freq
        remote_seeds_freq = seeds_freq.clone()
        remote_seeds_freq[cpu_sorted_idx[:num_cpu_cache_nodes]] = 0
        remote_seeds_freq = remote_seeds_freq.to(args.device)
        dist.all_reduce(remote_seeds_freq, op=dist.ReduceOp.SUM)
        # remote_seeds_freq = remote_seeds_freq + seeds_freq[cpu_sorted_idx[:num_cpu_cache_nodes]]
        remote_seeds_freq = remote_seeds_freq.cpu()
        remote_seeds_freq[cpu_sorted_idx[:num_cpu_cache_nodes]] += seeds_freq[cpu_sorted_idx[:num_cpu_cache_nodes]]
        total_remote_access = remote_seeds_freq.sum()
        total_remote_cached = remote_seeds_freq[gpu_sorted_idx[:num_gpu_cache_nodes]].sum()
        total_remote_miss = total_remote_access - total_remote_cached

        cpu_miss_time = cpu_cachemiss_vol_to_time(total_cpu_missed * input_dim)
        gpu_miss_time = cachemiss_vol_to_time(total_remote_miss * input_dim)
        return cpu_miss_time + gpu_miss_time


def cpu_cachemiss_vol_to_time(vol, bytes_per_element=4):
    return vol * bytes_per_element / (1024 * 1024 * BW_PURE_NETWORK)


def cachemiss_vol_to_time(vol, bytes_per_element=4):
    return vol * bytes_per_element / (1024 * 1024 * BW_PCIE)


def reshuffle_vol_to_time(args, vol, bytes_per_element=4):
    bw = BW_NETWORK if args.nproc_per_node else BW_PCIE
    return vol * bytes_per_element / (1024 * 1024 * bw)


def get_reshuffle_dim(args):
    return args.num_hidden if args.model != "GAT" else args.num_hidden * args.num_heads


def get_dp_reshuffle_vol():
    return 0


def get_np_reshuffle_vol(args, rank, send_offset, recv_offset):
    # print(f"[Note]Model:{args.model} np_reshuffle_dim:{np_reshuffle_dim}")
    send_offset = send_offset.tolist()
    recv_offset = recv_offset.tolist()
    total_send = send_offset[args.world_size - 1] - send_offset[rank] + (send_offset[rank - 1] if rank > 0 else 0)
    total_recv = recv_offset[args.world_size - 1] - recv_offset[rank] + (recv_offset[rank - 1] if rank > 0 else 0)
    total_reshuffle_vol = 2 * get_reshuffle_dim(args) * (total_send + total_recv)
    return total_reshuffle_vol


def get_sp_reshuffle_vol(args, rank, send_sizes, recv_sizes):
    # print(f"[Note]Model:{args.model} sp_reshuffle_dim:{args.num_hidden}")
    if args.model == "GAT":
        total_reshuffle_vol = get_np_reshuffle_vol(args, rank, send_sizes, recv_sizes)
    else:
        send_sizes = send_sizes.tolist()
        recv_sizes = recv_sizes.tolist()
        total_reshuffle_vol = 0

        if args.shuffle_with_dst:
            for r in range(args.world_size):
                if r != rank:
                    total_reshuffle_vol += (
                        2 * get_reshuffle_dim(args) * (send_sizes[3 * r] + send_sizes[3 * r + 2] + recv_sizes[3 * r] + recv_sizes[3 * r + 2])
                    )
        else:
            for r in range(args.world_size):
                if r != rank:
                    total_reshuffle_vol += 2 * get_reshuffle_dim(args) * (send_sizes[2 * r + 1] + recv_sizes[2 * r + 1])

    return total_reshuffle_vol


def get_mp_reshuffle_vol(args, rank, send_size, recv_size):
    # print(f"[Note]Model:{args.model} mp_reshuffle_dim:{args.num_hidden}")
    total_reshuffle_vol = 2 * get_reshuffle_dim(args) * (send_size.item() * (args.world_size - 2) + sum(recv_size).item())
    return total_reshuffle_vol


def load_dryrun_result(args):
    rank = args.rank
    world_size = args.world_size
    fanout_info = str(args.fan_out).replace(" ", "")
    config_key = args.configs_path.split("/")[-2]
    # dryrun_file_path = f"{args.caching_candidate_path_prefix}/{key}_{config_key}_{fanout_info}"
    ori_freq_list = [
        torch.load(f"{args.caching_candidate_path_prefix}/ori_{config_key}_{fanout_info}/rk#{r}_epo100.pt")[1] for r in range(world_size)
    ]
    npc_freq_list = [
        torch.load(f"{args.caching_candidate_path_prefix}/npc_{config_key}_{fanout_info}/rk#{r}_epo100.pt")[1] for r in range(world_size)
    ]

    gpu_ori_freq = torch.stack(ori_freq_list).sum(dim=0)
    gpu_npc_freq = npc_freq_list[rank]
    gpu_sp_freq = torch.empty_like(gpu_ori_freq)
    gpu_sp_freq[args.min_vids[rank] : args.min_vids[rank + 1]] = gpu_ori_freq[args.min_vids[rank] : args.min_vids[rank + 1]]
    gpu_ori_sorted_idx = torch.sort(gpu_ori_freq, descending=True)[1]
    gpu_sp_sorted_idx = torch.sort(gpu_sp_freq, descending=True)[1]
    gpu_npc_sorted_idx = torch.sort(gpu_npc_freq, descending=True)[1]

    if args.nproc_per_node != -1:
        st = args.node_rank * args.nproc_per_node
        en = st + args.nproc_per_node
        # sum [st, en)
        cpu_ori_freq = torch.stack(ori_freq_list[st:en]).sum(dim=0)
        cpu_npc_freq = torch.stack(npc_freq_list[st:en]).sum(dim=0)
        # force put [min_vids[st], min_vids[en]] at the front, others sort descending
        max_v = cpu_ori_freq.max()
        cpu_ori_freq[args.min_vids[st] : args.min_vids[en]] = max_v
        cpu_ori_sorted_idx = torch.sort(cpu_ori_freq, descending=True)[1]
        max_v = cpu_npc_freq.max()
        cpu_npc_freq[args.min_vids[st] : args.min_vids[en]] = max_v
        cpu_npc_sorted_idx = torch.sort(cpu_npc_freq, descending=True)[1]

    else:
        cpu_ori_freq = None
        cpu_npc_freq = None
        cpu_ori_sorted_idx = None
        cpu_npc_sorted_idx = None

    del ori_freq_list
    del npc_freq_list

    return gpu_ori_sorted_idx, gpu_npc_sorted_idx, gpu_sp_sorted_idx, cpu_ori_sorted_idx, cpu_npc_sorted_idx


if __name__ == "__main__":
    args, shared_tensor_list, global_nfeat = utils.pre_spawn()

    # we do not need labels
    shared_tensor_list.pop(0)
    # manually set cross machine feat load
    args.cross_machine_feat_load = False
    fanout_list = [[10, 15], [10, 25], [10, 10, 10]]
    for fanout in fanout_list:
        model = "SAGE"
        args.model = model
        # args.model = "SAGE"
        args.shuffle_with_dst = args.model != "GCN" and args.nproc_per_node != -1
        num_heads, num_hidden = (4, 8) if model == "GAT" else (-1, 16)
        args.num_heads = num_heads
        args.num_hidden = num_hidden
        world_size = args.world_size
        nproc = world_size if args.nproc_per_node == -1 else args.nproc_per_node
        ranks = args.ranks
        local_ranks = args.local_ranks
        processes = []
        for i in range(nproc):
            p = mp.Process(target=run, args=(ranks[i], local_ranks[i], world_size, args, shared_tensor_list))
            atexit.register(utils.kill_proc, p)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
