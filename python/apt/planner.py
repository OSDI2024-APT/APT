import torch
import dgl
from typing import Tuple, List
from dgl.heterograph import DGLBlock
from .ops import *
from .sageconv import *

BW_PCIE = 5594.79
BW_GPU = 5782.60
NUM_STRATEGIES = 4
SP_BASE = 1000000
DRYRUN_EPOCHS = 1


def pin_tensor(tensor: torch.Tensor):
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)


def np_sample_and_shuffle(seeds: torch.Tensor, fanout: int):
    return torch.ops.apt.np_sample_and_shuffle(seeds, fanout)


def srcdst_to_vir(fanout: int, dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.apt.srcdst_to_vir(fanout, dst, src)


def src_to_vir(fanout: int, num_dst: int, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.apt.src_to_vir(fanout, num_dst, src)


def sp_sample_and_shuffle(
    num_dst: int,
    send_frontier: torch.Tensor,
    sorted_allnodes: torch.Tensor,
    unique_frontier: torch.Tensor,
    shuffle_with_dst: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.apt.sp_sample_and_shuffle(num_dst, send_frontier, sorted_allnodes, unique_frontier, shuffle_with_dst)


def sp_sample_shuffle_src(
    unique_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.apt.sp_sample_shuffle_src(unique_src)


def mp_sample_shuffle(seeds: torch.Tensor, unique_frontier: torch.Tensor, coo_row: torch.Tensor) -> List[torch.Tensor]:
    return torch.ops.apt.mp_sample_shuffle(seeds, unique_frontier, coo_row)


def create_block_from_csc(indptr, indices, e_ids, num_src, num_dst) -> DGLBlock:
    hgidx = dgl.heterograph_index.create_unitgraph_from_csr(
        2,
        num_src,
        num_dst,
        indptr,
        indices,
        e_ids,
        formats=["coo", "csr", "csc"],
        transpose=True,
    )
    retg = DGLBlock(hgidx, (["_N"], ["_N"]), ["_E"])
    return retg


def create_block_from_coo(row, col, num_src, num_dst) -> DGLBlock:
    hgidx = dgl.heterograph_index.create_unitgraph_from_coo(
        2,
        num_src,
        num_dst,
        row,
        col,
        formats=["coo", "csr", "csc"],
    )
    retg = DGLBlock(hgidx, (["_N"], ["_N"]), ["_E"])
    return retg


def tensor_relabel_csc(seeds, neighbors) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.apt.relabel_csc(seeds, neighbors)


def create_dgl_block(seeds, neighbors, fanout, return_coo=False):
    unique_frontier, indices = tensor_relabel_csc(seeds, neighbors)
    coo_col = torch.arange(0, seeds.numel(), device=indices.device).repeat_interleave(fanout)

    block = create_block_from_coo(
        indices,
        coo_col,
        num_src=unique_frontier.numel(),
        num_dst=seeds.numel(),
    )
    block.srcdata["_ID"] = unique_frontier

    if return_coo:
        return block, (indices, coo_col)
    else:
        return block


# GDP: dummy op
def GDPReorganizeOp(samples, args):
    return samples


def NFPReorganizeOp(samples, args):
    input_nodes, output_nodes, blocks = samples
    seeds = blocks[0].dstdata["_ID"]
    neighbors = blocks[0].srcdata["_ID"]
    fanout = neighbors.numel() // seeds.numel()
    unique_frontier, coo_row = tensor_relabel_csc(seeds, neighbors)
    (
        all_frontier,
        all_coo_row,
        send_size,
        recv_size,
        recv_frontier_size,
        recv_coo_size,
    ) = mp_sample_shuffle(seeds, unique_frontier, coo_row)

    all_coo_col = torch.cat([torch.arange(0, i, device=all_coo_row.device).repeat_interleave(fanout) for i in recv_size])
    blocks.insert(0, (all_coo_row, all_coo_col, recv_frontier_size, recv_coo_size, recv_size))
    # sampling_result = (send_size, recv_size)
    return all_frontier, output_nodes, blocks, send_size, recv_size


def SNPReorganizeOp(samples, args):
    # print(f"[Note] SNPReorganizeOp samples:{samples}")
    input_nodes, output_nodes, blocks = samples
    # print(f"[Note] blocks:{blocks}")
    seeds = blocks[0].dstdata["_ID"]
    fanout = 10
    seeds, neighbors = torch.ops.apt.local_sample_one_layer(seeds, fanout)

    device = seeds.device
    num_dst = seeds.numel()

    map_src = src_to_vir(fanout, num_dst, neighbors)
    sorted_mapsrc, perm_mapsrc = torch.sort(map_src)

    unique_frontier, arange_src = torch.unique(map_src, return_inverse=True)
    arange_dst = unique_frontier % num_dst  # [0, num_dst)
    arange_src = torch.arange(0, unique_frontier.numel(), device=device)  # [0, #unique_frontier)
    block1 = create_block_from_coo(arange_src, arange_dst, unique_frontier.numel(), num_dst)
    # blocks.insert(0, block1)

    perm_dst = sorted_mapsrc % num_dst
    send_frontier = args.rank * (SP_BASE * args.num_nodes) + perm_dst * args.num_nodes + neighbors[perm_mapsrc]

    (
        recv_seeds,
        recv_neighbors,
        send_sizes,
        recv_sizes,
    ) = sp_sample_and_shuffle(
        num_dst,  # num_dst
        send_frontier,  # send_frontier
        sorted_mapsrc,  # sorted_mapsrc
        unique_frontier,  # unique_frontier
    )

    unique_src, arange_src = torch.unique(recv_neighbors, return_inverse=True)
    unique_dst, arange_dst = torch.unique(recv_seeds, return_inverse=True)
    block2 = create_block_from_coo(arange_src, arange_dst, unique_src.numel(), unique_dst.numel())

    # blocks.insert(0, block2)
    blocks[0] = [block2, block1]
    # sampling_result = (send_sizes, recv_sizes)
    seeds = torch.cat((seeds, unique_src))

    return seeds, output_nodes, blocks, send_sizes, recv_sizes


def DNPReorganizeOp(samples, args):
    input_nodes, output_nodes, blocks = samples
    seeds = blocks[1].srcdata[dgl.NID]
    # neighbors = blocks[0].srcdata["_ID"]
    fanout = 10
    blocks.pop(0)
    (
        shuffled_seeds,
        neighbors,
        perm,
        send_offset,
        recv_offset,
        inverse_idx,
    ) = np_sample_and_shuffle(seeds, fanout)
    blocks0 = create_dgl_block(shuffled_seeds, neighbors, fanout)
    blocks.insert(0, blocks0)
    seeds = blocks0.srcdata[dgl.NID]
    return seeds, output_nodes, blocks, perm, send_offset, recv_offset, inverse_idx


parallel_strategy = "GDP"


def layer_barrier(layer_id, features, fsi):
    if parallel_strategy == "GDP":
        return features
    elif parallel_strategy == "NFP":
        if layer_id != 1:
            return features
        else:
            return MPFeatureShuffle.apply(fsi[0], features)
    elif parallel_strategy == "SNP":
        if layer_id == 0:
            return (features, fsi[0])
        else:
            return features
    elif parallel_strategy == "DNP":
        if layer_id != 1:
            return features
        else:
            features = features[fsi[0].inverse_idx]
            return NPFeatureShuffle.apply(fsi[0], features)
    else:
        raise NotImplementedError


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


class Planner:
    def __init__(self, args, hardware_info, shared_tensor_list):
        self.hardware_info = hardware_info
        self.bw_pcie = self.hardware_info.get("bw_pcie", BW_PCIE)
        self.bw_gpu = self.hardware_info.get("bw_gpu", BW_GPU)
        self.args = args
        self.labels = shared_tensor_list.pop().to(args.device)
        self.shared_tensor_list = shared_tensor_list
        for tensor in self.shared_tensor_list:
            pin_tensor(tensor)

        indices = shared_tensor_list.pop()
        indptr = shared_tensor_list.pop()
        preparation(self.args, indptr, indices)
        # mix_cache_graphs(indptr, indices)

    def reshuffle_vol_to_time(self, vol, bytes_per_element=4):
        return vol * bytes_per_element / (1024 * 1024 * self.bw_gpu)

    def cachemiss_vol_to_time(self, vol, bytes_per_element=4):
        return vol * bytes_per_element / (1024 * 1024 * self.bw_pcie)

    def get_featload_time(self, seeds_freq, num_gpu_cache_nodes, input_dim, gpu_sorted_idx=None):
        if gpu_sorted_idx is None:
            gpu_sorted_idx = torch.sort(seeds_freq, descending=True)[1]

        num_gpu_cache_nodes = min(num_gpu_cache_nodes, seeds_freq.numel())

        total_access = seeds_freq.sum()
        total_cached = seeds_freq[gpu_sorted_idx[:num_gpu_cache_nodes]].sum()
        total_miss = total_access - total_cached
        return self.cachemiss_vol_to_time(total_miss * input_dim)

    def _plan(self, dataloader):
        sampling_overhead = [0 for _ in range(NUM_STRATEGIES)]

        input_nodes_freq = [torch.zeros(self.args.num_nodes).long() for _ in range(NUM_STRATEGIES)]
        reshuffle_vol = [0 for _ in range(NUM_STRATEGIES)]

        for epoch in range(DRYRUN_EPOCHS):
            for samples in dataloader:
                time_list = []
                t0 = time.time()
                gdp_samples = GDPReorganizeOp(samples, self.args)
                t1 = time.time()
                input_nodes_freq[0][gdp_samples[0].cpu()] += 1
                reshuffle_vol[0] += get_dp_reshuffle_vol()
                sampling_overhead[0] += t1 - t0

        for epoch in range(DRYRUN_EPOCHS):
            for samples in dataloader:
                t0 = time.time()
                nfp_samples = NFPReorganizeOp(samples, self.args)
                t1 = time.time()
                reshuffle_vol[1] += get_mp_reshuffle_vol(self.args, self.args.rank, nfp_samples[3], nfp_samples[4])
                input_nodes_freq[1][nfp_samples[0].cpu()] += 1
                sampling_overhead[0] += t1 - t0

        for epoch in range(DRYRUN_EPOCHS):
            for samples in dataloader:
                t0 = time.time()
                snp_samples = SNPReorganizeOp(samples, self.args)
                t1 = time.time()
                input_nodes_freq[2][snp_samples[0].cpu()] += 1
                reshuffle_vol[2] += get_sp_reshuffle_vol(self.args, self.args.rank, snp_samples[3], snp_samples[4])
                sampling_overhead[2] += t1 - t0

        for epoch in range(DRYRUN_EPOCHS):
            for samples in dataloader:
                t0 = time.time()
                dnp_samples = DNPReorganizeOp(samples, self.args)
                t1 = time.time()
                input_nodes_freq[3][dnp_samples[0].cpu()] += 1
                reshuffle_vol[3] += get_np_reshuffle_vol(self.args, self.args.rank, dnp_samples[4], dnp_samples[5])
                sampling_overhead[3] += t1 - t0

        total_time = [0 for _ in range(NUM_STRATEGIES)]
        for i in range(NUM_STRATEGIES):
            cache_miss_time = self.get_featload_time(input_nodes_freq[i], self.args.num_gpu_cache_nodes, self.args.input_dim)
            reshuffle_time = self.reshuffle_vol_to_time(reshuffle_vol[i])
            total_time[i] = sampling_overhead[i] + cache_miss_time + reshuffle_time

        # find minimum total_time
        if self.args.rank == 0:
            min_time = min(total_time)
            min_index = total_time.index(min_time)
            min_index = torch.tensor([min_index], dtype=torch.long, device=self.args.device)
        else:
            min_index = torch.empty(1, dtype=torch.long, device=self.args.device)
        torch.distributed.broadcast(min_index, 0)
        min_index = min_index.item()
        print("[Note]min_index:", min_index)
        sorted_gpu_idx = torch.sort(input_nodes_freq[min_index], descending=True)[1][: self.args.num_gpu_cache_nodes]

        return (min_index, sorted_gpu_idx)

    def plan(self, dataloader, model):
        # select optimal parallel strategy
        best_strategy_idx, cache_feat_node_idx = self._plan(dataloader)

        print(f"[Note]Best strategy:{best_strategy_idx}\t cache_feat_node_idx:{cache_feat_node_idx}")
        if best_strategy_idx == 0:
            self.parallel_strategy = "GDP"
            self.reorganize_op = GDPReorganizeOp
        elif best_strategy_idx == 1:
            self.parallel_strategy = "NFP"
            self.reorganize_op = NFPReorganizeOp
        elif best_strategy_idx == 2:
            self.parallel_strategy = "SNP"
            self.reorganize_op = SNPReorganizeOp
        elif best_strategy_idx == 3:
            self.parallel_strategy = "DNP"
            self.reorganize_op = DNPReorganizeOp
        else:
            raise NotImplementedError
        global parallel_strategy
        parallel_strategy = self.parallel_strategy

        self.shared_tensor_list.append(cache_feat_node_idx)
        self.args.system = self.parallel_strategy
        load_partition(self.args, self.shared_tensor_list)

        adjusted_ddp_model = self._adjust_ddp_model(model, self.parallel_strategy)

        return adjusted_ddp_model

    def _adjust_ddp_model(self, model, parallel_strategy):
        if parallel_strategy == "GDP":
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        elif parallel_strategy == "NFP":
            model.layers[0] = MPSAGEConv(self.args.mp_input_dim_list[self.args.rank], model.hid_size, "mean").to(torch.cuda.current_device())
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        elif parallel_strategy == "SNP":
            model.layers[0] = SPSAGEConv(model.in_size, model.hid_size, "mean").to(torch.cuda.current_device())
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        elif parallel_strategy == "DNP":
            return torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        else:
            raise NotImplementedError

    def reorganize(self, samples):
        return self.reorganize_op(samples, self.args)

    def fetch_features(self, sample_results: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.labels[sample_results[1]], load_subtensor(self.args, sample_results)


def build_planner(args, hardware_info, shared_tensor_list):
    return Planner(args, hardware_info, shared_tensor_list)
