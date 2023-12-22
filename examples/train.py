import apt, dgl, torch


def setup(rank, local_rank, world_size, args, backend=None):
    master_port = args.master_port
    master_addr = args.master_addr
    init_method = f"tcp://{master_addr}:{master_port}"
    torch.cuda.set_device(local_rank)
    print(f"[Note]dist setup: rank:{rank}\t world_size:{world_size}\t init_method:{init_method} \t backend:{backend}")
    torch.distributed.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    print("[Note]Done dist init")


def cleanup():
    torch.distributed.destroy_process_group()


class InjectedDGLModel(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        # Three-layer GraphSAGE-mean
        self.layers.append(dgl.nn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dgl.nn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dgl.nn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = torch.nn.Dropout(0.0)
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, sample_and_features):
        (samples, features, *rest) = sample_and_features
        for i, sample in enumerate(samples):
            features = apt.layer_barrier(i, features, rest)
            # print(f"[Note] layer:{i} features:{features.shape}")
            features = self.layers[i](sample, features)

        return features


def parallel_train(rank, args, graph, shared_tensor_list):
    # init
    device = torch.device("cuda:" + str(rank))

    args.device = device
    args.rank = rank
    world_size = args.world_size
    setup(rank, rank, world_size, args, backend="nccl")
    apt.init(rank, rank, world_size, world_size, device=device)

    model = InjectedDGLModel(args.input_dim, args.num_hidden, args.num_classes).to(device)

    val_idx = shared_tensor_list.pop()
    train_idx = shared_tensor_list.pop()

    # num_val_idx_per_rank = val_idx.numel() // world_size
    # val_nodes = val_idx[rank * num_val_idx_per_rank : (rank + 1) * num_val_idx_per_rank].to(device)
    num_train_nids_per_rank = train_idx.numel() // world_size
    train_nodes = train_idx[rank * num_train_nids_per_rank : (rank + 1) * num_train_nids_per_rank].to(device)

    dataloader = dgl.dataloading.DataLoader(
        graph,
        train_nodes,
        dgl.dataloading.NeighborSampler([10, 10, 10], replace=True),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )
    planner = apt.build_planner(args, args.hardware_info, shared_tensor_list)
    adapted_model = planner.plan(dataloader, model)
    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=0.001, weight_decay=0.0005)

    for epoch in range(args.num_epochs):
        print(f"[Note]train epoch:{epoch}")
        adapted_model.train()
        for samples in dataloader:
            samples = planner.reorganize(samples)

            y, sample_and_features = planner.fetch_features(samples)
            y_hat = adapted_model(sample_and_features)

            loss = torch.nn.functional.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def show_args(args):
    for k, v in args.__dict__.items():
        print(f"[Note]args.{k} = {v}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN with APT")

    parser.add_argument("--master_addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master_port", type=str, default="12345", help="Master port")
    parser.add_argument("--num_hidden", type=int, default=16, help="Hidden layer size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--world_size", type=int, default=..., help="Number of workers")
    parser.add_argument("--hardware-info", type=dict, default={}, help="Hardware information")
    parser.add_argument("--cache_memory", type=float, default=0.5, help="cache memory in GB")
    parser.add_argument("--model", type=str, default="SAGE", help="model name")
    parser.add_argument("--shuffle_with_dst", type=bool, default=False, help="shuffle with dst")
    parser.add_argument("--num_gpu_cache_nodes", type=int, default=0, help="num cached nodes")

    parser.add_argument("--configs_path", default="./custom_dataset/ogbn-products/configs.json", type=str, help="the path to the graph configs.json")
    # provide by graph configs
    # parser.add_argument("--out-size", type=int, default=..., help="Output feature size")
    # parser.add_argument("--in-size", type=int, default=..., help="Input feature size")
    args = parser.parse_args()
    # load args from configs
    import json, os

    if args.configs_path is not None and os.path.exists(args.configs_path):
        configs = json.load(open(args.configs_path))
        for key, value in configs.items():
            print(f"[Note] Set args.{key} = {value}")
            setattr(args, key, value)
    else:
        raise ValueError(f"Invalid configs path: {args.configs_path}")

    mp_input_dim_list = [int(args.input_dim // args.world_size) for r in range(args.world_size)]
    lef = args.input_dim % args.world_size
    for r in range(lef):
        mp_input_dim_list[r] += 1

    args.mp_input_dim_list = mp_input_dim_list
    from itertools import accumulate

    args.cumsum_feat_dim = list(accumulate([0] + mp_input_dim_list))
    # load graph
    graph = dgl.load_graphs(args.graph_path)[0][0]
    print(f"[Note] Done load graph: {graph}")
    # extract graph node features
    num_total_nodes = graph.number_of_nodes()
    global_train_mask = graph.ndata["train_mask"].bool()
    global_val_mask = graph.ndata["val_mask"].bool()
    global_train_nodes = torch.masked_select(torch.arange(num_total_nodes), global_train_mask)
    global_val_nodes = torch.masked_select(torch.arange(num_total_nodes), global_val_mask)
    gloabl_labels = graph.ndata["label"].long().nan_to_num()
    global_feats = graph.ndata["feat"]
    global_feats_idx = torch.arange(num_total_nodes)
    graph.ndata.clear()
    graph.edata.clear()
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = graph.formats("csc")
    indptr, indices, edges_ids = graph.adj_tensors("csc")
    shared_tensor_list = [
        global_feats_idx,
        global_feats,
        indptr,
        indices,
        gloabl_labels,
        global_train_nodes,
        global_val_nodes,
    ]

    for tensor in shared_tensor_list:
        tensor.share_memory_()

    # start multiprocess parallel train
    nproc = args.world_size
    ranks = list(range(nproc))
    processes = []

    def kill_proc(p):
        try:
            p.terminate()
        except Exception:
            pass

    import atexit

    torch.multiprocessing.set_start_method("spawn", force=True)
    for i in range(nproc):
        p = torch.multiprocessing.Process(target=parallel_train, args=(ranks[i], args, graph, shared_tensor_list))
        atexit.register(kill_proc, p)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
