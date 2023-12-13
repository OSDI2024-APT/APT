import argparse
import os
import time
import utils
import dgl
import dgl.nn as dglnn
import json
from dgl.utils import gather_pinned_tensor_rows
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    # NeighborSampler,
)
from copy_sampler import NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel


def pin_tensor(tensor):
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0)


def get_time():
    torch.cuda.synchronize()
    dist.barrier()
    return time.time()


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def evaluate(model, g, num_classes, dataloader, feat, label):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            # x = blocks[0].srcdata["feat"]
            # ys.append(blocks[-1].dstdata["label"])
            x = gather_pinned_tensor_rows(feat, input_nodes)
            ys.append(gather_pinned_tensor_rows(label, output_nodes))
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def train(
    proc_id,
    nprocs,
    device,
    g,
    num_classes,
    train_idx,
    val_idx,
    model,
    use_uva,
    num_epochs,
    feat,
    label,
):
    acc_file_path = f"./logs/accuracy/acc.csv"
    if proc_id == 0:
        acc_file = open(acc_file_path, "w")
    # Instantiate a neighbor sampler
    # sampler = NeighborSampler([10, 10, 10], prefetch_node_feats=["feat"], prefetch_labels=["label"], replace=True)
    val_flag = val_idx is not None
    sampler = NeighborSampler([10, 10, 10], replace=True)
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        use_uva=use_uva,
    )
    print(f"number of training batches:{len(train_dataloader)}")
    if val_flag:
        val_dataloader = DataLoader(
            g,
            val_idx,
            sampler,
            device=device,
            batch_size=1024,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            use_uva=use_uva,
        )
        print(f"number of validation batches:{len(val_dataloader)}")
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    print(f"[Note]model.training:{model.training}")
    for epoch in range(num_epochs):
        t0 = get_time()
        last_time = t0
        model.train()
        total_loss = 0
        total_sampling_time = 0
        total_loading_time = 0
        total_training_time = 0
        num_nodes_per_layer = [0, 0, 0]
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            for i in range(3):
                num_nodes_per_layer[i] += blocks[i].num_src_nodes()
            # x = blocks[0].srcdata["feat"]
            # y = blocks[-1].dstdata["label"].to(torch.int64)
            tx = get_time()
            x = gather_pinned_tensor_rows(feat, input_nodes)
            y = gather_pinned_tensor_rows(label, output_nodes)
            ty = get_time()
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()  # Gradients are synchronized in DDP
            total_loss += loss
            now_time = get_time()
            total_training_time += now_time - ty
            total_sampling_time += tx - last_time
            total_loading_time += ty - tx
            last_time = now_time

        dist.barrier()
        t1 = get_time()
        if val_flag:
            acc = evaluate(model, g, num_classes, val_dataloader, feat, label).to(device) / nprocs
            # Reduce `acc` tensors to process 0.
            dist.reduce(tensor=acc, dst=0)
        else:
            acc = torch.tensor(0.0)
        if proc_id == 0:
            acc_str = f"Epoch {epoch:05d} | Loss {total_loss / (it + 1):.4f} |  Accuracy {acc.item():.4f} | Sampling time {total_sampling_time:.4f}\n | Loading time {total_loading_time:.4f}\n | Training time {total_training_time:.4f} | Epoch time {t1 - t0:.4f}\n"
            print(acc_str)
            acc_file.write(acc_str)

        num_nodes_per_layer = [num_nodes_per_layer[i] / (it + 1) for i in range(3)]
        print(f"[Note]Rk#{proc_id} {it+1}: num_nodes_per_layer:{num_nodes_per_layer}")


def run(proc_id, nprocs, g, data, mode, num_epochs, feat, label):
    # device = devices[proc_id]
    device = torch.device(f"cuda:{proc_id}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="tcp://127.0.0.1:12345",
        world_size=nprocs,
        rank=proc_id,
    )
    (
        num_classes,
        train_idx,
        val_idx,
    ) = data
    print(f"num_classes: {num_classes}")

    num_train_nids_per_rank = train_idx.numel() // nprocs
    local_train_idx = train_idx[proc_id * num_train_nids_per_rank : (proc_id + 1) * num_train_nids_per_rank]
    local_train_idx = local_train_idx.to(device)

    if val_idx is not None:
        num_val_per_rank = val_idx.numel() // nprocs
        local_val_idx = val_idx[proc_id * num_val_per_rank : (proc_id + 1) * num_val_per_rank]
        local_val_idx = local_val_idx.to(device)
    else:
        local_val_idx = None

    g = g.to(device if mode == "puregpu" else "cpu")

    in_size = feat.shape[1]
    pin_tensor(feat)
    pin_tensor(label)

    model = SAGE(in_size, 16, num_classes).to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Training.
    use_uva = mode == "mixed"
    if proc_id == 0:
        print("Training...")
    train(
        proc_id,
        nprocs,
        device,
        g,
        num_classes,
        local_train_idx,
        local_val_idx,
        model,
        use_uva,
        num_epochs,
        feat,
        label,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, " "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="world_size, n_procs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs for train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="Dataset name.",
    )
    parser.add_argument(
        "--configs_path", default="/efs/khma/Projects/NPC/papers_w4_metis/configs.json", type=str, help="the path to the graph configs.json"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    print(args)

    nprocs = args.world_size
    assert torch.cuda.is_available(), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # Load and preprocess the dataset.
    print("Loading data")

    if args.debug:
        graph, val_idx = utils.load_graph(args)
        # dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset_name), save_dir="./dgl_dataset")
        # g: dgl.DGLGraph = dataset[0]
        feat: torch.Tensor = graph.ndata["feat"]
        label: torch.Tensor = graph.ndata["label"].long().nan_to_num()
        global_train_mask = graph.ndata["train_mask"].bool()
        global_train_nid = torch.masked_select(torch.arange(graph.num_nodes()), global_train_mask)
        # print(torch.eq(global_train_nid, dataset.train_idx).all())
        feat.share_memory_()
        label.share_memory_()
        graph.ndata.clear()
        graph.edata.clear()

        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.formats("csc")
    else:
        configs = json.load(open(args.configs_path))
        for key, value in configs.items():
            print(f"[Note]Set args {key} = {value}")
            setattr(args, key, value)

        graph = utils.load_graph(args)
        total_nodes = graph.num_nodes()

        feat = torch.rand((total_nodes, args.input_dim), dtype=torch.float32)
        label = torch.randint(args.num_classes, (total_nodes,))
        val_idx = None

    global_train_mask = graph.ndata["train_mask"].bool()
    global_train_nid = torch.masked_select(torch.arange(graph.num_nodes()), global_train_mask)
    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    """
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )
    """
    data = (
        args.num_classes,
        global_train_nid,
        val_idx,
    )
    # To use DDP with n GPUs, spawn up n processes.
    mp.spawn(
        run,
        args=(nprocs, graph, data, args.mode, args.num_epochs, feat, label),
        nprocs=nprocs,
    )
