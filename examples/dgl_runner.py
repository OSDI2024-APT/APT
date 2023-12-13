import dgl
import torch
import argparse
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
import torch.multiprocessing as mp
import os

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
        "--dataset_name",
        type=str,
        default="ogbn-products",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Root directory of dataset.",
    )
    args = parser.parse_args()
    print(args)
    nprocs = args.world_size
    assert torch.cuda.is_available(), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # Load and preprocess the dataset.
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset_name), save_dir="./dgl_dataset")
    g: dgl.DGLGraph = dataset[0]
    feat: torch.Tensor = g.ndata["feat"]
    label: torch.Tensor = g.ndata["label"].long().nan_to_num()
    global_train_mask = g.ndata["train_mask"].bool()
    global_train_nid = torch.masked_select(torch.arange(g.num_nodes()), global_train_mask)
    print(torch.eq(global_train_nid, dataset.train_idx).all())
    feat.share_memory_()
    label.share_memory_()
    g.ndata.clear()
    g.edata.clear()

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.formats("csc")
    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )
    # press any key to run
    import importlib

    run_module = importlib.import_module("distributed_training_dgl")
    while True:
        anykey = input("Press any key to run...")
        print(f"[Note]key:{anykey} is pressed, start running...")
        # To use DDP with n GPUs, spawn up n processes.
        # reimport run function
        run = importlib.reload(run_module).run
        mp.spawn(
            run,
            args=(nprocs, g, data, args.mode, args.num_epochs, feat, label),
            nprocs=nprocs,
        )
