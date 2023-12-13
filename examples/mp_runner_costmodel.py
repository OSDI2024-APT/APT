import torch
import torch.multiprocessing as mp
import utils
import atexit
import importlib


def get_nl(args):
    if "papers" in args.configs_path:
        num_localnode_feats_in_workers = [13.25, 14.25, 15.25, 16.25, 17.25]
    elif "friendster" in args.configs_path:
        num_localnode_feats_in_workers = [15.65, 17.65, 19.65, 21.65, 23.65, 25.65, 27.65, 29.65, 31.65, 33.65]
    elif "igbfull" in args.configs_path:
        num_localnode_feats_in_workers = [32.11, 33.11, 34.11, 35.11, 36.11]
    else:
        print(f"[Note]args.configs_path:{args.configs_path}")
        num_localnode_feats_in_workers = [0.1 * i for i in range(1, 10)]
    return num_localnode_feats_in_workers


if __name__ == "__main__":
    args, shared_tensor_list, global_nfeat = utils.pre_spawn()
    # we do not need labels
    shared_tensor_list.pop(0)
    # manually set cross machine feat load
    args.cross_machine_feat_load = False
    args.nl = get_nl(args)
    print(f"[Note]args.nl:{args.nl}")
    model_list = ["SAGE"]
    while True:
        # press any key to start a new run
        user_args = input("Please input the args:").split(";")
        dryrun_module = importlib.import_module("dryrun_costmodel")
        run = dryrun_module.run
        for model in model_list:
            args.model = model
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
