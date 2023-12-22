import dgl, torch
import os
import json
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset


def prepare_data(dataset_name, world_size, output_dir="./custom_dataset"):
    dataset = AsNodePredDataset(DglNodePropPredDataset(dataset_name), save_dir="./dgl_dataset")
    graph: dgl.DGLGraph = dataset[0]

    num_classes = dataset.num_classes

    from partition import custom_partition_graph

    node_part = custom_partition_graph(
        graph,
        num_parts=world_size,
        part_method="metis",
        balance_ntypes=graph.ndata["train_mask"],
        balance_edges=True,
        num_trainers_per_machine=1,
        objtype="cut",
    )
    (pid, counts) = torch.unique(node_part, return_counts=True)
    sorted, indices = torch.sort(node_part, stable=True)
    output_path = os.path.join(output_dir, dataset_name)
    reorder_graph = dgl.reorder_graph(graph, node_permute_algo="custom", permute_config={"nodes_perm": indices})
    reorder_graph_path = os.path.join(output_path, "graph.bin")
    dgl.save_graphs(reorder_graph_path, reorder_graph)

    min_vids = torch.cumsum(counts, dim=0)
    min_vids = torch.cat((torch.LongTensor([0]), min_vids)).tolist()
    input_dim = reorder_graph.ndata["feat"].shape[1]
    json_dict = {
        "world_size": world_size,
        "num_nodes": reorder_graph.num_nodes(),
        "num_edges": reorder_graph.num_edges(),
        "graph_path": reorder_graph_path,
        "min_vids": min_vids,
        "input_dim": input_dim,
        "num_classes": num_classes,
    }
    json_path = os.path.join(output_path, "configs.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    prepare_data(dataset_name="ogbn-products", world_size=4)
