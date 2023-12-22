"""Functions for partitions. """

import time
import numpy as np
from dgl import backend as F
from dgl.base import DGLError, EID, ETYPE, NID, NTYPE
from dgl.convert import to_homogeneous
from dgl.partition import (
    get_peak_mem,
    metis_partition_assignment,
)
from dgl.random import choice as random_choice

RESERVED_FIELD_DTYPE = {
    "inner_node": F.uint8,  # A flag indicates whether the node is inside a partition.
    "inner_edge": F.uint8,  # A flag indicates whether the edge is inside a partition.
    NID: F.int64,
    EID: F.int64,
    NTYPE: F.int16,
    # `sort_csr_by_tag` and `sort_csc_by_tag` works on int32/64 only.
    ETYPE: F.int32,
}


def _set_trainer_ids(g, sim_g, node_parts):
    """Set the trainer IDs for each node and edge on the input graph.

    The trainer IDs will be stored as node data and edge data in the input graph.

    Parameters
    ----------
    g : DGLGraph
       The input graph for partitioning.
    sim_g : DGLGraph
        The homogeneous version of the input graph.
    node_parts : tensor
        The node partition ID for each node in `sim_g`.
    """
    if g.is_homogeneous:
        g.ndata["trainer_id"] = node_parts
        # An edge is assigned to a partition based on its destination node.
        g.edata["trainer_id"] = F.gather_row(node_parts, g.edges()[1])
    else:
        for ntype_id, ntype in enumerate(g.ntypes):
            type_idx = sim_g.ndata[NTYPE] == ntype_id
            orig_nid = F.boolean_mask(sim_g.ndata[NID], type_idx)
            trainer_id = F.zeros((len(orig_nid),), F.dtype(node_parts), F.cpu())
            F.scatter_row_inplace(trainer_id, orig_nid, F.boolean_mask(node_parts, type_idx))
            g.nodes[ntype].data["trainer_id"] = trainer_id
        for c_etype in g.canonical_etypes:
            # An edge is assigned to a partition based on its destination node.
            _, _, dst_type = c_etype
            trainer_id = F.gather_row(g.nodes[dst_type].data["trainer_id"], g.edges(etype=c_etype)[1])
            g.edges[c_etype].data["trainer_id"] = trainer_id


def custom_partition_graph(
    g,
    num_parts,
    part_method="metis",
    balance_ntypes=None,
    balance_edges=False,
    return_mapping=False,
    num_trainers_per_machine=1,
    objtype="cut",
):
    assert "coo" in np.concatenate(list(g.formats().values())), "'coo' format should be allowed for partitioning graph."

    def get_homogeneous(g, balance_ntypes):
        if g.is_homogeneous:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        elif isinstance(balance_ntypes, dict):
            # Here we assign node types for load balancing.
            # The new node types includes the ones provided by users.
            num_ntypes = 0
            for key in g.ntypes:
                if key in balance_ntypes:
                    g.nodes[key].data["bal_ntype"] = F.astype(balance_ntypes[key], F.int32) + num_ntypes
                    uniq_ntypes = F.unique(balance_ntypes[key])
                    assert np.all(F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes)))
                    num_ntypes += len(uniq_ntypes)
                else:
                    g.nodes[key].data["bal_ntype"] = F.ones((g.num_nodes(key),), F.int32, F.cpu()) * num_ntypes
                    num_ntypes += 1
            sim_g = to_homogeneous(g, ndata=["bal_ntype"])
            bal_ntypes = sim_g.ndata["bal_ntype"]
            print("The graph has {} node types and balance among {} types".format(len(g.ntypes), len(F.unique(bal_ntypes))))
            # We now no longer need them.
            for key in g.ntypes:
                del g.nodes[key].data["bal_ntype"]
            del sim_g.ndata["bal_ntype"]
        else:
            sim_g = to_homogeneous(g)
            bal_ntypes = sim_g.ndata[NTYPE]
        return sim_g, bal_ntypes

    if objtype not in ["cut", "vol"]:
        raise ValueError

    if num_parts == 1:
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print("Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB".format(time.time() - start, get_peak_mem()))
        assert num_trainers_per_machine >= 1
        if num_trainers_per_machine > 1:
            # First partition the whole graph to each trainer and save the trainer ids in
            # the node feature "trainer_id".
            start = time.time()
            node_parts = metis_partition_assignment(
                sim_g,
                num_parts * num_trainers_per_machine,
                balance_ntypes=balance_ntypes,
                balance_edges=balance_edges,
                mode="k-way",
            )
            _set_trainer_ids(g, sim_g, node_parts)
            print("Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB".format(time.time() - start, get_peak_mem()))

        node_parts = F.zeros((sim_g.num_nodes(),), F.int64, F.cpu())
        parts = {0: sim_g.clone()}
        orig_nids = parts[0].ndata[NID] = F.arange(0, sim_g.num_nodes())
        orig_eids = parts[0].edata[EID] = F.arange(0, sim_g.num_edges())
        # For one partition, we don't really shuffle nodes and edges. We just need to simulate
        # it and set node data and edge data of orig_id.
        parts[0].ndata["orig_id"] = orig_nids
        parts[0].edata["orig_id"] = orig_eids
        if return_mapping:
            if g.is_homogeneous:
                orig_nids = F.arange(0, sim_g.num_nodes())
                orig_eids = F.arange(0, sim_g.num_edges())
            else:
                orig_nids = {ntype: F.arange(0, g.num_nodes(ntype)) for ntype in g.ntypes}
                orig_eids = {etype: F.arange(0, g.num_edges(etype)) for etype in g.canonical_etypes}
        parts[0].ndata["inner_node"] = F.ones(
            (sim_g.num_nodes(),),
            RESERVED_FIELD_DTYPE["inner_node"],
            F.cpu(),
        )
        parts[0].edata["inner_edge"] = F.ones(
            (sim_g.num_edges(),),
            RESERVED_FIELD_DTYPE["inner_edge"],
            F.cpu(),
        )
    elif part_method in ("metis", "random"):
        start = time.time()
        sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
        print("Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB".format(time.time() - start, get_peak_mem()))
        if part_method == "metis":
            assert num_trainers_per_machine >= 1
            start = time.time()
            if num_trainers_per_machine > 1:
                # First partition the whole graph to each trainer and save the trainer ids in
                # the node feature "trainer_id".
                node_parts = metis_partition_assignment(
                    sim_g,
                    num_parts * num_trainers_per_machine,
                    balance_ntypes=balance_ntypes,
                    balance_edges=balance_edges,
                    mode="k-way",
                    objtype=objtype,
                )
                _set_trainer_ids(g, sim_g, node_parts)

                # And then coalesce the partitions of trainers on the same machine into one
                # larger partition.
                node_parts = F.floor_div(node_parts, num_trainers_per_machine)
            else:
                node_parts = metis_partition_assignment(
                    sim_g,
                    num_parts,
                    balance_ntypes=balance_ntypes,
                    balance_edges=balance_edges,
                    objtype=objtype,
                )
            print("Assigning nodes to METIS partitions takes {:.3f}s, peak mem: {:.3f} GB".format(time.time() - start, get_peak_mem()))
        else:
            node_parts = random_choice(num_parts, sim_g.num_nodes())

        return node_parts

    else:
        raise Exception("Unknown partitioning method: " + part_method)
