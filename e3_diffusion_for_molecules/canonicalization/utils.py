import torch
from torch import nn
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from canonicalization.custom_canon_networks import EGNN
import wandb
from torch_geometric.utils import to_dense_batch
import matplotlib.pyplot as plt

def run_canonicalization(x, one_hot, charges, node_mask, device, canonicalizer):
    
    # need to first convert x and h into dense representation before doing the canonicalization
    hidden = torch.cat((one_hot, charges), 2)
    batch_size = hidden.shape[0]
    max_nodes = hidden.shape[1]

    bool_node_mask = (node_mask > 0).squeeze(-1)
    x = x[bool_node_mask]
    hidden = hidden[bool_node_mask]

    flat_node_mask = torch.reshape(node_mask, (max_nodes*batch_size, 1))
    n_nodes = node_mask.squeeze(2).sum(1).int() # gives number of nodes per batch
    batch = torch.repeat_interleave(torch.arange(batch_size).to(device), n_nodes.to(device))

    # make adjacency matrix dense
    edges = get_dense_adj(n_nodes.detach().cpu(), batch_size, device)
    x = canonicalizer(hidden, x, edges = edges, edge_attr = None, node_mask = flat_node_mask, edge_mask = None, n_nodes=n_nodes, batch = batch) # all of this should be in the sparse representation

    # convert x back to sparse representation
    x, _ = to_dense_batch(x, batch)
    
    return x

# major slowdown... what can i do about this
adj_dict = {}
def get_dense_adj(n_nodes, batch_size, device): # creates a dense adjacency matrix for QM9
    rows = []
    cols = []
    cumulative_nodes = 0
    for batch_idx in range(batch_size):
        n_nodes_graph = n_nodes[batch_idx].item()

        if n_nodes_graph in adj_dict:
            new_row, new_col = adj_dict[n_nodes_graph]

        else:
            new_row = torch.arange(n_nodes_graph).tile((n_nodes_graph,))
            new_col = torch.arange(n_nodes_graph).repeat_interleave(n_nodes_graph)
            # adj_dict[n_nodes_graph] = (new_row, new_col)

        rows.append((new_row + cumulative_nodes))
        cols.append((new_col + cumulative_nodes))
        cumulative_nodes += n_nodes_graph

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    edges = [rows.to(device), cols.to(device)]
    return edges


def get_EGNN_QM9(canonicalization_hyperparams, dataset_info, args):
    """
    A funtion returning a channel-egnn with the given hyperparams for QM9 canonicalization
    """
    nf = canonicalization_hyperparams.hidden_dim
    n_layers = canonicalization_hyperparams.num_layers
    device = canonicalization_hyperparams.device
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)

    qm9_egnn = EGNN(in_node_nf = in_node_nf, in_edge_nf = 0, hidden_edge_nf = nf, hidden_node_nf = nf, hidden_coord_nf = nf,
                act_fn=nn.SiLU(), n_layers=n_layers, device=device,
                coords_weight=1.0,attention=False, node_attr=1,
                num_vectors=1, update_coords=True)
    
    return qm9_egnn


if __name__ == '__main__':
    device = 'cpu'
    n_nodes = [2, 3, 3, 4]
    batch_size = 4
    print(get_dense_adj(n_nodes, batch_size, device))