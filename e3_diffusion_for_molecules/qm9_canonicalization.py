import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from channels_egnn.qm9.models import EGNN
import wandb
from torch_geometric.utils import to_dense_batch
import matplotlib.pyplot as plt
from qm9.visualizer import plot_molecule

def get_dense_adj(n_nodes, batch_size, device): # creates a dense adjacency matrix for QM9
    rows, cols = [], []
    cumulative_nodes = 0
    for batch_idx in range(batch_size):
        n_nodes_batch = n_nodes[batch_idx]
        for i in range(n_nodes_batch):
            for j in range(n_nodes_batch):
                rows.append(i + cumulative_nodes)
                cols.append(j + cumulative_nodes)

        cumulative_nodes += n_nodes_batch
       
    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
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

class SO3_QM9(ContinuousGroupCanonicalization):
    """
    A class representing the continuous group for QM9.

    Args:
        canonicalization_network (torch.nn.Module): The canonicalization network.

    Attributes:
        canonicalization_info_dict (dict): A dictionary containing the group element information.

    """

    def __init__(
        self,
        canonicalization_network: torch.nn.Module,
    ) -> None:
        super().__init__(canonicalization_network)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor, 
        edge_attr: torch.Tensor, 
        node_mask: torch.Tensor, 
        edge_mask: torch.Tensor,
        n_nodes: int,
        batch: torch.Tensor,
        targets: Optional[List] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        
        return self.canonicalize(h, x, edges, edge_attr, node_mask, edge_mask, n_nodes, batch, None, **kwargs)

    def get_groupelement(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor, 
        edge_attr: torch.Tensor, 
        node_mask: torch.Tensor, 
        edge_mask: torch.Tensor,
        n_nodes: int,
        batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        group_element_dict: Dict[str, torch.Tensor] = {}

        rotation_vectors = self.canonicalization_network(
            h, x, edges, edge_attr, node_mask, edge_mask, n_nodes, batch = batch
        )

        # wandb.log({"Max Rot Vector": rotation_vectors.max()}, commit=False)

        dense_x, _ = to_dense_batch(x, batch)
        rotation_matrix = self.modified_gram_schmidt(rotation_vectors, dense_x)
        # wandb.log({"Max Rot Matrix": rotation_matrix.max()}, commit=False)

        # Check whether canonicalization_info_dict is already defined
        if not hasattr(self, "canonicalization_info_dict"):
            self.canonicalization_info_dict = {}

        group_element_dict["rotation_matrix"] = rotation_matrix
        group_element_dict["rotation_matrix_inverse"] = rotation_matrix.transpose(
            1, 2
        )  # Inverse of a rotation matrix is its transpose.

        self.canonicalization_info_dict["group_element"] = group_element_dict

        return group_element_dict

    def canonicalize(
        self, h: torch.Tensor, 
        x: torch.Tensor, 
        edges: torch.Tensor, 
        edge_attr: torch.Tensor, 
        node_mask: torch.Tensor, 
        edge_mask: torch.Tensor,
        n_nodes: int,
        batch: torch.Tensor,
        targets: Optional[List] = None, 
        **kwargs: Any
    ) -> torch.Tensor:

        group_element_dict = self.get_groupelement(
            h, x, edges, edge_attr, node_mask, edge_mask, n_nodes, batch
        )
        rotation_matrix_inverse = group_element_dict["rotation_matrix_inverse"]

        # Canonicalizes coordinates by rotating node coordinates.
        # Shape: (n_nodes * batch_size) x coord_dim.

        # wandb.log({"Max Inv Rot Matrix": rotation_matrix_inverse.max()}, commit=False)
        
        canonical_loc = torch.bmm(x[:, None, :], rotation_matrix_inverse[batch]).squeeze()

        # wandb.log({"Max Canonical Loc": canonical_loc.max()}, commit=False)

        return canonical_loc

    def modified_gram_schmidt(self, vectors: torch.Tensor, coords = None) -> torch.Tensor:
        """
        Apply the modified Gram-Schmidt process to the input vectors.

        Args:
            vectors: Input vectors.

        Returns:
            The orthonormalized vectors.

        """

        # GRAM SCHMIDT THROWS NANS

        # going to try to noise the output vectors slightly

        v1 = vectors[:, :, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, :, 1] - torch.sum(vectors[:, :, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = torch.cross(v1, v2) # replaces gram-schmidt

        if torch.isnan(v1.max()) or torch.isnan(v2.max()) or torch.isnan(v3.max()):
            print("SHAPES")
            print(v1.shape)
            print(v2.shape)
            print(v3.shape)

            if torch.isnan(v1.max()):
                print('v1 error')
                i = torch.argmax(v1.sum(1))
            elif torch.isnan(v2.max()):
                print('v2 error')
                i = torch.argmax(v2.sum(1))
            elif torch.isnan(v3.max()):
                print('v3 error')
                i = torch.argmax(v3.sum(1))

            print("IDENTIFYING PROBLEM MOLECULE") # these is an input where all of these are 0
            print(i)
            torch.save(i, "problem.pt")
            print(vectors[i, :, 0])
            print(vectors[i, :, 1])
            print(vectors[i, :, 2])

            fig = plt.figure(figsize = (10, 7))
            ax = plt.axes(projection ="3d")

            x = coords[i, :, 0].cpu().numpy()
            y = coords[i, :, 1].cpu().numpy()
            z = coords[i, :, 2].cpu().numpy()

            ax.scatter3D(x, y, z, color = "green")            
            plt.savefig('problem3.png')

            # print("V1")
            # v1 = vectors[:, 0]
            # print(v1) # SOMETIMES THIS OUTUPUTS ZERO... why?
            # print(torch.norm(v1, dim=1, keepdim=True))
            # v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
            # print(v1)

            

            # print("V2")
            # print(vectors[:, 1])

            # v2 = vectors[:, 1] - torch.sum(vectors[:, 1] * v1, dim=1, keepdim=True) * v1 # sometimes these are equal and this returns 0, causing numerical error later on, why?
            # print(v2)
            # print(torch.norm(v2, dim=1, keepdim=True))

            # v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
            # print(v2)

            # print("V3")
            # print(vectors[:, 2])
            # v3 = vectors[:, 2] - torch.sum(vectors[:, 2] * v1, dim=1, keepdim=True) * v1 # sometimes these are equal and this returns 0, why?
            # print(v3)
            # v3 = v3 - torch.sum(v3 * v2, dim=1, keepdim=True) * v2
            # print(v3)
            # print(torch.norm(v3, dim=1, keepdim=True))
            # v3 = v3 / torch.norm(v3, dim=1, keepdim=True)
            # print(v3)


        return torch.stack([v1, v2, v3], dim=1)