import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from equiadapt.common.basecanonicalization import ContinuousGroupCanonicalization
from channels_egnn.qm9.models import EGNN
import wandb
from torch_geometric.utils import to_dense_batch
import matplotlib.pyplot as plt
from qm9.visualizer import plot_molecule

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

        dense_x, _ = to_dense_batch(x, batch)
        rotation_matrix = self.modified_gram_schmidt(rotation_vectors, device=None, coords = dense_x)

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
        
        canonical_loc = torch.bmm(x[:, None, :], rotation_matrix_inverse[batch]).squeeze()

        return canonical_loc

    def modified_gram_schmidt(self, vectors: torch.Tensor, device = None, coords = None) -> torch.Tensor:
        """
        Apply the modified Gram-Schmidt process to the input vectors.

        Args:
            vectors: Input vectors.

        Returns:
            The orthonormalized vectors.

        """

        # GRAM SCHMIDT THROWS NANS

        v1 = vectors[:, :, 0]
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True)
        v2 = vectors[:, :, 1] - torch.sum(vectors[:, :, 1] * v1, dim=1, keepdim=True) * v1
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True)
        v3 = torch.cross(v1, v2) # replaces gram-schmidt

        if torch.isnan(v1.max()) or torch.isnan(v2.max()) or torch.isnan(v3.max()):
            # print("SHAPES")
            # print(v1.shape)
            # print(v2.shape)
            # print(v3.shape)

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
            print(vectors[i, :, 0])
            print(vectors[i, :, 1])
            
            # fig = plt.figure(figsize = (10, 7))
            # ax = plt.axes(projection ="3d")

            # x = coords[i, :, 0].detach().cpu().numpy()
            # y = coords[i, :, 1].detach().cpu().numpy()
            # z = coords[i, :, 2].detach().cpu().numpy()

            # ax.scatter3D(x, y, z, color = "green")            
            # plt.savefig('problem.png')
            
            import sys
            sys.exit()


        return torch.stack([v1, v2, v3], dim=1)