import sys
import torch
from equiadapt.images.canonicalization_networks import ESCNNEquivariantNetwork
from model import MNISTDiffusion
import e2cnn
import torch.nn as nn
from e2cnn import gspaces
from unet import TimeMLP

class ESCNN_UBlock(torch.nn.Module):
    def __init__(        self,
        in_shape: tuple,
        in_channels: int,
        out_channels: int,
        padding: int,
        mode: str,
        stride: int = 1,
        kernel_size: int = 5,
        type: str = "regular",
        group_type: str = "rotation",
        num_rotations: int = 4,):
        super().__init__()

        self.in_channels = in_shape[0]
        self.channels = in_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        if type == "input":
            r1 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.trivial_repr] * in_channels))
        else:
            r1 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.regular_repr] * in_channels))

        r2 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.regular_repr] * out_channels))

        self.in_type = r1
        self.out_type = r2

        modules = [
            e2cnn.nn.R2Conv(self.in_type, self.out_type, kernel_size, padding = padding, stride= stride),
            e2cnn.nn.InnerBatchNorm(self.out_type, momentum=0.9),
            e2cnn.nn.ReLU(self.out_type, inplace=True),
            e2cnn.nn.PointwiseDropout(self.out_type, p=0.5),
        ]

        if mode == "up":
            modules[0] = e2cnn.nn.R2ConvTransposed(self.in_type, self.out_type, kernel_size, padding = padding, stride = stride)


        self.network = e2cnn.nn.SequentialModule(*modules)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ESCNNUnet(torch.nn.Module):

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        timesteps,
        kernel_size: int = 5,
        time_embedding_dim=256,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations
        self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)
        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=1) # could be a problem

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        r1 = e2cnn.nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        r2 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.regular_repr] * out_channels))

        self.in_type = r1
        self.hid_type = r2
        self.out_type = r2
        padding = (kernel_size - 1) // 2

        self.initial = ESCNN_UBlock(in_shape, self.in_channels, self.out_channels, padding= 0, type = "input", num_rotations=self.num_rotations, mode = "down")
        
        self.down1 = ESCNN_UBlock(in_shape, self.out_channels, self.out_channels * 2, padding= 0, type = "regular", num_rotations=self.num_rotations, mode = "down", stride = 1)
        self.down2 = ESCNN_UBlock(in_shape, self.out_channels * 2, self.out_channels * 4, padding= 0, type = "regular", num_rotations=self.num_rotations, mode = "down", stride = 2)

        self.mid1 = ESCNN_UBlock(in_shape, self.out_channels * 4, self.out_channels * 4, padding= padding, type = "regular", num_rotations=self.num_rotations, mode = "down", stride = 1)
        self.mid2 = ESCNN_UBlock(in_shape, self.out_channels * 4, self.out_channels * 4, padding= padding, type = "regular", num_rotations=self.num_rotations, mode = "down", stride = 1)

        self.up1 = ESCNN_UBlock(in_shape, self.out_channels * 4, self.out_channels * 2, padding= 0, type = "regular", num_rotations=self.num_rotations, mode="up", stride = 2, kernel_size = 6)
        self.up2 = ESCNN_UBlock(in_shape, self.out_channels * 2, self.out_channels, padding= 0, type = "regular", num_rotations=self.num_rotations, mode="up", stride = 1, kernel_size = 5)
        
        self.final = e2cnn.nn.R2ConvTransposed(self.hid_type, self.out_type, kernel_size)

    def forward(self, x: torch.Tensor, t = None) -> torch.Tensor:
        if t is not None:
            t = self.time_embedding(t)
            x = self.time_mlp(x,t)

        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        x_init = self.initial(x) # (128, 24, 24)

        x_down1 = self.down1(x_init) # (256, 20, 20)
        x_down2 = self.down2(x_down1) # (512, 8, 8)

        x_mid1 = self.mid1(x_down2)# (512, 8, 8)
        x_mid2 = self.mid2(x_mid1) + x_down2 # (512, 8, 8)

        x_up1 = self.up1(x_mid2) + x_down1 # output thus needs to be (256, 20, 20)

        x_up2 = self.up2(x_up1) + x_init # output thus needs to be (128, 24, 24)

        out = self.final(x_up2) # should be (128, 28, 28)
        
        feature_map = out.tensor
        pred = torch.mean(feature_map, dim=1, keepdim=True)
        return pred
    
class ESCNNDenoiser(torch.nn.Module):
    """
    This class represents an Equivariant Convolutional Neural Network (Equivariant CNN).

    The network is equivariant to a group of transformations, which can be either rotations or roto-reflections. The network consists of a sequence of equivariant convolutional layers, each followed by batch normalization, a ReLU activation function, and dropout. The number of output channels of the convolutional layers is the same for all layers.

    Methods:
        __init__: Initializes the ESCNNEquivariantNetwork instance.
        forward: Performs a forward pass through the network.
    """

    def __init__(
        self,
        in_shape: tuple,
        out_channels: int,
        timesteps,
        kernel_size: int = 5,
        time_embedding_dim=256,
        group_type: str = "rotation",
        num_rotations: int = 4,
        num_layers: int = 1,
    ):
        """
        Initializes the ESCNNEquivariantNetwork instance.

        Args:
            in_shape (tuple): The shape of the input data. It should be a tuple of the form (in_channels, height, width).
            out_channels (int): The number of output channels of the convolutional layers.
            kernel_size (int): The size of the kernel of the convolutional layers.
            group_type (str, optional): The type of the group of transformations. It can be either "rotation" or "roto-reflection". Defaults to "rotation".
            num_rotations (int, optional): The number of rotations in the group. Defaults to 4.
            num_layers (int, optional): The number of convolutional layers. Defaults to 1.
        """
        super().__init__()

        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.group_type = group_type
        self.num_rotations = num_rotations
        self.time_embedding=nn.Embedding(timesteps,time_embedding_dim)
        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=1) # could be the problem

        if group_type == "rotation":
            self.gspace = gspaces.Rot2dOnR2(num_rotations)
        elif group_type == "roto-reflection":
            self.gspace = gspaces.FlipRot2dOnR2(num_rotations)
        else:
            raise ValueError("group_type must be rotation or roto-reflection for now.")

        # If the group is roto-reflection, then the number of group elements is twice the number of rotations
        self.num_group_elements = (
            num_rotations if group_type == "rotation" else 2 * num_rotations
        )

        r1 = e2cnn.nn.FieldType(
            self.gspace, [self.gspace.trivial_repr] * self.in_channels
        )
        r2 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.regular_repr] * out_channels))
        # r3 = e2cnn.nn.FieldType(self.gspace, ([self.gspace.regular_repr] * 28 * 28))

        self.in_type = r1
        self.hid_type = r2
        self.out_type = r2
        padding = (kernel_size - 1) // 2

        modules = [
            e2cnn.nn.R2Conv(self.in_type, self.hid_type, kernel_size, padding = padding),
            e2cnn.nn.InnerBatchNorm(self.hid_type, momentum=0.9),
            e2cnn.nn.ReLU(self.hid_type, inplace=True),
            e2cnn.nn.PointwiseDropout(self.hid_type, p=0.5),
        ]
        for _ in range(num_layers - 2):
            modules.append(
                e2cnn.nn.R2Conv(self.hid_type, self.hid_type, kernel_size, padding = padding),
            )
            modules.append(
                e2cnn.nn.InnerBatchNorm(self.hid_type, momentum=0.9),
            )
            modules.append(
                e2cnn.nn.ReLU(self.hid_type, inplace=True),
            )
            modules.append(
                e2cnn.nn.PointwiseDropout(self.hid_type, p=0.5),
            )

        modules.append(
            e2cnn.nn.R2Conv(self.hid_type, self.out_type, kernel_size, padding = padding),
        )

        self.eqv_network = e2cnn.nn.SequentialModule(*modules)

    def forward(self, x: torch.Tensor, t = None) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input data. It should have the shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The output of the network. It has the shape (batch_size, num_group_elements).
        """
        if t is not None:
            t = self.time_embedding(t) # figure out whats going on here
            x = self.time_mlp(x,t)

        x = e2cnn.nn.GeometricTensor(x, self.in_type)
        out = self.eqv_network(x)

        feature_map = out.tensor

        pred = torch.mean(feature_map, dim=1, keepdim=True)

        return pred

if __name__ == "__main__":
    device = "cpu"
    img = torch.randn((1, 1, 28, 28)).to(device)
    t = torch.Tensor([[10]])
    print(t.shape)
    diff=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=16,
                dim_mults=[2,4]).to(device)
    

    escnn = ESCNNDenoiser(
                in_shape = (1, 28,28),
                out_channels=8, 
                kernel_size= 3, 
                num_layers = 3,
                group_type='rotation', 
                num_rotations=4,
                timesteps=1000 
            ).to(device)

    noise=torch.randn_like(img).to(device)
    # print(diff(img, noise).shape)
    print(escnn(img, t).shape)