# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import solaris as sr

import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class TransformerBranchNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_layers=4,
        num_heads=4,
        dim_feedforward=128,
        dropout=0.0,
    ):
        """
        Transformer-based network for processing input functions in DeepONet

        Args:
            input_dim (int): Size of input function representation
            output_dim (int): Size of output features (p)
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dim_feedforward (int): Dimension of feedforward network in transformer
            dropout (float): Dropout rate
        """
        super(TransformerBranchNet, self).__init__()

        # Embedding layer to project input to transformer dimension
        hidden_dim = num_heads * 16  # Must be divisible by num_heads
        self.embedding = nn.Linear(1, hidden_dim)

        # Positional encoding
        self.register_buffer(
            "pos_encoding", self._get_positional_encoding(input_dim, hidden_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _get_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding for transformer"""
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pos_enc = torch.zeros(1, seq_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(pos * div_term)
        pos_enc[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_enc[:, :seq_len]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
        Returns:
            torch.Tensor: Output features [batch_size, output_dim]
        """
        batch_size = x.shape[0]

        # Reshape to sequence format and embed
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        x = self.embedding(x)  # [batch_size, input_dim, hidden_dim]

        # Add positional encoding
        pos_enc = self.pos_encoding[:, : x.size(1)]
        x = x + pos_enc

        # Apply transformer
        x = self.transformer(x)  # [batch_size, input_dim, hidden_dim]

        # Global pooling and projection
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        x = self.output_proj(x)  # [batch_size, output_dim]

        return x


class VisONet(nn.Module):
    def __init__(
        self,
        branch_layer_sizes,
        trunk_layer_sizes,
        activation=nn.Tanh(),
        dropout_rate=0.0,
        output_activation=None,
        bias=True,
        transformer_layers=4,
        transformer_heads=4,
    ):
        """
        Deep Operator Network (DeepONet) implementation with optional Vision Transformer branch

        Args:
            branch_layer_sizes (list): List of integers specifying the size of each layer in branch network.
                                       First element is input size, last element is output size (p)
            trunk_layer_sizes (list): List of integers specifying the size of each layer in trunk network.
                                      First element is input size, last element is output size (p)
            activation (nn.Module): Activation function to use between layers (default: Tanh)
            dropout_rate (float): Dropout probability for regularization (default: 0.0)
            output_activation (nn.Module, optional): Activation function to apply to final outputs
            bias (bool): Whether to add a bias term to the dot product (default: True)
            use_transformer_branch (bool): Whether to use transformer for branch network (default: True)
            transformer_layers (int): Number of transformer layers if using transformer branch
            transformer_heads (int): Number of attention heads if using transformer branch
        """
        super(VisONet, self).__init__()

        # Check that branch and trunk networks output the same dimension
        if branch_layer_sizes[-1] != trunk_layer_sizes[-1]:
            raise ValueError(
                "Branch and trunk networks must have the same output dimension"
            )

        # Branch network (processes the input function)
        self.branch_net = TransformerBranchNet(
            input_dim=branch_layer_sizes[0],
            output_dim=branch_layer_sizes[-1],
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout_rate,
        )

        # Trunk network (processes the evaluation locations)
        self.trunk_net = sr.MLP(
            trunk_layer_sizes,
            activation=activation,
            dropout_rate=dropout_rate,
            bias=bias,
        )

        # Optional bias parameter
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))  # requires_grad=True by default

        # Store output activation
        self.output_activation = output_activation

    def forward(self, u, y):
        """
        Forward pass through the DeepONet

        Args:
            u (torch.Tensor): Input function values at sensor points, shape [batch_size, branch_input_dim]
            y (torch.Tensor): Coordinates where output function is evaluated, shape [n_points, trunk_input_dim]
                              or shape [batch_size, n_points, trunk_input_dim]

        Returns:
            torch.Tensor: Output function values at coordinates y
        """
        # Process input function through branch network
        branch_output = self.branch_net(u)  # [batch_size, p]

        # Process coordinates through trunk network
        trunk_output = self.trunk_net(y)  # [n_points, p] or [batch_size, n_points, p]

        # Handle different tensor dimensions
        if trunk_output.dim() <= 2:
            # 2D case: exactly equivalent to original branch_output @ trunk_output.t()
            output = branch_output @ trunk_output.t()
        else:
            # 3D case: use batch matrix multiplication for 3D tensors
            branch_reshaped = branch_output.unsqueeze(1)  # [batch_size, 1, p]
            # Batch matrix multiplication
            output = torch.bmm(
                trunk_output, branch_reshaped.transpose(1, 2)
            )  # [batch_size, n_points, 1]
            output = output.squeeze(-1)  # [batch_size, n_points]

        # Add bias if specified
        if self.bias is not None:
            output = output + self.bias

        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


# %%
from scipy.io import loadmat

vars = loadmat("burgers_data_R10.mat")
# %%
subsample = 2 ** 3
ntrain = 1000
ntest = 100

xtrain = vars["a"][0:ntrain, ::subsample]
ytrain = vars["u"][0:ntrain, ::subsample]
# %%
nsensors = xtrain.shape[1]
grid = np.linspace(0, 1, nsensors)
grid = grid.reshape(-1, 1)
# %%
data = sr.GeneralDataset((xtrain, grid), ytrain)
dataloader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
# %%
branch_sizes = [nsensors, 1024]
trunk_sizes = [1, 1024, 1024, 1024]

model = VisONet(
    branch_sizes,
    trunk_sizes,
    activation=nn.Tanh(),
    dropout_rate=0.0,
    output_activation=None,
    bias=True,
    transformer_layers=3,
    transformer_heads=4,
)
# %%
model = sr.sci_train(
    model,
    dataloader,
    lr=1e-4,
    epochs=100,
    device="cuda",
    multi_gpu=True,
    save_best=True,
    log=True,
)
# %%
model.to("cpu")
sol0 = model(*data.x)
# %%
sol = sol0.detach().numpy()
# %%
idx = 2
plt.plot(sol[idx, :])
plt.plot(ytrain[idx, :])
# %%
# torch.save(model.state_dict(), "checkpoints/final_model.pth")
