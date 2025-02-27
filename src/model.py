import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes,
        activation=nn.Tanh(),
        dropout_rate=0.0,
        output_activation=None,
        bias=True,
    ):
        """
        Multilayer Perceptron with configurable layers

        Args:
            layer_sizes (list): List of integers specifying the size of each layer.
                                First element is input size, last element is output size
            activation (nn.Module): Activation function to use between layers (default: Tanh)
            dropout_rate (float): Dropout probability for regularization (default: 0.0)
            output_activation (nn.Module, optional): Activation function to apply to the output layer.
                                                     If None, no activation is applied (default: None)
        """
        super(MLP, self).__init__()

        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

            # Add activation for hidden layers
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            # Add activation for output layer if specified
            elif output_activation is not None:
                layers.append(output_activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output from the network
        """
        return self.model(x)


class DeepONet(nn.Module):
    def __init__(
        self,
        branch_layer_sizes,
        trunk_layer_sizes,
        activation=nn.Tanh(),
        dropout_rate=0.0,
        output_activation=None,
        bias=True,
    ):
        """
        Deep Operator Network (DeepONet) implementation

        Args:
            branch_layer_sizes (list): List of integers specifying the size of each layer in branch network.
                                       First element is input size, last element is output size (p)
            trunk_layer_sizes (list): List of integers specifying the size of each layer in trunk network.
                                      First element is input size, last element is output size (p)
            activation (nn.Module): Activation function to use between layers (default: Tanh)
            dropout_rate (float): Dropout probability for regularization (default: 0.0)
            output_activation (nn.Module, optional): Activation function to apply to final outputs
            bias (bool): Whether to add a bias term to the dot product (default: True)
        """
        super(DeepONet, self).__init__()

        # Check that branch and trunk networks output the same dimension
        if branch_layer_sizes[-1] != trunk_layer_sizes[-1]:
            raise ValueError(
                "Branch and trunk networks must have the same output dimension"
            )

        # Branch network (processes the input function)
        self.branch_net = MLP(
            branch_layer_sizes,
            activation=activation,
            dropout_rate=dropout_rate,
            bias=bias,
        )

        # Trunk network (processes the evaluation locations)
        self.trunk_net = MLP(
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
        # Common use cases would include:
        # output_activation=nn.Sigmoid() for outputs in the range [0,1]
        # output_activation=nn.Tanh() for outputs in the range [-1,1]
        # output_activation=nn.ReLU() for non-negative outputs

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
            output = torch.bmm(trunk_output, branch_reshaped.transpose(1, 2))  # [batch_size, n_points, 1]
            output = output.squeeze(-1)  # [batch_size, n_points]

        # Add bias if specified
        if self.bias is not None:
            output = output + self.bias

        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output