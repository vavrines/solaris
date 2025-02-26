import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import DeepONet
from src.data import DonDataset, array_tensor


def sci_train(
    model,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module = nn.MSELoss(),
    optimizer=None,
    lr: float = 0.001,
    epochs: int = 100,
    device: str = "cpu",
    log: bool = True,
):
    """
    Generic training function for scientific ML models

    Parameters:
        model      (nn.Module) : any PyTorch model
        dataloader (DataLoader): PyTorch dataloader containing (input, target) pairs
        criterion  (nn.Module) : loss function
        optimizer  (optim)     : PyTorch optimizer (defaults to Adam if None)
        lr        (float)     : learning rate for optimizer
        epochs    (int)       : number of training epochs
        device    (str)       : device to run training on ('cpu' or 'cuda')
        log       (bool)      : whether to print training progress
    """
    model = model.to(device)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []

        for batch in dataloader:
            x, y = batch[:-1], batch[-1]
            if len(x) == 1:
                x = x[0]

            if isinstance(x, torch.Tensor):
                x = x.to(device)
            else:
                x = [t.to(device) for t in x]
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x) if isinstance(x, torch.Tensor) else model(*x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        if log:
            print("=" * 30)
            print(f"loss at epoch {epoch+1}: {avg_loss:.6f}")
            print("=" * 30)

    return model


def train_don(
    model: DeepONet,
    x_branch,
    x_trunk,
    y_,
    lr: float = 0.001,
    batch_size=32,
    epochs=100,
    log=True,
):
    """
    trains a deep operator network

    Parameters:
        model    (DeepONet)     : the network to be trained
        x_branch (torch.tensor) : the branch input data
        x_trunk  (torch.tensor) : the trunk input data
        y        (torch.tensor) : the targets
    """

    # creating DonDataset#
    dataset = DonDataset(x_branch, x_trunk, y_)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # using standard MSE loss#
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # getting trunk data#
    trunk = array_tensor(x_trunk)

    # creating training loop#
    for epoch in range(epochs):
        losses = []
        for branch, _, y in dataloader:

            # removing previous gradients#
            optimizer.zero_grad()

            # forward pass through model#
            output = model.forward(branch, trunk)
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # calculate avg loss across batches#
            losses.append(loss.item())

            # Update parameters
            optimizer.step()

        avg_loss = np.mean(losses)

        if log == True:
            print("=" * 30)
            print(f"loss at epoch {epoch+1}:{avg_loss}")
            print("=" * 30)
