import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.model import DeepONet
from src.data import Datahandler, array_tensor

def train_DON(
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

    # creating datahandler#
    dataset = Datahandler(x_branch, x_trunk, y_)
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
