import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

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
    multi_gpu: bool = False,
    log: bool = True,
    save_best: bool = False,
    checkpoint_dir: str = "./checkpoints",
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
        save_checkpoints (bool): whether to save model checkpoints during training
        checkpoint_dir (str)  : directory to save checkpoints
        checkpoint_interval (int): save checkpoint every N epochs
    """
    model = model.to(device)
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # multiple GPUs

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create checkpoint directory if saving checkpoints
    if save_best:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # To store the best model (lowest loss)
    best_loss = float("inf")

    print("=" * 50)
    print(f"training: {epochs} epochs, device: {device}")
    print("=" * 50)

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

        # Save best model state
        if save_best and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
        if log:
            print(f"loss at epoch {epoch+1}: {avg_loss:.6f}")

    # Optionally restore best model
    # model.load_state_dict(best_model_state)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

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
            print("=" * 50)
            print(f"loss at epoch {epoch+1}:{avg_loss}")
            print("=" * 50)
