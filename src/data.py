import numpy as np
import torch
from torch.utils.data import Dataset


def array_tensor(array, precision=torch.float32):
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
        return tensor.to(precision)
    else:
        return array


class Datahandler(Dataset):
    def __init__(self, x_branch_, x_trunk_, y_, precision=torch.float32):
        x_branch = array_tensor(x_branch_, precision=precision)
        x_trunk = array_tensor(x_trunk_, precision=precision)
        y = array_tensor(y_, precision=precision)

        self.x_branch = x_branch
        self.x_trunk = x_trunk
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x_branch[index, :], self.x_trunk, self.y[index, :]
