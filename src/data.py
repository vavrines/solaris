import numpy as np
import torch
from torch.utils.data import Dataset


def array_tensor(data, precision=torch.float32):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        return tensor.to(precision)
    elif isinstance(data, (list, tuple)):
        converted = [array_tensor(x, precision) for x in data]
        return converted
    else:
        return data


class GeneralDataset(Dataset):
    def __init__(self, x_, y_, precision=torch.float32):
        x = array_tensor(x_, precision=precision)
        y = array_tensor(y_, precision=precision)

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if isinstance(self.x, (list, tuple)):
            return self.x[0][index, :], self.x[1], self.y[index, :]
        else:
            return self.x[index, :], self.y[index, :]


class DonDataset(Dataset):
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
