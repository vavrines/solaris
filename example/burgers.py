# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import solaris as sr

import numpy as np
import torch
from scipy.io import loadmat

vars = loadmat('burgers_data_R10.mat')
# %%
subsample = 2**3
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
dataloader = torch.utils.data.DataLoader(
    data, batch_size=10, shuffle=True
)
# %%
branch_sizes = [nsensors, 1024, 1024, 1024]
trunk_sizes = [1, 1024, 1024, 1024]
model = sr.DeepONet(branch_sizes, trunk_sizes)
# %%
model = sr.sci_train(model, dataloader, lr=1e-4, epochs=500,
    device="cuda", save_best=True, log=True)
# %%
model.to("cpu")
sol0 = model(*data.x)
# %%
import matplotlib.pyplot as plt
# %%
sol = sol0.detach().numpy()
# %%
idx = 1
plt.plot(sol[idx, :])
plt.plot(ytrain[idx, :])
# %%
torch.save(model.state_dict(), 'checkpoints/final_model.pth')
# %%
