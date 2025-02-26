# %%
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import solaris as sr

import numpy as np
import torch
# %%
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
nsensors = 10
ndim = 2
npoints = 32
ncases = 50
u = torch.randn(ncases, nsensors).to(device)
y = torch.randn(npoints, ndim).to(device)
sol = torch.randn(ncases, npoints).to(device)
# %%
branch_sizes = [nsensors, 40, 40, 10]  # input function represented by 20 sensors, output dimension 10
trunk_sizes = [ndim, 40, 40, 10] # 2D coordinates as input, output dimension 10
model = sr.DeepONet(branch_sizes, trunk_sizes).to(device)
# %%
sr.train_don(
    model,
    u,
    y,
    sol,
    lr = 0.001,
    batch_size=1,
    epochs=10,
    log=True,
)
# %%
# sci_train can be applied as well
data = sr.GeneralDataset((u, y), sol)
sr.sci_train(
    model,
    data
)
# %%
