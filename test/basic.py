# %%
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import solaris as sr

import torch

# %%
model = sr.MLP([3, 4, 1])
model(torch.rand(10, 3))
# %%
x = torch.randn(10, 3)
y = torch.randn(10, 1)
data = sr.GeneralDataset(x, y)

sr.sci_train(
    model,
    data,
)
# %%
