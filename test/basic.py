# %%
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import solaris as sr

import torch
# %%
m = sr.MLP([3, 4, 1])
m(torch.rand(10, 3))
# %%