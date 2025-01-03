import torch
import numpy as np
import matplotlib.pyplot as plt
from ipdb_hook import ipdb_sys_excepthook
ipdb_sys_excepthook()
act =  torch.distributions.pareto.Pareto(1, 10).sample((1,1024))
weights = torch.distributions.normal.Normal(0, 0.12).sample((3, 64, 7, 7)).flatten()
x