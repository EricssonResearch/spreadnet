from typing import Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

a = torch.tensor([[0.1, -0.1],
                  [0.1, -0.1],
                  [-0.1, -0.1]])
b = torch.tensor([[0.0000, 2.9999],
                  [0.7293, 0.6518],
                  [0.0000, 0.0000]])

c = torch.tensor([[1.0000],
                  [2.0000],
                  [1.0000]])

d = (a * b * c)
index = (0, 1)
print(type(b.size()[0]))

