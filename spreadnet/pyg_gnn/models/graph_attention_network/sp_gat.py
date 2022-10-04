"""Basic GAT model.

Usage:
    python train.py [--config config_file_path]

@Time    : 10/03/2022 2:05 PM
@Author  : Haoyuan Li
"""

from typing import Optional, Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SPGATNet(torch.nn.Module):
    def __init__(
            self,
            num_hidden_layers: int,
            in_channels: Union[int, Tuple[int, int]],
            hidden_channels: int,
            out_channels: int,
            heads: int = 1,
            dropout: float = 0.6,
            concat: bool = False
    ):
        """

                Args:
                    num_hidden_layers:
                    in_channels: Size of each input sample.
                    hidden_channels: Size of the first hidden layer output sample.
                    out_channels: Size of each output sample.
                    heads: Number of multi-head-attentions.
                    dropout:
                    concat:
                """
        super(SPGATNet, self).__init__()
        self.conv = nn.Sequential()

        if in_channels is not None:
            sizes = [in_channels] + [hidden_channels] * num_hidden_layers + [out_channels]
        else:
            raise ValueError

        for i in range(len(sizes) - 1):
            self.conv.append(
                GATConv(
                    in_channels=sizes[i], out_channels=sizes[i+1], heads=heads, concat=concat, dropout=dropout
                ),
            )

    def forward(self, x, edge_index, edge_attr):
        for _, gat_layer in enumerate(self.conv):
            x = F.dropout(x, p=0.6, training=self.training)
            x = gat_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.elu(x)
        return x