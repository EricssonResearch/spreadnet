"""Basic GAT model.

Usage:
    python train.py [--config config_file_path]

@Time    : 10/03/2022 2:05 PM
@Author  : Haoyuan Li
"""

from typing import Union, Tuple
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
            dropout: float = 0.5,
            concat: bool = False,
            add_self_loops: bool = True,
            bias: bool = True
    ):
        """

                Args:
                    num_hidden_layers:
                    in_channels: Size of each input sample.
                    hidden_channels: Size of the first hidden layer output sample.
                    out_channels: Size of each output sample.
                    heads: Number of multi-head-attentions.
                    dropout: Dropout probability of the normalized attention coefficients
                    concat: If 'Flase', the multi-head attentions are averaged instead of concatenated.
                    add_self_loops: If 'True', add self loops to input graph
                    bias: If 'True', the layer will learn an additive bias
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
                    in_channels=sizes[i],
                    out_channels=sizes[i + 1],
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                    bias=bias
                ),
            )

    def forward(self, x, edge_index, edge_attr):
        for _, gat_layer in enumerate(self.conv):
            src, dst = edge_index
            x = F.dropout(x, p=0.6, training=self.training)
            x = gat_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
            score = self.edge_update(x, src, dst, edge_attr)
            x = F.leaky_relu_(x)
            score = torch.sigmoid(score)
        return x, score

    def edge_update(self, x, alpha_sct, alpha_dst, edge_attr):
        alpha = (x[alpha_sct] * x[alpha_dst] * edge_attr).sum(dim=-1)
        return alpha
