"""Implementation of Graph Convolutional Network.

@Time    : 10/1/2022 1:54 PM
@Author  : Haodong Zhao
"""

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hidden_layers: int,
        hidden_channels: int,
        out_channels: int,
        use_normalization: bool = True,
        use_bias: bool = True,
    ):
        super(GCNet, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.gcn_stack = torch.nn.Sequential()
        sizes = [in_channels] + [hidden_channels] * num_hidden_layers + [out_channels]

        for i in range(len(sizes) - 1):
            self.gcn_stack.append(
                GCNConv(
                    in_channels=sizes[i],
                    out_channels=sizes[i + 1],
                    add_self_loops=True,
                    normalize=use_normalization,
                    bias=use_bias,
                )
            )

    def forward(self, x, edge_index, edge_weight):
        for i, gcn_layer in enumerate(self.gcn_stack):
            x = F.dropout(x, p=0.5, training=self.training)
            if i == len(self.gcn_stack) - 1:
                x = gcn_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
            else:
                x = gcn_layer(
                    x=x, edge_index=edge_index, edge_weight=edge_weight
                ).relu()

        return x
