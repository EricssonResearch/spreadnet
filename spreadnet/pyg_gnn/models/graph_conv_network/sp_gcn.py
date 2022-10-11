"""Implementation of Graph Convolutional Network.

@Time    : 10/1/2022 1:54 PM
@Author  : Haodong Zhao
"""

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, MLP


class SPGCNet(torch.nn.Module):
    def __init__(
        self,
        node_gcn_in_channels: int,
        node_gcn_num_hidden_layers: int,
        node_gcn_hidden_channels: int,
        node_gcn_out_channels: int,
        node_gcn_use_normalization: bool,
        node_gcn_use_bias: bool,
        edge_mlp_in_channels: int,
        edge_mlp_bias: bool,
        edge_mlp_hidden_channels: int,
        edge_mlp_num_layers: int,
        edge_mlp_out_channels: int,
    ):
        super(SPGCNet, self).__init__()

        self.node_classifier = GCNStack(
            in_channels=node_gcn_in_channels,
            num_hidden_layers=node_gcn_num_hidden_layers,
            hidden_channels=node_gcn_hidden_channels,
            out_channels=node_gcn_out_channels,
            use_normalization=node_gcn_use_normalization,
            use_bias=node_gcn_use_bias,
        )

        self.edge_classifier = MLP(
            in_channels=edge_mlp_in_channels,  # node + node + edge_attr
            out_channels=edge_mlp_out_channels,
            bias=edge_mlp_bias,
            hidden_channels=edge_mlp_hidden_channels,
            num_layers=edge_mlp_num_layers,
        )

    def forward(self, x, edge_index, edge_weight):
        x = self.node_classifier(x, edge_index, edge_weight)
        node_embeds = x.relu()
        node_embeds_src, node_embeds_dst = (
            node_embeds[edge_index[0]],
            node_embeds[edge_index[1]],
        )
        edge_feat = torch.cat([node_embeds_src, edge_weight, node_embeds_dst], dim=-1)
        out = self.edge_classifier(edge_feat)

        return x, out


class GCNStack(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hidden_layers: int,
        hidden_channels: int,
        out_channels: int,
        use_normalization: bool = True,
        use_bias: bool = True,
    ):
        super(GCNStack, self).__init__()

        # node classification
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
