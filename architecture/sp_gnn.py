"""

    @Time    : 9/15/2022 11:31 PM
    @Author  : Haodong Zhao
    
"""
from typing import Tuple, Any

# %%
import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SPMLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_hidden_layers: int,
                 hidden_size: int,
                 output_size: int,
                 use_layer_norm: bool = True,
                 activation=torch.nn.ReLU,
                 activate_final=torch.nn.Identity,
                 ):
        super(SPMLP, self).__init__()
        sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]

        self.mlp = nn.Sequential()
        for i in range(len(sizes) - 1):
            layer_activation = activation
            if i == len(sizes) - 2:
                layer_activation = activate_final
            self.mlp.append(
                torch.nn.Linear(in_features=sizes[i], out_features=sizes[i + 1], bias=True),
            )
            self.mlp.append(layer_activation())
        if use_layer_norm:
            self.mlp.append(torch.nn.LayerNorm(output_size))

    def forward(self, x):
        return self.mlp(x)


class SPGNN(MessagePassing):
    def __init__(self,
                 node_in,
                 node_out,
                 edge_in,
                 edge_out,
                 num_mlp_hidden_layers,
                 mlp_hidden_size
                 ):
        super(SPGNN, self).__init__(aggr='add')  # "Add" aggregation
        self.node_fn = nn.Sequential(*[
            SPMLP(input_size=node_in + edge_out, num_hidden_layers=num_mlp_hidden_layers, hidden_size=mlp_hidden_size,
                  output_size=node_out)
        ])  # use * to unpack SPMLP

        self.edge_fn = nn.Sequential(*[
            SPMLP(input_size=node_in + node_in + edge_in, num_hidden_layers=num_mlp_hidden_layers,
                  hidden_size=mlp_hidden_size,
                  output_size=edge_out)
        ])

    def forward(self, x, edge_index, edge_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # edge_features: (E, edge_in)
        _x = x, _edge_feature = edge_features
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        return x + _x, edge_features + _edge_feature

    def message(self, edge_index, x_i, x_j, edge_features):
        """ Construct Message """
        edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_fn(edge_features)
        return edge_features

    def update(self, x_aggregated, x, edge_features):
        # x_aggregated: (E, edge_out)
        # x: (E, node_in)
        x_aggregated = torch.cat([x_aggregated, x], dim=-1)
        x_aggregated = self.node_fn(x_aggregated)
        return x_aggregated, edge_features
