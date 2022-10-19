"""

    @Time    : 10/19/2022 1:40 PM
    @Author  : Haodong Zhao

"""
import torch
from torch_geometric.nn import MLP

from spreadnet.pyg_gnn.models.graph_conv_network.sp_gcn_modules import SPDeepGENLayer


class SPCoDeepGCN(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden_channels: int,
        num_layers: int,
        node_out: int = 2,
        edge_out: int = 2,
    ):
        super(SPCoDeepGCN, self).__init__()

        self.node_encoder = MLP(
            in_channels=node_in,
            out_channels=hidden_channels,
            bias=True,
            hidden_channels=128,
            num_layers=2,
        )

        self.edge_encoder = MLP(
            in_channels=edge_in,
            out_channels=hidden_channels,
            bias=True,
            hidden_channels=128,
            num_layers=2,
        )

        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers + 1):
            layer = SPDeepGENLayer(
                hidden_channels, hidden_channels, hidden_channels, hidden_channels
            )
            self.layers.append(layer)

        self.node_decoder = MLP(
            in_channels=hidden_channels,
            out_channels=node_out,
            bias=True,
            hidden_channels=128,
            num_layers=2,
        )

        self.edge_decoder = MLP(
            in_channels=hidden_channels,
            out_channels=edge_out,
            bias=True,
            hidden_channels=128,
            num_layers=2,
        )

    def forward(self, v_x, v_edge_index, e_x, e_edge_index):
        v_x = self.node_encoder(v_x)
        e_x = self.edge_encoder(e_x)

        for layer in self.layers:
            v_x, v_e = layer(v_x, v_edge_index, e_x, e_edge_index)

        node_out = self.node_decoder(v_x)
        edge_out = self.edge_decoder(e_x)

        return node_out, edge_out
