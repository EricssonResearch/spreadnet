"""

    @Time    : 10/19/2022 1:40 PM
    @Author  : Haodong Zhao

"""
from typing import Optional, Tuple

import torch
from torch_geometric.nn import MLP

from spreadnet.pyg_gnn.models.co_graph_conv_network.sp_gcn_modules import SPGENLayer


class SPCoDeepGCNet(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden_channels: int,
        num_layers: int,
        node_out: int = 2,
        edge_out: int = 2,
    ):
        super(SPCoDeepGCNet, self).__init__()

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
        self.layers.append(
            SPGENLayer(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                hidden_channels,
                make_deep_layer=False,
            )
        )

        for i in range(2, num_layers + 1):
            layer = SPGENLayer(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                hidden_channels,
                ckpt_grad=bool(i % 3),
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

    def forward(
        self,
        v_x=None,
        v_edge_index=None,
        e_x=None,
        e_edge_index=None,
        inputs_tuple: Optional[Tuple] = None,
    ):
        if isinstance(inputs_tuple, Tuple):
            v_x, v_edge_index, e_x, e_edge_index = inputs_tuple

        v_x = self.node_encoder(v_x)
        e_x = self.edge_encoder(e_x)

        for layer in self.layers:
            v_x, e_x = layer(v_x, v_edge_index, e_x, e_edge_index)

        node_out = self.node_decoder(v_x)
        edge_out = self.edge_decoder(e_x)

        return node_out, edge_out


class SPCoGCNet(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden_channels: int,
        num_layers: int,
        node_out: int = 2,
        edge_out: int = 2,
    ):
        super(SPCoGCNet, self).__init__()

        self.node_encoder = MLP(
            in_channels=node_in,
            out_channels=hidden_channels,
            bias=True,
            hidden_channels=128,
            num_layers=2,
            norm="batch_norm",
        )

        self.edge_encoder = MLP(
            in_channels=edge_in,
            out_channels=hidden_channels,
            bias=True,
            hidden_channels=128,
            num_layers=2,
            norm="batch_norm",
        )

        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers + 1):
            self.layers.append(
                SPGENLayer(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    make_deep_layer=False,
                )
            )

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

    def forward(
        self,
        v_x=None,
        v_edge_index=None,
        e_x=None,
        e_edge_index=None,
        inputs_tuple: Optional[Tuple] = None,
    ):
        if isinstance(inputs_tuple, Tuple):
            v_x, v_edge_index, e_x, e_edge_index = inputs_tuple

        v_x = self.node_encoder(v_x)
        e_x = self.edge_encoder(e_x)

        for layer in self.layers:
            v_x, e_x = layer(v_x, v_edge_index, e_x, e_edge_index)

        node_out = self.node_decoder(v_x)
        edge_out = self.edge_decoder(e_x)

        return node_out, edge_out
