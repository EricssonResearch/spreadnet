"""Co-embedding Deep GENConv Networks.

@Time    : 10/19/2022 1:40 PM
@Author  : Haodong Zhao
"""

import torch
from torch_geometric.nn import MLP
from torch_scatter import scatter_add
from torch_sparse import coalesce

from spreadnet.pyg_gnn.models.co_graph_conv_network.sp_gcn_modules import SPGENLayer


@torch.no_grad()
def undirected_linegraph_index(edge_index, edge_attr, num_nodes):
    N = num_nodes
    (row, col), edge_attr = coalesce(edge_index, edge_attr, N, N)

    i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

    count = scatter_add(torch.ones_like(row), row, dim=0, dim_size=N)
    cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)
    cols = [i[cumsum[col[j]] : cumsum[col[j] + 1]] for j in range(col.size(0))]
    rows = [row.new_full((c.numel(),), j) for j, c in enumerate(cols)]
    row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)
    e_edge_index = torch.stack([row, col], dim=0)

    return e_edge_index


class SPCoDeepGCNet(torch.nn.Module):
    """Co-embedding Deep GENConv Network. Similar to Encode-Process-Decode.

    Add mechanisms in DeepGCNLayer: normalize, residual, dropout, etc.
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        mlp_hidden_channels: int,
        mlp_hidden_layers: int,
        gcn_hidden_channels: int,
        gcn_num_layers: int,
        node_out: int = 2,
        edge_out: int = 2,
    ):
        super(SPCoDeepGCNet, self).__init__()

        self.node_encoder = MLP(
            in_channels=node_in,
            out_channels=gcn_hidden_channels,
            hidden_channels=mlp_hidden_channels,
            num_layers=mlp_hidden_layers + 1,
            bias=True,
            norm="batch_norm",
        )

        self.edge_encoder = MLP(
            in_channels=edge_in,
            out_channels=gcn_hidden_channels,
            hidden_channels=mlp_hidden_channels,
            num_layers=mlp_hidden_layers + 1,
            bias=True,
            norm="batch_norm",
        )

        self.layers = torch.nn.ModuleList()
        self.layers.append(
            SPGENLayer(
                gcn_hidden_channels,
                gcn_hidden_channels,
                gcn_hidden_channels,
                gcn_hidden_channels,
                make_deep_layer=True,
            )
        )

        for i in range(2, gcn_num_layers + 1):
            layer = SPGENLayer(
                gcn_hidden_channels,
                gcn_hidden_channels,
                gcn_hidden_channels,
                gcn_hidden_channels,
                ckpt_grad=bool(i % 3),
            )
            self.layers.append(layer)

        self.node_decoder = MLP(
            in_channels=gcn_hidden_channels,
            out_channels=node_out,
            hidden_channels=mlp_hidden_channels,
            num_layers=mlp_hidden_layers + 1,
            bias=True,
            norm="batch_norm",
        )

        self.edge_decoder = MLP(
            in_channels=gcn_hidden_channels,
            out_channels=edge_out,
            hidden_channels=mlp_hidden_channels,
            num_layers=mlp_hidden_layers + 1,
            bias=True,
            norm="batch_norm",
        )

    def forward(self, x, edge_index, edge_attr):

        v_x = self.node_encoder(x)
        num_node = v_x.size()[0]

        e_edge_index = undirected_linegraph_index(edge_index, edge_attr, num_node)
        e_x = self.edge_encoder(edge_attr)

        for layer in self.layers:
            v_x, e_x = layer(v_x, edge_index, e_x, e_edge_index)

        node_out = self.node_decoder(v_x)
        edge_out = self.edge_decoder(e_x)

        return node_out, edge_out
