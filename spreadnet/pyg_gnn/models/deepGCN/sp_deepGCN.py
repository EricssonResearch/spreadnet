import torch

from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.nn import MLP
from torch.nn import LayerNorm, ReLU
import torch.nn.functional as F


class SPDeepGCN(torch.nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        encoder_hidden_channels: int,
        encoder_layers: int,
        gcn_hidden_channels: int,
        gcn_layers: int,
        decoder_hidden_channels: int,
        decoder_layers: int,
    ):
        super(SPDeepGCN, self).__init__()

        self.node_encoder = MLP(
            in_channels=node_in,
            out_channels=gcn_hidden_channels,
            bias=True,
            hidden_channels=encoder_hidden_channels,
            num_layers=encoder_layers,
        )

        self.edge_encoder = MLP(
            in_channels=edge_in,
            out_channels=gcn_hidden_channels,
            bias=True,
            hidden_channels=encoder_hidden_channels,
            num_layers=encoder_layers,
        )

        self.layers = torch.nn.ModuleList()

        for i in range(1, gcn_layers + 1):
            conv = GENConv(
                gcn_hidden_channels,
                gcn_hidden_channels,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            norm = LayerNorm(gcn_hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block="res+", dropout=0.1, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.node_decoder = MLP(
            in_channels=gcn_hidden_channels,
            out_channels=2,
            bias=True,
            hidden_channels=decoder_hidden_channels,
            num_layers=decoder_layers,
        )

        self.edge_decoder = MLP(
            in_channels=gcn_hidden_channels * 2,
            out_channels=2,
            bias=True,
            hidden_channels=decoder_hidden_channels * 2,
            num_layers=decoder_layers,
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        x_src_node, x_dst_node = (
            x[edge_index[0]],
            x[edge_index[1]],
        )

        edge_attr = torch.concat([x_src_node, x_dst_node], dim=-1)

        x_out = self.node_decoder(x)
        edge_out = self.edge_decoder(edge_attr)

        return x_out, edge_out
