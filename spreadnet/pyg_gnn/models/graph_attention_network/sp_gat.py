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
from torch_geometric.nn import GATConv, MLP, Linear


class SPGATNet(torch.nn.Module):
    def __init__(
            self,
            # in_channels: int,
            # hidden_channels: int,
            # out_channels: int,
            # heads: int = 1,
            # dropout: float = 0.6,
            # concat: bool = False,
            # add_self_loops: bool = True,
            # bias: bool = True,
            # return_attention_weights: bool = True,
            # edge_in_channels: int = 5,
            # edge_hidden_channels: int = 16,
            # edge_out_channels: int = 2,
            # edge_num_layers: int = 2,
            # edge_bias: bool = True
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
                    return_attention_weights:
                    edge_in_channels:
                    edge_hidden_channels:
                    edge_out_channels:
                    edge_num_layers:
                    edge_bias:
                """
        super(SPGATNet, self).__init__()

        self.node_encoder = Linear(3, 64)
        self.edge_encoder = Linear(1, 64)

        self.conv1 = GATConv(
            in_channels=64,
            out_channels=64,
            heads=8,
            concat=True,
            dropout=0.6,
            add_self_loops=True,
            bias=True,
            return_attention_weights=True
        )
        self.conv2 = GATConv(
            in_channels=64 * 8,
            out_channels=64,
            heads=8,
            concat=True,
            dropout=0.6,
            add_self_loops=True,
            bias=True,
            return_attention_weights=True
        )
        self.conv3 = GATConv(
            in_channels=64 * 8,
            out_channels=64,
            heads=8,
            concat=False,
            dropout=0.6,
            add_self_loops=True,
            bias=True,
            return_attention_weights=True
        )

        self.lin = Linear(64, 2)
        self.linear = Linear(3, 512)
        self.edge_classifier = MLP(
            in_channels=136,
            hidden_channels=256,
            out_channels=2,
            num_layers=2,
            bias=True
        )

    def forward(self, x, edge_index, edge_attr, return_attention_weights):
        out = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        out = F.dropout(out, p=0.3, training=self.training)
        out, (_, alpha) = self.conv1(x=out,
                                     edge_index=edge_index,
                                     edge_attr=edge_attr,
                                     return_attention_weights=True
                                     )
        out = F.elu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = out + self.linear(x)

        out, (_, alpha) = self.conv2(x=out,
                                     edge_index=edge_index,
                                     edge_attr=edge_attr,
                                     return_attention_weights=True
                                     )
        out = F.elu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = out + self.linear(x)

        out, (_, alpha) = self.conv3(x=out,
                                     edge_index=edge_index,
                                     edge_attr=edge_attr,
                                     return_attention_weights=True
                                     )
        node_embedding = F.elu(out)
        x_out = self.lin(node_embedding)

        node_embeds_src, node_embeds_dst = (
            node_embedding[edge_index[0]],
            node_embedding[edge_index[1]],
        )

        edge_len = edge_attr.size()[0]
        alpha = alpha[0:edge_len, :]
        edge_embedding = torch.cat([node_embeds_src, alpha, node_embeds_dst], dim=-1)
        edge_out = self.edge_classifier(edge_embedding)

        return x_out, edge_out
