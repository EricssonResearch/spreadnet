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
            num_hidden_layers: int,
            in_channels: Union[int, Tuple[int, int]],
            hidden_channels: int,
            out_channels: int,
            heads: int = 1,
            dropout: float = 0.5,
            concat: bool = False,
            add_self_loops: bool = False,
            bias: bool = True,
            return_attention_weights: bool = True,
            edge_in_channels: int = 5,
            edge_hidden_channels: int = 16,
            edge_out_channels: int = 2,
            edge_num_layers: int = 2,
            edge_bias: bool = True
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

        # if in_channels is not None:
        #    sizes = [in_channels] + [hidden_channels] * num_hidden_layers + [out_channels]
        # else:
        #    raise ValueError

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            return_attention_weights=return_attention_weights
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            return_attention_weights=return_attention_weights
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            return_attention_weights=return_attention_weights
        )
        self.edge_classifier = MLP(
            in_channels=edge_in_channels,
            hidden_channels=edge_hidden_channels,
            out_channels=edge_out_channels,
            num_layers=edge_num_layers,
            bias=edge_bias
        )
        #self.linear = Linear(
        #    out_channels=hidden_channels,
        #    bias=True
        #)

    def forward(self, x, edge_index, edge_attr, return_attention_weights):

        edge_len = edge_attr.size()[0]

        out, (edge_index, alpha) = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                              return_attention_weights=return_attention_weights,
                                              )
        #out = F.dropout(out, p=0.5, training=self.training)

        #out, (edge_index, alpha) = self.conv2(x=out, edge_index=edge_index, edge_attr=alpha, return_attention_weights=return_attention_weights)
        #out = F.dropout(out, p=0.5, training=self.training)

        out, (edge_index, alpha) = self.conv3(x=out, edge_index=edge_index, edge_attr=alpha,
                                              return_attention_weights=return_attention_weights,
                                              )
        out = F.dropout(out, p=0.5, training=self.training)


        edge_index = edge_index[:, 0:edge_len]
        alpha = alpha[0:edge_len, :]

        out_src, out_dst = out[edge_index[0]], out[edge_index[1]]
        edge_feat = torch.cat([out_src, alpha, out_dst], dim=-1)

        edge_out = self.edge_classifier(edge_feat)

        return out, edge_out
