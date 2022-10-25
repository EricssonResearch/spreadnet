"""Basic GAT model.
Usage:
    python train.py [--config config_file_path]
@Time    : 10/03/2022 2:05 PM
@Author  : Haoyuan Li
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MLP, Linear


class SPGATNet(torch.nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        # dropout: float = 0.5,
        add_self_loops: bool,
        bias: bool,
        edge_hidden_channels: int,
        edge_out_channels: int,
        edge_num_layers: int,
        edge_bias: bool,
        encode_node_in: int,
        encode_edge_in: int,
        encode_node_out: int,
        encode_edge_out: int,
        concat_hidden_layer: bool = True,
        concat_out_layer: bool = False,
    ):
        """
        Args:
            num_hidden_layers: The number of hidden layers.
            in_channels: Size of each input sample.
            hidden_channels: Size of the hidden layer output sample.
            out_channels: Size of each output sample.
            heads: Number of multi-head-attentions.
            #dropout: Dropout probability of the normalized attention coefficients.
            add_self_loops: If 'True', add self loops to input graph.
            bias: If 'True', the layer will learn an additive bias.
            encode_node_in: The input size of node encoder.
            encode_edge_in: The input size of edge encoder.
            encode_node_out: The output size of node encoder.
            encode_edge_out: The output size of edge encoder.
            concat_hidden_layer: The input and hidden state of the multi-head attentions
                                 should be concatenated.
            concat_out_layer: The output of the multi-head attentions should be averaged
        """
        super(SPGATNet, self).__init__()
        self.conv = nn.Sequential()

        if in_channels is not None:
            input_sizes = (
                [in_channels]
                + [hidden_channels] * num_hidden_layers
                + [hidden_channels]
            )
        else:
            raise ValueError

        self.node_encoder = Linear(encode_node_in, encode_node_out)
        self.edge_encoder = Linear(encode_edge_in, encode_edge_out)

        for i in range(len(input_sizes)):
            if i < len(input_sizes) - 1:
                self.conv.append(
                    GATConv(
                        in_channels=input_sizes[i],
                        out_channels=out_channels,
                        heads=heads,
                        concat=concat_hidden_layer,
                        add_self_loops=add_self_loops,
                        bias=bias,
                    )
                )
            else:
                self.conv.append(
                    GATConv(
                        in_channels=input_sizes[i],
                        out_channels=out_channels,
                        heads=heads,
                        concat=concat_out_layer,
                        add_self_loops=add_self_loops,
                        bias=bias,
                    )
                )

        self.linear = Linear(out_channels, edge_out_channels)
        self.skip_connection = Linear(encode_node_in, hidden_channels)
        self.edge_classifier = MLP(
            in_channels=out_channels + heads + out_channels,
            hidden_channels=edge_hidden_channels,
            out_channels=edge_out_channels,
            num_layers=edge_num_layers,
            bias=edge_bias,
        )

    def forward(self, x, edge_index, edge_attr, return_attention_weights):
        global alpha
        node_embedding = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        # out = F.dropout(out, p=0.5, training=self.training)

        for i, gat_layer in enumerate(self.conv):
            if i == len(self.conv) - 1:
                node_embedding, (_, alpha) = gat_layer(
                    x=node_embedding,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=return_attention_weights,
                )
            else:
                node_embedding, (_, alpha) = gat_layer(
                    x=node_embedding,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_attention_weights=return_attention_weights,
                )
                node_embedding = F.elu(node_embedding)
                node_embedding = F.dropout(
                    node_embedding, p=0.5, training=self.training
                )
                node_embedding = node_embedding + self.skip_connection(x)

        node_embedding = F.elu(node_embedding)
        x_out = self.linear(node_embedding)
        node_embeds_src, node_embeds_dst = (
            node_embedding[edge_index[0]],
            node_embedding[edge_index[1]],
        )

        edge_len = edge_attr.size()[0]
        alpha = alpha[0:edge_len, :]
        edge_embedding = torch.cat([node_embeds_src, alpha, node_embeds_dst], dim=-1)
        edge_out = self.edge_classifier(edge_embedding)

        return x_out, edge_out
