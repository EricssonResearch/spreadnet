"""
    The basic Blocks for EncodeProcessDecode:
    1. SPMLP: simple multi-layer Perceptron
    2. SPGNN: GNN Block

    @Time    : 9/15/2022 11:31 PM
    @Author  : Haodong Zhao

"""
from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class SPMLP(nn.Module):
    """simple multi-layer Perceptron."""

    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        output_size: int,
        use_layer_norm: bool = True,
        activation=torch.nn.ReLU,
        activate_final=torch.nn.Identity,
        input_size: Optional[int] = None,
    ):
        """

        Args:
            input_size: the input size of the MLP.
            num_hidden_layers: the number of hidden layers in the MLP.
            hidden_size: the size of each hidden layer in the MLP.
            output_size: the output size of the MLP.
            use_layer_norm: if apply Layer Normalization. If true, add
            `torch.nn.LayerNorm(output_size)` to the MLP.
            activation: the activation function. By default, it's `torch.nn.ReLU`.
            activate_final: the activation of the final layer.

        """
        super(SPMLP, self).__init__()
        self.mlp = nn.Sequential()
        if input_size is not None:
            sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
        else:
            self.mlp.append(nn.LazyLinear(out_features=hidden_size))
            sizes = [hidden_size] * num_hidden_layers + [output_size]

        for i in range(len(sizes) - 1):
            layer_activation = activation
            if i == len(sizes) - 2:
                layer_activation = activate_final
            self.mlp.append(
                torch.nn.Linear(
                    in_features=sizes[i], out_features=sizes[i + 1], bias=True
                ),
            )
            self.mlp.append(layer_activation())
        if use_layer_norm:
            self.mlp.append(torch.nn.LayerNorm(output_size))

    def forward(self, x):
        return self.mlp(x)


class SPGNN(MessagePassing):
    """The basic GN block.

    It only contains the computations for the nodes and edges, no computations
    related to "global" so far.
    """

    def __init__(
        self,
        node_in: int,
        node_out: int,
        edge_in: int,
        edge_out: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
    ):
        """

        Args:
            node_in: the node input size
            node_out: the node output size
            edge_in: the edge input size
            edge_out: the edge output size
            num_mlp_hidden_layers: the number of hidden layers in the MLPs.
            mlp_hidden_size: the size of the hidden layer in the MLP.
        """
        super(SPGNN, self).__init__(aggr="add")  # "Add" aggregation
        self.node_fn = nn.Sequential(
            *[
                SPMLP(
                    input_size=node_in + edge_out,
                    num_hidden_layers=num_mlp_hidden_layers,
                    hidden_size=mlp_hidden_size,
                    output_size=node_out,
                )
            ]
        )  # use * to unpack SPMLP

        self.edge_fn = nn.Sequential(
            *[
                SPMLP(
                    input_size=node_in + node_in + edge_in,
                    num_hidden_layers=num_mlp_hidden_layers,
                    hidden_size=mlp_hidden_size,
                    output_size=edge_out,
                )
            ]
        )

    def message(self, edge_index, x_i, x_j, edge_features):
        """Construct Message. This function overrides `message()` in class
        MassagePassing, which can take any argument as input which was
        initially passed to `propagate`

        Args:
            edge_index: the source, target information of the edges.
            x_i: the nodes which aggregate information.
                (message passing flow is source_to_target)
            x_j: the nodes which send information along the edges.
                (message passing flow is source_to_target)
            edge_features: the edge features.

        Returns:
            the constructed edge message
        """
        edge_features = torch.concat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_fn(edge_features)
        return edge_features

    def update(self, aggregated, x, edge_features):
        """Update nodes.

            Takes in the output of aggregation as first argument and any argument
            which was initially passed to `propagate`.

        Args:
            aggregated: the aggregated information
            x: node features
            edge_features: edge_features

        Returns:
            the updated node data and edges features.
        """

        x_updated = torch.concat([aggregated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, edge_features

    def forward(self, x, edge_index, edge_features):
        """Define the GNN computation. It utilizes `propagate`(message =>
        aggregate => update).

        Args:
            x: the node features.
            edge_index: the source, target information of the edges.
            edge_features: the edge features.

        Returns:
            the output of the model
        """

        _x = x
        _edge_feature = edge_features
        # propagate: message => aggregate => update
        x, edge_features = self.propagate(
            edge_index=edge_index, x=x, edge_features=edge_features
        )
        return x + _x, edge_features + _edge_feature
