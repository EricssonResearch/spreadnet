"""Adatptive MPNN Model.

@Time    : 11/23/2022 11:27 PM
@Author  : Haodong Zhao
"""

from typing import Optional

from torch import nn
from torch_geometric.nn import MessagePassing

from .sp_modules import SPMLP, SPGNN


class AdaptiveMPNN(nn.Module):
    """The encode-process-decode model.

    It combines `Encoder`, `Processor`  and `Decoder`.
    """

    def __init__(
        self,
        node_out: int,
        edge_out: int,
        latent_size: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
        node_in: Optional[int] = None,
        edge_in: Optional[int] = None,
        basic_num_layer: int = 3,
        repeat_times: int = 1,
    ):
        """

        Args:
            node_out: the output size for the node output
            edge_out: the output size for the edge output
            latent_size: the latent size.
                latent_size = encoder output size = processor input size = processor
                 output size = decoder input size
            num_mlp_hidden_layers:  the num of the hidden layers in the MLPs.
            mlp_hidden_size: the size of the hidden layers in the MLPs.
            node_in: the input size of the node features
            edge_in:  the input size of the edge features
        """
        super(AdaptiveMPNN, self).__init__()
        self._encoder = Encoder(
            node_in=node_in,
            node_out=latent_size,
            edge_in=edge_in,
            edge_out=latent_size,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size,
        )
        self._processor = AdaptiveProcessor(
            node_in=latent_size,
            node_out=latent_size,
            edge_in=latent_size,
            edge_out=latent_size,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size,
            basic_num_mpnn=basic_num_layer,
            repeat_times=repeat_times,
        )
        self._decoder = Decoder(
            node_in=latent_size,
            node_out=node_out,
            edge_in=latent_size,
            edge_out=edge_out,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size,
        )

    def forward(self, x, edge_index, edge_features):
        x, edge_features = self._encoder(x, edge_features)
        x, edge_features = self._processor(x, edge_index, edge_features)
        output_node, output_edge = self._decoder(x, edge_features)
        return output_node, output_edge


class AdaptiveProcessor(MessagePassing):
    """
    Processor: a list of GNN Blocks
    """

    def __init__(
        self,
        node_in: int,
        node_out: int,
        edge_in: int,
        edge_out: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
        basic_num_mpnn: int,
        repeat_times: int,
    ):
        super(AdaptiveProcessor, self).__init__(aggr="add")
        self.repeat_times = repeat_times
        self.basic_num_mpnn = basic_num_mpnn
        self.sub_processors = nn.ModuleList(
            [
                SPGNN(
                    node_in=node_in,
                    node_out=node_out,
                    edge_in=edge_in,
                    edge_out=edge_out,
                    num_mlp_hidden_layers=num_mlp_hidden_layers,
                    mlp_hidden_size=mlp_hidden_size,
                )
                for _ in range(basic_num_mpnn)
            ]
        )

    def forward(self, x, edge_index, edge_features):
        for _ in range(self.repeat_times):
            for sub_processor in self.sub_processors:
                x, edge_features = sub_processor(x, edge_index, edge_features)

        return x, edge_features


class Encoder(nn.Module):
    """Encoder."""

    def __init__(
        self,
        node_out: int,
        edge_out: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
        node_in: Optional[int] = None,
        edge_in: Optional[int] = None,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(
            *[
                SPMLP(
                    input_size=node_in,
                    hidden_size=mlp_hidden_size,
                    num_hidden_layers=num_mlp_hidden_layers,
                    output_size=node_out,
                    use_layer_norm=True,
                )
            ]
        )

        self.edge_fn = nn.Sequential(
            *[
                SPMLP(
                    input_size=edge_in,
                    num_hidden_layers=num_mlp_hidden_layers,
                    hidden_size=mlp_hidden_size,
                    output_size=edge_out,
                    use_layer_norm=True,
                )
            ]
        )

    def forward(self, x, edge_features):
        return self.node_fn(x), self.edge_fn(edge_features)


class Decoder(nn.Module):
    """Decoder."""

    def __init__(
        self,
        node_in: int,
        node_out: int,
        edge_in: int,
        edge_out: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
    ):
        super(Decoder, self).__init__()
        self.node_fn = SPMLP(
            input_size=node_in,
            num_hidden_layers=num_mlp_hidden_layers,
            hidden_size=mlp_hidden_size,
            output_size=node_out,
            use_layer_norm=False,
        )

        self.edge_fn = SPMLP(
            input_size=edge_in,
            num_hidden_layers=num_mlp_hidden_layers,
            hidden_size=mlp_hidden_size,
            output_size=edge_out,
            use_layer_norm=False,
        )

    def forward(self, x, edge_features):
        return self.node_fn(x), self.edge_fn(edge_features)
