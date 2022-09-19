"""

    @Time    : 9/15/2022 11:27 PM
    @Author  : Haodong Zhao
    
"""
from torch import nn
from torch_geometric.nn import MessagePassing

from architecture.sp_gnn import SPMLP, SPGNN


class EncodeProcessDecode(nn.Module):
    """
        The encode-process-decode model
    """

    def __init__(self,
                 node_in,
                 node_out,
                 edge_in,
                 edge_out,
                 latent_size: int,
                 num_message_passing_steps: int,
                 num_mlp_hidden_layers: int,
                 mlp_hidden_size: int
                 ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = _Encoder(
            node_in=node_in,
            node_out=node_out,
            edge_in=edge_in,
            edge_out=latent_size,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size
        )
        self._processor = _Processor(
            node_in=node_in,
            node_out=node_out,
            edge_in=edge_in,
            edge_out=latent_size,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size,
            num_message_passing_steps=num_message_passing_steps
        )
        self._decoder = _Decoder(
            node_in=latent_size,
            node_out=node_out,
            edge_in=latent_size,
            edge_out=edge_out,
            num_mlp_hidden_layers=num_mlp_hidden_layers,
            mlp_hidden_size=mlp_hidden_size
        )

    def forward(self, x, edge_index, edge_features):
        x, edge_features = self._encoder(x, edge_index, edge_features)
        x, edge_features = self._processor(x, edge_index, edge_features)
        x, edge_features = self._decoder(x, edge_features)
        return x, edge_features


class _Encoder(nn.Module):
    """
        Encoder
    """

    def __init__(self,
                 node_in,
                 node_out,
                 edge_in,
                 edge_out,
                 num_mlp_hidden_layers: int,
                 mlp_hidden_size: int
                 ):
        super(_Encoder, self).__init__()
        self.node_fn = nn.Sequential(
            *[SPMLP(
                input_size=node_in,
                hidden_size=mlp_hidden_size,
                num_hidden_layers=num_mlp_hidden_layers,
                output_size=node_out,
                use_layer_norm=True)])

        self.edge_fn = nn.Sequential(
            *[SPMLP(
                input_size=edge_in,
                num_hidden_layers=num_mlp_hidden_layers,
                hidden_size=mlp_hidden_size,
                output_size=edge_out,
                use_layer_norm=True)])

    def forward(self, x, edge_index, edge_features):
        return self.node_fn(x), self.edge_fn(edge_features)


class _Processor(MessagePassing):
    """
        Processor:
            consists of a list of GNN Blocks
    """

    def __init__(self,
                 node_in,
                 node_out,
                 edge_in,
                 edge_out,
                 num_message_passing_steps: int,
                 num_mlp_hidden_layers: int,
                 mlp_hidden_size: int,
                 ):
        super(_Processor, self).__init__(aggr='add')
        self.sub_processors = nn.ModuleList(
            [
                SPGNN(
                    node_in=node_in,
                    node_out=node_out,
                    edge_in=edge_in,
                    edge_out=edge_out,
                    num_mlp_hidden_layers=num_mlp_hidden_layers,
                    mlp_hidden_size=mlp_hidden_size
                ) for _ in range(num_message_passing_steps)
            ]
        )

    def forward(self, x, edge_index, edge_features):
        for sub_processor in self.sub_processors:
            x, edge_features = sub_processor(x, edge_index, edge_features)

        return x, edge_features


class _Decoder(nn.Module):
    """
        Decoder
    """

    def __init__(
            self,
            node_in,
            node_out,
            edge_in,
            edge_out,
            num_mlp_hidden_layers: int,
            mlp_hidden_size: int
    ):
        super(_Decoder, self).__init__()
        self.node_fn = SPMLP(
            input_size=node_in,
            num_hidden_layers=num_mlp_hidden_layers,
            hidden_size=mlp_hidden_size,
            output_size=node_out,
            use_layer_norm=True
        )

        self.edge_fn = SPMLP(
            input_size=edge_in,
            num_hidden_layers=num_mlp_hidden_layers,
            hidden_size=mlp_hidden_size,
            output_size=edge_out,
            use_layer_norm=True
        )

    def forward(self, x, edge_features):
        return self.node_fn(x), self.edge_fn(edge_features)
