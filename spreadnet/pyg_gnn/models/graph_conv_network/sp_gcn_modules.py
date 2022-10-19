"""

    @Time    : 10/19/2022 11:48 AM
    @Author  : Haodong Zhao

"""
from torch import nn
from torch.nn import LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer


class SPDeepGENLayer(nn.Module):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        ckpt_grad: bool = False,
    ):
        super(SPDeepGENLayer, self).__init__()
        _edge_conv = GENConv(
            edge_in_channels,
            edge_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        _edge_norm = LayerNorm(edge_out_channels, elementwise_affine=True)
        _edge_act = ReLU(inplace=True)
        self.edge_layer = DeepGCNLayer(
            _edge_conv,
            _edge_norm,
            _edge_act,
            block="res+",
            dropout=0.1,
            ckpt_grad=ckpt_grad,
        )

        _node_conv = GENConv(
            node_in_channels,
            node_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        _node_norm = LayerNorm(node_out_channels, elementwise_affine=True)
        _node_act = ReLU(inplace=True)
        self.node_layer = DeepGCNLayer(
            _node_conv,
            _node_norm,
            _node_act,
            block="res+",
            dropout=0.1,
            ckpt_grad=ckpt_grad,
        )

    def forward(self, v_x, v_edge_index, e_x, e_edge_index):

        e_x = self.edge_layer(e_x, e_edge_index)
        v_x = self.node_layer(v_x, v_edge_index, e_x)

        return v_x, e_x
