"""

    @Time    : 10/19/2022 11:48 AM
    @Author  : Haodong Zhao

"""
import torch
from torch.nn import LayerNorm, ReLU, Linear
from torch_geometric.nn import GENConv, DeepGCNLayer, MessagePassing


class SPDeepGENLayer(MessagePassing):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        ckpt_grad: bool = False,
    ):
        super(SPDeepGENLayer, self).__init__()
        # assemble edge deep conv layer

        self.edge_pre = Linear(
            edge_in_channels + node_out_channels + node_out_channels, edge_out_channels
        )
        _edge_conv = GENConv(
            edge_in_channels,
            edge_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        _edge_norm = LayerNorm(edge_in_channels, elementwise_affine=True)
        _edge_act = ReLU(inplace=True)

        self.edge_fn = DeepGCNLayer(
            _edge_conv,
            _edge_norm,
            _edge_act,
            block="res+",
            dropout=0.1,
            ckpt_grad=ckpt_grad,
        )

        # assemble node deep conv layer
        _node_conv = GENConv(
            node_in_channels,
            node_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        _node_norm = LayerNorm(node_in_channels, elementwise_affine=True)
        _node_act = ReLU(inplace=True)

        self.node_fn = DeepGCNLayer(
            _node_conv,
            _node_norm,
            _node_act,
            block="res+",
            dropout=0.1,
            ckpt_grad=ckpt_grad,
        )

    def edge_update(self, x_i, x_j, edge_features, e_edge_index):
        # print("Edge Update: ", x_i.size())
        # print("Edge Update: ", x_j.size())
        # print("Edge Update: ", edge_features.size())
        # print("Edge Update: ", e_edge_index.size())
        edge_features = torch.concat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_pre(edge_features)
        # print("Edge Update: After Concat:  ", edge_features.size())

        edge_features = self.edge_fn(edge_features, e_edge_index)

        # print("Edge Update: After:  ", edge_features.size())
        return edge_features

    def forward(self, v_x, v_edge_index, e_x, e_edge_index):
        e_x = self.edge_updater(
            edge_index=v_edge_index, x=v_x, edge_features=e_x, e_edge_index=e_edge_index
        )
        v_x = self.node_fn(v_x, v_edge_index, e_x)
        return v_x, e_x
