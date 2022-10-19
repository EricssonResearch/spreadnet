"""

    @Time    : 10/19/2022 11:48 AM
    @Author  : Haodong Zhao

"""
import torch
from torch.nn import LayerNorm, ReLU, Linear
from torch_geometric.nn import GENConv, DeepGCNLayer, MessagePassing


class SPGENLayer(MessagePassing):
    def __init__(
        self,
        node_in_channels,
        node_out_channels,
        edge_in_channels,
        edge_out_channels,
        make_deep_layer: bool = True,
        ckpt_grad: bool = False,
    ):
        super(SPGENLayer, self).__init__(aggr="add")

        # assemble edge deep conv layer
        self.edge_linear = Linear(
            edge_in_channels + node_out_channels * 2, edge_out_channels
        )
        edge_conv = GENConv(
            edge_in_channels,
            edge_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )

        if make_deep_layer:
            edge_norm = LayerNorm(edge_in_channels, elementwise_affine=True)
            edge_act = ReLU(inplace=True)
            self.edge_fn = DeepGCNLayer(
                edge_conv,
                edge_norm,
                edge_act,
                block="res+",
                dropout=0.1,
                ckpt_grad=ckpt_grad,
            )
        else:
            self.edge_fn = edge_conv

        # assemble node deep conv layer
        node_conv = GENConv(
            node_in_channels,
            node_out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )

        if make_deep_layer:
            node_norm = LayerNorm(node_in_channels, elementwise_affine=True)
            node_act = ReLU(inplace=True)

            self.node_fn = DeepGCNLayer(
                node_conv,
                node_norm,
                node_act,
                block="res+",
                dropout=0.1,
                ckpt_grad=ckpt_grad,
            )
        else:
            self.node_fn = node_conv

    def edge_update(self, x_i, x_j, edge_features, e_edge_index):
        # print("Edge Update: ", x_i.size())
        # print("Edge Update: ", x_j.size())
        # print("Edge Update: ", edge_features.size())
        # print("Edge Update: ", e_edge_index.size())
        edge_features = torch.concat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_linear(edge_features)
        # print("Edge Update: After Concat:  ", edge_features.size())

        edge_features = self.edge_fn(edge_features, e_edge_index)
        return edge_features

    # def message(self, edge_features):
    #     return edge_features
    #
    # def update(self, aggregated, x, edge_index):
    #     x_updated = torch.concat([aggregated, x], dim=-1)
    #     x_updated = self.node_fn(x_updated, edge_index)
    #     return x_updated

    def forward(self, v_x, v_edge_index, e_x, e_edge_index):
        e_x = self.edge_updater(
            edge_index=v_edge_index, x=v_x, edge_features=e_x, e_edge_index=e_edge_index
        )

        v_x = self.node_fn(v_x, v_edge_index)
        return v_x, e_x
