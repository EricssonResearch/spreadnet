"""The loss function.

@Time    : 9/18/2022 10:38 PM
@Author  : Haodong Zhao
"""
import torch


def hybrid_loss(node_pred, edge_pred, node_true, edge_true):
    """A hybrid cross entropy loss combining edges and nodes."""

    # print("[in loss] node_pred: ", node_pred.size())
    # print("[in loss] node_true: ", node_true.size())
    # print("[in loss] edge_logits", edge_logits.size())
    # print("[in loss] edge_true", edge_true.size())
    losses = {
        "nodes": torch.nn.functional.cross_entropy(
            node_pred, node_true, reduction="mean"
        ),
        "edges": torch.nn.functional.cross_entropy(
            edge_pred, edge_true, reduction="mean"
        ),
    }

    # track the num of correct predictions
    predicted_node_labels = torch.argmax(node_pred, dim=-1).type(torch.int64)
    predicted_edge_labels = torch.argmax(edge_pred, dim=-1).type(torch.int64)

    node_comps = torch.Tensor(
        (node_true == predicted_node_labels).to(torch.float)
    ).type(torch.int64)
    edge_comps = torch.Tensor(
        (edge_true == predicted_edge_labels).to(torch.float)
    ).type(torch.int64)

    corrects = {"nodes": torch.sum(node_comps), "edges": torch.sum(edge_comps)}

    # assert data.num_nodes >= torch.sum(node_comps)
    # assert data.num_edges >= torch.sum(edge_comps)
    return losses, corrects
