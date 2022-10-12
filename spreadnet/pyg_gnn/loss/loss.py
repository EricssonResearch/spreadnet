"""The loss function.

@Time    : 9/18/2022 10:38 PM
@Author  : Haodong Zhao
"""
import torch


def hybrid_loss(node_pred, edge_pred, node_true, edge_true):
    """A hybrid cross entropy loss combining edges and nodes.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label

    Returns:
        losses: (node_loss, edge_loss)
    """

    losses = {
        "nodes": torch.nn.functional.cross_entropy(
            node_pred, node_true, reduction="mean"
        ),
        "edges": torch.nn.functional.cross_entropy(
            edge_pred, edge_true, reduction="mean"
        ),
    }

    node_infer = torch.argmax(node_pred, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edge_pred, dim=-1).type(torch.int64)
    nodes_correct = torch.sum(
        (node_true == node_infer)
        .clone()
        .detach()
        .type(torch.int64)
        .to(node_pred.device)
    )
    edges_correct = torch.sum(
        (edge_true == edge_infer)
        .clone()
        .detach()
        .type(torch.float)
        .to(node_pred.device)
    )

    return losses, (nodes_correct, edges_correct)
