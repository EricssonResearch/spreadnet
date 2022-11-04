"""The utils function.

@Time    : 9/18/2022 10:38 PM
@Author  : Haodong Zhao
"""
import torch


def hybrid_loss(node_pred, edge_pred, node_true, edge_true):
    """A hybrid cross entropy utils combining edges and nodes.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label

    Returns:
        losses: { nodes: utils, edges: utils }
        corrects: { nodes: correct, edges: correct }
    """

    losses = {
        "nodes": torch.nn.functional.cross_entropy(
            node_pred, node_true, reduction="mean"
        ),
        "edges": torch.nn.functional.cross_entropy(
            edge_pred, edge_true, reduction="mean"
        ),
    }

    return losses
