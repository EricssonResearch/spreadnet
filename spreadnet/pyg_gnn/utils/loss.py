"""The loss function.

@Time    : 9/18/2022 10:38 PM
@Author  : Haodong Zhao
"""
import torch

from spreadnet.pyg_gnn.utils.balancing_loss import BalancingLoss


def hybrid_loss(node_pred, edge_pred, node_true, edge_true, loss_type, edge_data):
    """A hybrid cross entropy loss combining edges and nodes.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label
        loss_type: to use the original loss (d), the weighted loss (w),
                   or the sequenced weighted loss (s)
        edge_data: source and target nodes for each edge

    Returns:
        losses: { nodes: loss, edges: loss }
        corrects: { nodes: correct, edges: correct }
    """

    losses_tensor = {
        "nodes": torch.nn.functional.cross_entropy(
            node_pred, node_true, reduction="none"
        ),
        "edges": torch.nn.functional.cross_entropy(
            edge_pred, edge_true, reduction="none"
        ),
    }

    if loss_type != "d" or loss_type != "D":
        penalized_losses = BalancingLoss(
            node_true,
            node_pred,
            losses_tensor["nodes"],
            edge_true,
            edge_pred,
            losses_tensor["edges"],
            edge_data,
        )
        if loss_type == "w" or loss_type == "W":
            return penalized_losses.get_weighted_loss()
        elif loss_type == "s" or loss_type == "S":
            return penalized_losses.get_sequence_weighted_loss()

    return {
        "nodes": torch.mean(losses_tensor["nodes"]),
        "edges": torch.mean(losses_tensor["edges"]),
    }
