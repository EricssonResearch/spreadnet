"""The loss function.

@Time    : 9/18/2022 10:38 PM
@Author  : Haodong Zhao
"""
import torch


def get_masked_loss(
    node_true, node_pred, node_losses, edge_true, edge_pred, edge_losses
):
    # getting the node and edge multiplication factor:
    # ratio = nr_nodes_not_in_path / nr_nodes_in_path
    count_nodes_in_path = torch.bincount(node_true)  # tensor([nr of 0's, nr of 1's])
    node_mult_factor = count_nodes_in_path[0] / count_nodes_in_path[1]
    count_edges_in_path = torch.bincount(edge_true)
    edge_mult_factor = count_edges_in_path[0] / count_edges_in_path[1]

    # converts logits to their classes
    node_0 = node_pred[:, 0]
    node_1 = node_pred[:, 1]
    node_pred_classes = torch.where(node_0 > node_1, 0, 1)
    # adds penalties to the losses with a node wrongly classified as "not in path"
    updated_node_losses = torch.where(
        torch.logical_and(node_pred_classes == 0, node_true == 1),
        node_losses * node_mult_factor,
        node_losses,
    )
    updated_node_losses = torch.mean(updated_node_losses)

    # repeat for edges
    edge_0 = edge_pred[:, 0]
    edge_1 = edge_pred[:, 1]
    edge_pred_classes = torch.where(edge_0 > edge_1, 0, 1)
    updated_edge_losses = torch.where(
        torch.logical_and(edge_pred_classes == 0, edge_true == 1),
        edge_losses * edge_mult_factor,
        edge_losses,
    )
    updated_edge_losses = torch.mean(updated_edge_losses)

    return {"nodes": updated_node_losses, "edges": updated_edge_losses}


def hybrid_loss(node_pred, edge_pred, node_true, edge_true):
    """A hybrid cross entropy loss combining edges and nodes.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label

    Returns:
        losses: { nodes: loss, edges: loss }
        corrects: { nodes: correct, edges: correct }
    """

    losses = {
        "nodes": torch.nn.functional.cross_entropy(
            node_pred, node_true, reduction="none"
        ),
        "edges": torch.nn.functional.cross_entropy(
            edge_pred, edge_true, reduction="none"
        ),
    }
    penalized_losses = get_masked_loss(
        node_true, node_pred, losses["nodes"], edge_true, edge_pred, losses["edges"]
    )

    return penalized_losses
