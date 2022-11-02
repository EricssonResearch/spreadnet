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
        losses: { nodes: loss, edges: loss }
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

    (_, corrects) = get_infers(node_pred, edge_pred, node_true, edge_true)

    return losses, corrects


def get_infers(node_pred, edge_pred, node_true, edge_true):
    """Get number of correct predictions.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label

    Returns:
        infers: { nodes: infer, edges: infer }
        corrects: { nodes: correct, edges: correct }
    """

    node_infer = torch.argmax(node_pred, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edge_pred, dim=-1).type(torch.int64)

    infers = {"nodes": node_infer, "edges": edge_infer}

    corrects = {
        "nodes": torch.sum(
            (node_true == node_infer)
            .clone()
            .detach()
            .type(torch.int64)
            .to(node_pred.device)
        ),
        "edges": torch.sum(
            (edge_true == edge_infer)
            .clone()
            .detach()
            .type(torch.float)
            .to(edge_pred.device)
        ),
    }
    return infers, corrects


def corrects_in_path(node_pred, edge_pred, node_true, edge_true):
    """calculate the accuracies of elements in the shortest path.

    Args:
        node_pred: the node prediction
        edge_pred: the edge prediction
        node_true: the ground-truth node label
        edge_true: the ground-truth edge label

    Returns:
        node_in_path {"in_path": node_corrects_in_path, "total":  node_in_path_total}
        edge_in_path {"in_path": edge_corrects_in_path, "total": edge_in_path_total}
    """
    node_infer = torch.argmax(node_pred, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edge_pred, dim=-1).type(torch.int64)

    node_true_mask = node_true.type(torch.int64) == 1  # node_true_in_path
    node_in_path_total = torch.sum(node_true_mask.type(torch.int64) == 1)
    # extract the values who have some index of ground-truth values that in the path
    node_infer_in_path = (
        node_infer[node_true_mask]
        .clone()
        .detach()
        .type(torch.int64)
        .to(node_pred.device)
    )

    node_corrects_in_path = torch.sum(node_infer_in_path == 1)

    edge_true_mask = edge_true.type(torch.int64) == 1  # edge_true_in_path
    edge_in_path_total = torch.sum(edge_true_mask.type(torch.int64) == 1)
    edge_infer_in_path = (
        edge_infer[edge_true_mask]
        .clone()
        .detach()
        .type(torch.int64)
        .to(edge_pred.device)
    )
    edge_corrects_in_path = torch.sum(edge_infer_in_path == 1)

    node_in_path = {"in_path": node_corrects_in_path, "total": node_in_path_total}
    edge_in_path = {"in_path": edge_corrects_in_path, "total": edge_in_path_total}
    # print(node_in_path)
    # print(edge_in_path)

    return node_in_path, edge_in_path
