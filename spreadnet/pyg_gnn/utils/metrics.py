"""

    @Time    : 11/4/2022 11:41 AM
    @Author  : Haodong Zhao

"""
import torch


def get_corrects_in_path(node_pred, edge_pred, node_true, edge_true):
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

    return node_in_path, edge_in_path


def get_correct_predictions(node_pred, edge_pred, node_true, edge_true):
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


def get_precise_corrects(corrects, data_nums):
    """
    Get precise Corrections
    Args:
        corrects: `_, corrects = get_correct_predictions()`  {"nodes":..., "edges":...}
        data_nums: Tuple: data.num_nodes, data.num_edges

    Returns: precise_corrects

    """
    num_nodes, num_edges = data_nums
    precise_corrects = 0.0

    if corrects["nodes"] == num_nodes:
        precise_corrects += 0.5

    if corrects["edges"] == num_edges:
        precise_corrects += 0.5

    return precise_corrects
