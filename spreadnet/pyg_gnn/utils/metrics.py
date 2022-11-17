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


def get_precise_corrects(infers, labels, graph_sizes):
    """
    Get precise corrects
    Args:
        infers: {"nodes":..., "edges":...}
        labels: Tuple: node_labels, edge_labels
        graph_sizes: [{"nodes": n, "edges": n}]

    Returns: precise_corrects

    """
    precise_corrects = 0.0
    node_idx = 0
    edge_idx = 0
    node_infers = infers["nodes"].tolist()
    edge_infers = infers["edges"].tolist()

    node_labels = labels[0].tolist()
    edge_labels = labels[1].tolist()

    for graph_size in graph_sizes:
        node_corrects = 0
        edge_corrects = 0

        for _ in range(graph_size["nodes"]):
            if node_infers[node_idx] == node_labels[node_idx]:
                node_corrects += 1
            node_idx += 1

        for _ in range(graph_size["edges"]):
            if edge_infers[edge_idx] == edge_labels[edge_idx]:
                edge_corrects += 1
            edge_idx += 1

        if node_corrects == graph_size["nodes"]:
            precise_corrects += 0.5
        if edge_corrects == graph_size["edges"]:
            precise_corrects += 0.5

    return precise_corrects
