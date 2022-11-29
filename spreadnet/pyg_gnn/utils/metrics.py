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


def calc_f_score(precision, recall, beta=2):
    """
    Calc f score
    Args:
        precision: true positive / (true and false positives)
        recall: true positive / (true positives and false negatives)
        beta: chosen such that recall is considered n times as important as precision

    Returns: f_score

    """
    return (1 + (beta**2)) * (
        (precision * recall) / (((beta**2) * precision) + recall)
    )


def get_precise_and_f_score(infers, labels, graph_sizes):
    """
    Get precise corrects
    Args:
        infers: {"nodes":..., "edges":...}
        labels: Tuple: node_labels, edge_labels
        graph_sizes: [{"nodes": n, "edges": n}]

    Returns: nodes_precise_corrects, edges_precise_corrects, nodes_score, edges_score

    """

    node_idx = 0
    edge_idx = 0

    node_infers = infers["nodes"].tolist()
    edge_infers = infers["edges"].tolist()
    node_labels = labels[0].tolist()
    edge_labels = labels[1].tolist()

    nodes_corrects = 0.0
    edges_corrects = 0.0
    nodes_score = 0.0
    edges_score = 0.0

    for graph_size in graph_sizes:
        node_corrects = 0
        node_true_positives = 0
        node_false_positives = 0
        node_false_negatives = 0

        edge_corrects = 0
        edge_true_positives = 0
        edge_false_positives = 0
        edge_false_negatives = 0

        for _ in range(graph_size["nodes"]):
            if node_infers[node_idx] == node_labels[node_idx]:
                node_corrects += 1

            if node_labels[node_idx] and node_infers[node_idx]:
                node_true_positives += 1
            elif node_labels[node_idx]:
                node_false_negatives += 1
            elif node_infers[node_idx]:
                node_false_positives += 1

            node_idx += 1

        for _ in range(graph_size["edges"]):
            if edge_infers[edge_idx] == edge_labels[edge_idx]:
                edge_corrects += 1

            if edge_labels[edge_idx] and edge_infers[edge_idx]:
                edge_true_positives += 1
            elif edge_labels[edge_idx]:
                edge_false_negatives += 1
            elif edge_infers[edge_idx]:
                edge_false_positives += 1

            edge_idx += 1

        if node_corrects == graph_size["nodes"]:
            nodes_corrects += 1
        if edge_corrects == graph_size["edges"]:
            edges_corrects += 1

        try:
            nodes_precision = node_true_positives / (
                node_true_positives + node_false_positives
            )
            nodes_recall = node_true_positives / (
                node_true_positives + node_false_negatives
            )
            nodes_score += calc_f_score(nodes_precision, nodes_recall)
        except Exception:
            pass

        try:
            edges_precision = edge_true_positives / (
                edge_true_positives + edge_false_positives
            )
            edges_recall = edge_true_positives / (
                edge_true_positives + edge_false_negatives
            )
            edges_score += calc_f_score(edges_precision, edges_recall)
        except Exception:
            pass

    return nodes_corrects, edges_corrects, nodes_score, edges_score
