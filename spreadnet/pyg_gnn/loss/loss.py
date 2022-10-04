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
        corrects: (node_corrects, edge_corrects)
    """

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

    node_comps = (
        (node_true == predicted_node_labels)
            .clone()
            .detach()
            .type(torch.int64)
            .to(node_pred.device)
    )
    edge_comps = (
        (edge_true == predicted_edge_labels)
            .clone()
            .detach()
            .type(torch.float)
            .to(node_pred.device)
    )

    corrects = {"nodes": torch.sum(node_comps), "edges": torch.sum(edge_comps)}

    # assert data.num_nodes >= torch.sum(node_comps)
    # assert data.num_edges >= torch.sum(edge_comps)
    return losses, corrects


def cross_entropy_loss(node_pred, node_true):
    loss = torch.nn.functional.cross_entropy(node_pred, node_true, reduction="mean")

    predicted_labels = torch.argmax(node_pred, dim=-1).type(torch.int64)
    comp = (
        (node_true == predicted_labels)
            .clone()
            .detach()
            .type(torch.int64)
            .to(node_pred.device)
    )
    correct = torch.sum(comp)
    return loss, correct
