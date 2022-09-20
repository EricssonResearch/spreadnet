"""

    @Time    : 9/18/2022 10:38 PM
    @Author  : Haodong Zhao
    
"""
import torch
import torch.nn.functional as F

from architecture.models import EncodeProcessDecode
from utils import data_to_input_label


def loss_fn(data, trainable_gnn: EncodeProcessDecode):
    input_data, labels = data_to_input_label(data)
    nodes_data, edges_data = input_data
    edge_index = data.edge_index

    node_logits, edge_logits = trainable_gnn(nodes_data, edge_index, edges_data)
    node_labels, edge_labels = labels
    # print("[in loss] node_logits: ", node_logits.size())
    # print("[in loss] node_labels: ", node_labels.size())
    # print("[in loss] edge_logits", edge_logits.size())
    # print("[in loss] edge_labels", edge_labels.size())

    losses = {
        "nodes": torch.nn.functional.cross_entropy(node_logits, node_labels, reduction='mean'),
        "edges": torch.nn.functional.cross_entropy(edge_logits, edge_labels, reduction='mean'),
    }

    # track the accuracy
    predicted_node_labels = torch.argmax(node_logits, dim=-1).type(torch.int64)
    predicted_edge_labels = torch.argmax(edge_logits, dim=-1).type(torch.int64)

    accuracies = {
        "edges":
            torch.mean(torch.Tensor(node_labels == predicted_node_labels).type(torch.float32)),
        "nodes":
            torch.mean(torch.Tensor(edge_labels == predicted_edge_labels).type(torch.float32))
    }

    return losses, accuracies
