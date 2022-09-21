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
    node_labels = node_labels.type(torch.int64)
    edge_labels = edge_labels.type(torch.int64)
    # print("[in loss] node_logits: ", node_logits.size())
    # print("[in loss] node_labels: ", node_labels.size())
    # print("[in loss] edge_logits", edge_logits.size())
    # print("[in loss] edge_labels", edge_labels.size())

    losses = {
        "nodes": torch.nn.functional.cross_entropy(node_logits, node_labels, reduction='mean'),
        "edges": torch.nn.functional.cross_entropy(edge_logits, edge_labels, reduction='mean'),
    }

    # track the num of correct predictions
    predicted_node_labels = torch.argmax(node_logits, dim=-1).type(torch.int64)
    predicted_edge_labels = torch.argmax(edge_logits, dim=-1).type(torch.int64)

    node_comps = torch.Tensor((node_labels == predicted_node_labels).to(torch.float)).type(torch.int64)
    edge_comps = torch.Tensor((edge_labels == predicted_edge_labels).to(torch.float)).type(torch.int64)

    corrects = {
        "nodes": torch.sum(node_comps),
        "edges": torch.sum(edge_comps)
    }

    assert (data.num_nodes >= torch.sum(node_comps))
    assert (data.num_edges >= torch.sum(edge_comps))

    return losses, corrects
