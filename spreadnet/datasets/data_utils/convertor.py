"""Data Convertors.

@Time    : 9/18/2022 11:39 PM
@Author  : Haodong Zhao
"""

import numpy as np


def graphnx_to_dict_spec(graph_nx):
    """Convert networkx graph to a dictionary.

    Args:
        graph_nx: a networkx graph

    Returns:
        A dictionary that contains graph data.
    """
    ret_dict = {}

    nodes_data = [data for _, data in graph_nx.nodes(data=True)]
    nodes_weight = np.array([data["weight"] for data in nodes_data], dtype=np.float64)
    nodes_position = np.array([data["pos"] for data in nodes_data], dtype=np.float64)
    nodes_start = np.array([data["is_start"] for data in nodes_data], dtype=np.int64)
    nodes_end = np.array([data["is_end"] for data in nodes_data], dtype=np.int64)
    nodes_in_sp = np.array([data["is_in_path"] for data in nodes_data], dtype=np.int64)

    ret_dict["nodes_feature"] = {
        "weights": nodes_weight,
        "pos": nodes_position,
        "is_start": nodes_start,
        "is_end": nodes_end,
        "is_in_path": nodes_in_sp,
    }

    source_indices, target_indices, edges_data = zip(*graph_nx.edges(data=True))
    source_indices = np.array(source_indices, dtype=np.int64)
    target_indices = np.array(target_indices, dtype=np.int64)
    edges_weight = np.array([data["weight"] for data in edges_data], dtype=np.float64)
    edges_in_sp = np.array([data["is_in_path"] for data in edges_data], dtype=np.int64)

    ret_dict["edges_feature"] = {
        "source_indices": source_indices,
        "target_indices": target_indices,
        "weights": edges_weight,
        "is_in_path": edges_in_sp,
    }

    return ret_dict


# def data_to_input_label(pyg_data):
#     """
#     Convert PyG data to input_data and ground-truth labels.

#     Args:
#         pyg_data: one graph data from pyg dataset set

#     Returns:
#         input_data: Tuple:[nodes_data, edges_data],
#         labels: the ground-truth labels of nodes and edges

#     """
#     node_labels, edge_labels = pyg_data.label

#     node_labels = node_labels.type(torch.int64)  # node: is_in_path
#     edge_labels = edge_labels.type(torch.int64)  # edge: is_in_path
#     label = (node_labels, edge_labels)

#     nodes_weight = pyg_data["weight"].type(torch.float32)
#     nodes_start = pyg_data["is_start"].type(torch.float32)
#     nodes_end = pyg_data["is_end"].type(torch.float32)

#     edges_weight = pyg_data["edge_weight"].type(torch.float32)
#     nodes_data = torch.concat(
#         [
#             nodes_weight[..., None],
#             nodes_start[..., None],
#             nodes_end[..., None]],
#             dim=-1
#     )
#     edges_data = torch.concat([edges_weight[..., None]], dim=-1)

#     input_data = (nodes_data, edges_data)

#     # print("[node_data size] ", nodes_data.size())
#     # print("[edges_data size] ", edges_data.size())

#     return input_data, label
