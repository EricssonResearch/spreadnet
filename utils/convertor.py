"""

    @Time    : 9/18/2022 11:39 PM
    @Author  : Haodong Zhao
    
"""

import numpy as np


def graphnx_to_dict_spec(graph_nx):
    ret_dict = {}

    nodes_data = [data for _, data in graph_nx.nodes(data=True)]
    nodes_weight = np.array([data["weight"] for data in nodes_data], dtype=np.float64)
    nodes_position = np.array([data["pos"] for data in nodes_data], dtype=np.float64)
    nodes_start = np.array([data["is_start"] for data in nodes_data], dtype=np.int64)
    nodes_end = np.array([data["is_end"] for data in nodes_data], dtype=np.int64)
    nodes_in_sp = np.array([data["is_in_path"] for data in nodes_data], dtype=np.int64)

    ret_dict['nodes_feature'] = {
        'weights': nodes_weight,
        'pos': nodes_position,
        'is_start': nodes_start,
        'is_end': nodes_end,
        'is_in_path': nodes_in_sp
    }

    source_indices, target_indices, edges_data = zip(*graph_nx.edges(data=True))
    source_indices = np.array(source_indices, dtype=np.int64)
    target_indices = np.array(target_indices, dtype=np.int64)
    edges_weight = np.array([data["weight"] for data in edges_data], dtype=np.float64)
    edges_in_sp = np.array([data["is_in_path"] for data in edges_data], dtype=np.int64)

    ret_dict['edges_feature'] = {
        'source_indices': source_indices,
        'target_indices': target_indices,
        'weights': edges_weight,
        'is_in_path': edges_in_sp
    }

    return ret_dict


