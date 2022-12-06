import webdataset as wds
import json
import torch
from networkx import node_link_graph, DiGraph
from torch_geometric.data import Data
import os
from glob import glob

from spreadnet.datasets.data_utils.convertor import graphnx_to_dict_spec
from spreadnet.datasets.data_utils.encoder import pt_encoder


def process_nx(graph_nx: DiGraph):
    """Process networkx graph to PyG Data.

    Args:
        graph_nx: networkx graph

    Returns:
        PyG Data
    """
    graph_dict = graphnx_to_dict_spec(graph_nx)
    # Get ground truth labels.
    node_tensor = torch.tensor(graph_dict["nodes_feature"]["is_in_path"])
    node_labels = node_tensor.type(torch.int64)

    edge_tensor = torch.tensor(graph_dict["edges_feature"]["is_in_path"])
    edge_labels = edge_tensor.type(torch.int64)

    nodes_data = [data for _, data in graph_nx.nodes(data=True)]
    nodes_weight = torch.tensor(
        [data["weight"] for data in nodes_data], dtype=torch.float
    ).view(-1, 1)
    nodes_is_start = torch.tensor(
        [data["is_start"] for data in nodes_data], dtype=torch.int
    ).view(-1, 1)
    nodes_is_end = torch.tensor(
        [data["is_end"] for data in nodes_data], dtype=torch.int
    ).view(-1, 1)
    nodes_pos = torch.tensor([data["pos"] for data in nodes_data], dtype=torch.float)
    x = torch.cat((nodes_weight, nodes_is_start, nodes_is_end), 1)

    _, _, edges_data = zip(*graph_nx.edges(data=True))
    edges_weight = torch.tensor(
        [data["weight"] for data in edges_data], dtype=torch.float
    ).view(-1, 1)

    # get edge source and target nodes
    edge_index_data = [list(tpl) for tpl in graph_nx.edges]
    edge_index_t = torch.tensor(edge_index_data, dtype=torch.long)

    # get edge_index from graph_nx
    edge_index = edge_index_t.t().contiguous()

    return Data(
        edge_index=edge_index,
        pos=nodes_pos,
        x=x,
        edge_attr=edges_weight,
        y=(node_labels, edge_labels),
        edge_data=edge_index_t,
    )


def process_raw_data_folder(dataset_path, output_name, raw_matcher=""):
    """Convert json to networkx graph and write as tar file.

    Args:
        dataset_path: path to json graph folder
        output_name: output file prefix
        raw_matcher: pattern to match raw files in dataset_path

    Returns:
        None
    """
    raw_path = dataset_path + "/raw"
    processed_path = dataset_path + "/processed"

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    idx = 0
    sink = wds.ShardWriter(
        os.path.abspath(processed_path + f"/{output_name}_%06d.tar"),
        maxsize=2e9,
        encoder=pt_encoder,
        compress=True,
    )  # 2GB per shard
    raw_file_paths = list(
        map(os.path.basename, glob(raw_path + f"/{raw_matcher}*.json"))
    )

    for raw_file_path in raw_file_paths:
        graphs_json = list(json.load(open(raw_path + "/" + raw_file_path)))

        for graph_json in graphs_json:
            graph_nx = node_link_graph(graph_json)
            data = process_nx(graph_nx)

            sink.write(
                {
                    "__key__": "data_%06d" % idx,
                    "pt": data,
                }
            )
            idx += 1

    sink.close()
    print(f"Size of the dataset {output_name}: " + str(idx))
