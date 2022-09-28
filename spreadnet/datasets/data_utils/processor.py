import webdataset as wds
import json
import torch
from networkx import node_link_graph
from spreadnet.datasets.data_utils.convertor import graphnx_to_dict_spec
from torch_geometric.utils import from_networkx
from spreadnet.datasets.data_utils.encoder import pt_encoder
import os
from glob import glob


def process(dataset_path):
    raw_path = dataset_path + "/raw"
    processed_path = dataset_path + "/processed"

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    idx = 0
    sink = wds.ShardWriter(
        processed_path + "/all_%06d.tar", maxsize=2e9, encoder=pt_encoder
    )  # 2GB per shard
    raw_file_paths = list(map(os.path.basename, glob(raw_path + "/*.json")))

    for raw_file_path in raw_file_paths:
        graphs_json = list(json.load(open(raw_path + "/" + raw_file_path)))

        for graph_json in graphs_json:
            graph_nx = node_link_graph(graph_json)
            graph_dict = graphnx_to_dict_spec(graph_nx)
            # Get ground truth labels.
            node_tensor = torch.tensor(graph_dict["nodes_feature"]["is_in_path"])
            node_labels = node_tensor.type(torch.int64)

            edge_tensor = torch.tensor(graph_dict["edges_feature"]["is_in_path"])
            edge_labels = edge_tensor.type(torch.int64)

            # remove node and edge features
            for (n, d) in graph_nx.nodes(data=True):
                del d["is_in_path"]

            for (s, e, d) in graph_nx.edges(data=True):
                del d["is_in_path"]

            data = from_networkx(graph_nx)
            data.label = (node_labels, edge_labels)

            sink.write(
                {
                    "__key__": "data_%06d" % idx,
                    "pt": data,
                }
            )
            idx += 1

    sink.close()
    print("Size of the dataset: " + str(idx))
