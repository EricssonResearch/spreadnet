import webdataset as wds
import json
import torch
from networkx import node_link_graph
from torch_geometric.data import Data

from spreadnet.datasets.data_utils.convertor import graphnx_to_dict_spec
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
        os.path.abspath(processed_path + "/all_%06d.tar"),
        maxsize=2e9,
        encoder=pt_encoder,
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
            nodes_pos = torch.tensor(
                [data["pos"] for data in nodes_data], dtype=torch.float
            )
            x = torch.cat((nodes_weight, nodes_is_start, nodes_is_end), 1)

            _, _, edges_data = zip(*graph_nx.edges(data=True))
            edges_weight = torch.tensor(
                [data["weight"] for data in edges_data], dtype=torch.float
            ).view(-1, 1)

            # get edge_index from graph_nx
            edge_index_data = [list(tpl) for tpl in graph_nx.edges]
            edge_index_t = torch.tensor(edge_index_data, dtype=torch.long)
            edge_index = edge_index_t.t().contiguous()

            data = Data(edge_index=edge_index)
            data.pos = nodes_pos
            data.x = x
            data.edge_attr = edges_weight
            data.y = (node_labels, edge_labels)
            # print(data)

            # remove node and edge features
            # for (n, d) in graph_nx.nodes(data=True):
            #     del d["is_in_path"]
            #     del d["weight"]
            #     del d["is_end"]
            #     del d["is_start"]
            #
            # for (s, e, d) in graph_nx.edges(data=True):
            #     del d["is_in_path"]
            #     del d["weight"]

            # data_ori = from_networkx(graph_nx)
            # data_ori.x = x
            # data_ori.edge_attr = edges_weight
            # data_ori.y = (node_labels, edge_labels)
            # print(data_ori)

            # print(torch.eq(data_ori.edge_index, data.edge_index))
            # print(torch.eq(data_ori.x, data.x))
            # print(torch.eq(data_ori.edge_attr, data.edge_attr))
            # print(torch.eq(data_ori.y[0], data.y[0]))
            # print(torch.eq(data_ori.y[1], data.y[1]))

            sink.write(
                {
                    "__key__": "data_%06d" % idx,
                    "pt": data,
                }
            )
            idx += 1

    sink.close()
    print("Size of the dataset: " + str(idx))
