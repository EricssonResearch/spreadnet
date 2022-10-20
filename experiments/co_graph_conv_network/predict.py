"""Use the trained GCN model to do the prediction.

Usage:
    python predict.py [--config config_file_path] [--model weight_file_path]

    @Time    : 10/4/2022 10:55 AM
    @Author  : Haodong Zhao
"""

import argparse
import os
from random import randrange
from os import path as osp
import torch
import webdataset as wds
from torch_geometric.transforms import LineGraph

from spreadnet.pyg_gnn.models import SPCoGCNet, SPCoDeepGCNet
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

default_yaml_path = osp.join(osp.dirname(__file__), "configs.yaml")
default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "../dataset_configs.yaml"
)

parser = argparse.ArgumentParser(description="Use trained GCN to do the predictions.")
parser.add_argument(
    "--config", default=default_yaml_path, help="Specify the path of the config file. "
)
parser.add_argument(
    "--dataset-config",
    default=default_dataset_yaml_path,
    help="Specify the path of the dataset config file. ",
)
parser.add_argument(
    "--model",
    default="model_weights_best.pth",
    help="Specify the model we want to use.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_path = args.config
dataset_yaml_path = args.dataset_config
configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)
which_model = args.model
train_configs = configs.train
model_configs = configs.model
data_configs = dataset_configs.data
dataset_path = osp.join(
    osp.dirname(__file__), "..", data_configs["dataset_path"]
).replace("\\", "/")

line_graph = LineGraph(force_directed=True)


def load_model(model_path):
    """load the entire model.

    :param model_path: the path of the model(entire model)
    :return: the loaded model
    """
    model = torch.load(model_path)
    return model


def infer(model, graph_data):
    """do the inference.

    :param model: the model to do the prediction
    :param graph_data: graph data from dataset
    :return: the shortest path info
    """
    model.eval()
    n_data = graph_data.to(device)
    e_data = n_data.clone()
    e_data = line_graph(data=e_data)

    n_index, e_index = n_data.edge_index, e_data.edge_index
    n_feats, e_feats = n_data.x, e_data.x

    nodes_output, edges_output = model(n_feats, n_index, e_feats, e_index)

    node_infer = torch.argmax(nodes_output, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edges_output, dim=-1).type(torch.int64)

    return node_infer, edge_infer


if __name__ == "__main__":
    # load model
    model_name = train_configs["which_model"]
    weight_base_path = train_configs["weight_base_path"]
    model = SPCoGCNet(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        hidden_channels=model_configs["hidden_channels"],
        num_layers=model_configs["num_layers"],
        node_out=model_configs["node_out"],
        edge_out=model_configs["edge_out"],
    ).to(device)

    if model_name == "deep":
        print("Prepare to infer with deep GCN model...")
        weight_base_path = train_configs["deep_weight_base_path"]
        model_configs = configs.deep_model
        model = SPCoDeepGCNet(
            node_in=model_configs["node_in"],
            edge_in=model_configs["edge_in"],
            hidden_channels=model_configs["hidden_channels"],
            num_layers=model_configs["num_layers"],
            node_out=model_configs["node_out"],
            edge_out=model_configs["edge_out"],
        ).to(device)
    else:
        print("Prepare to infer with GCN model...")

    weight_base_path = os.path.join(os.path.dirname(__file__), weight_base_path)
    model_path = osp.abspath(osp.join(weight_base_path, which_model))
    # print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # test data
    dataset = (
        wds.WebDataset("file:" + dataset_path + "/processed/all_000000.tar")
        .decode(pt_decoder)
        .to_tuple(
            "pt",
        )
    )
    (graph,) = list(dataset)[randrange(data_configs["dataset_size"])]
    node_label, edge_label = graph.y
    print("--- Ground_truth --- ")
    print("node: ", node_label)
    print("edge: ", edge_label)

    # predict
    node_infer, edge_infer = infer(model, graph.to(device))
    print("--- Predicted ---")
    print("node: ", node_infer)
    print("edge: ", edge_infer)
