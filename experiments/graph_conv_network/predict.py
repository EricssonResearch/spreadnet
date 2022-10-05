"""Use the trained GCN model to do the prediction.

Usage:
    python predict.py [--config config_file_path] [--model weight_file_path]

    @Time    : 10/4/2022 10:55 AM
    @Author  : Haodong Zhao
"""

import argparse
from random import randrange
from os import path as osp
import torch
import webdataset as wds

from spreadnet.pyg_gnn.models import SPGCNet
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

default_yaml_path = osp.join(osp.dirname(__file__), "configs.yaml")
parser = argparse.ArgumentParser(description="Use trained GCN to do the predictions.")
parser.add_argument(
    "--config", default=default_yaml_path, help="Specify the path of the config file. "
)
parser.add_argument(
    "--model",
    default="model_weights_best.pth",
    help="Specify the model we want to use.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_path = args.config
which_model = args.model
configs = yaml_parser(yaml_path)
train_configs = configs.train
model_configs = configs.model
data_configs = configs.data
dataset_path = osp.join(osp.dirname(__file__), data_configs["dataset_path"]).replace(
    "\\", "/"
)


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
    nodes_output, edges_output = model(
        graph_data.x, graph_data.edge_index, graph_data.edge_attr
    )

    node_infer = torch.argmax(nodes_output, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edges_output, dim=-1).type(torch.int64)

    return node_infer, edge_infer


if __name__ == "__main__":

    # load model
    model = SPGCNet(
        node_gcn_in_channels=model_configs["node_gcn_in_channels"],
        node_gcn_num_hidden_layers=model_configs["node_gcn_num_hidden_layers"],
        node_gcn_hidden_channels=model_configs["node_gcn_hidden_channels"],
        node_gcn_out_channels=model_configs["node_gcn_out_channels"],
        node_gcn_use_normalization=model_configs["node_gcn_use_normalization"],
        node_gcn_use_bias=model_configs["node_gcn_use_bias"],
        edge_mlp_in_channels=model_configs["edge_mlp_in_channels"],
        edge_mlp_bias=model_configs["edge_mlp_bias"],
        edge_mlp_hidden_channels=model_configs["edge_mlp_hidden_channels"],
        edge_mlp_num_layers=model_configs["edge_mlp_num_layers"],
        edge_mlp_out_channels=model_configs["edge_mlp_out_channels"],
    ).to(device)

    weight_base_path = osp.join(
        osp.dirname(__file__), train_configs["weight_base_path"]
    )

    # model_path = weight_base_path + "model_weights_best.pth"
    model_path = osp.abspath(osp.join(weight_base_path, which_model))
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
