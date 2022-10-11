"""Use the trained model to do the prediction.

Usage:
    python predict.py [--config config_file_path] [--model weight_file_path]

@Time    : 9/20/2022 8:28 PM
@Author  : Haodong Zhao
"""
import argparse
from random import randrange
from os import path as osp
import torch
import webdataset as wds

from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

default_yaml_path = osp.join(osp.dirname(__file__), "configs.yaml")
default_dataset_yaml_path = osp.join(osp.dirname(__file__), "../dataset_configs.yaml")
parser = argparse.ArgumentParser(description="Do predictions.")
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
which_model = args.model
configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)
train_configs = configs.train
model_configs = configs.model
data_configs = dataset_configs.data
dataset_path = osp.join(
    osp.dirname(__file__), "..", data_configs["dataset_path"]
).replace("\\", "/")


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
    model = EncodeProcessDecode(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        node_out=model_configs["node_out"],
        edge_out=model_configs["edge_out"],
        latent_size=model_configs["latent_size"],
        num_message_passing_steps=model_configs["num_message_passing_steps"],
        num_mlp_hidden_layers=model_configs["num_mlp_hidden_layers"],
        mlp_hidden_size=model_configs["mlp_hidden_size"],
    ).to(device)

    weight_base_path = osp.join(
        osp.dirname(__file__), train_configs["weight_base_path"]
    )
    # model_path = weight_base_path + "model_weights_best.pth"
    model_path = osp.join(weight_base_path, which_model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # test data
    dataset = (
        wds.WebDataset("file:" + dataset_path + "/processed/all_000000.tar")
        .decode(pt_decoder)
        .to_tuple(
            "pt",
        )
    )
    (graph,) = list(dataset)[randrange(len(list(dataset)))]
    node_label, edge_label = graph.y
    print("--- Ground_truth --- ")
    print("node: ", node_label)
    print("edge: ", edge_label)

    # predict
    node_infer, edge_infer = infer(model, graph.to(device))
    print("--- Predicted ---")
    print("node: ", node_infer)
    print("edge: ", edge_infer)
