"""Use the trained model to do the prediction.

@Time    : 9/20/2022 8:28 PM
@Author  : Haodong Zhao
"""
from random import randrange

import torch

from models import EncodeProcessDecode
from utils import get_project_root, SPGraphDataset, data_to_input_label, yaml_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yaml_path = str(get_project_root()) + "/configs.yaml"
configs = yaml_parser(yaml_path)
train_configs = configs.train
model_configs = configs.model
data_configs = configs.data


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
    input_data, labels = data_to_input_label(graph_data)
    nodes_data, edges_data = input_data
    edge_index = graph_data.edge_index

    nodes_output, edges_output = model(nodes_data, edge_index, edges_data)

    node_infer = torch.argmax(nodes_output, dim=-1).type(torch.int64)
    edge_infer = torch.argmax(edges_output, dim=-1).type(torch.int64)

    return node_infer, edge_infer


if __name__ == "__main__":
    which_model = "model_weights_ep_1950.pth"

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

    weight_base_path = str(get_project_root()) + train_configs["weight_base_path"]
    # model_path = weight_base_path + "model_weights_best.pth"
    model_path = weight_base_path + which_model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # test data
    dataset = SPGraphDataset(
        root=str(get_project_root()) + data_configs["dataset_path"]
    )
    graph = dataset.get(randrange(data_configs["dataset_size"]))
    node_label, edge_label = graph.label
    print("--- Ground_truth --- ")
    print("node: ", node_label)
    print("edge: ", edge_label)

    # predict
    node_infer, edge_infer = infer(model, graph.to(device))
    print("--- Predicted ---")
    print("node: ", node_infer)
    print("edge: ", edge_infer)
