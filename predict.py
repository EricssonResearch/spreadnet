"""
    Use the trained model to do the prediction.

    @Time    : 9/20/2022 8:28 PM
    @Author  : Haodong Zhao
    
"""
from random import randrange
import torch
from architecture import *
from utils import get_project_root, SPGraphDataset, data_to_input_label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    """
    load the entire model
    :param model_path: the path of the model(entire model)
    :return: the loaded model
    """
    model = torch.load(model_path)
    return model


def infer(model, graph_data):
    """
    do the inference

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


if __name__ == '__main__':
    # load model
    model = EncodeProcessDecode(
        node_in=3,
        edge_in=1,
        node_out=2,
        edge_out=2,
        latent_size=128,
        num_message_passing_steps=12,
        num_mlp_hidden_layers=2,
        mlp_hidden_size=128
    ).to(device)

    weight_base_path = str(get_project_root()) + "/weights/"
    # model_path = weight_base_path + "model_weights_best.pth"
    model_path = weight_base_path + "model_weights_ep_1950.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # test data
    dataset = SPGraphDataset(root=str(get_project_root()) + "/dataset/")
    graph = dataset.get(randrange(1000))
    node_label, edge_label = graph.label
    print("--- Ground_truth --- ")
    print('node: ', node_label)
    print('edge: ', edge_label)

    # predict
    node_infer, edge_infer = infer(model, graph.to(device))
    print("--- Predicted ---")
    print('node: ', node_infer)
    print('edge: ', edge_infer)
