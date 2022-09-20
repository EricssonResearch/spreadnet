"""

    @Time    : 9/20/2022 8:28 PM
    @Author  : Haodong Zhao
    
"""
from random import randrange
import torch
from utils import get_project_root, SPGraphDataset

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
    weight_base_path = str(get_project_root()) + "/weights/"
    model_path = weight_base_path + "model_weights_best.pth"
    model = load_model(model_path)

    # test data
    dataset = SPGraphDataset(root=str(get_project_root()) + "/dataset/")
    graph = dataset.get(randrange(1000))
    node_label, edge_label = graph.label
    print("--- Ground_truth --- ")
    print(node_label)
    print(edge_label)

    # predict
    node_infer, edge_infer = infer(model, graph.to(device))
    print("--- Predicted ---")
    print(node_infer)
    print(edge_infer)
