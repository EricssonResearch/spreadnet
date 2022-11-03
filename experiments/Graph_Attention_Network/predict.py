"""Use the trained model to do the prediction.

Usage:
    python predict.py [--config config_file_path] [--model weight_file_path]

@Time    : 10/25/2022 1:05 PM
@Author  : Haoyuan Li
"""
import argparse
from os import path as osp
import torch
import networkx as nx
import json
import os
from glob import glob
import matplotlib.pyplot as plt

from spreadnet.pyg_gnn.models import SPGATNet
from spreadnet.utils import yaml_parser
from spreadnet.pyg_gnn.loss.loss import get_infers
from spreadnet.datasets.data_utils.processor import process_nx, process_prediction
from spreadnet.datasets.data_utils.draw import draw_networkx


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
predictions_path = osp.join(osp.dirname(__file__), "predictions").replace("\\", "/")

if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)


def load_model(model_path):
    """load the entire model.

    :param model_path: the path of the model(entire model)
    :return: the loaded model
    """
    model = torch.load(model_path)
    return model


def predict(model, graph):
    """Make prediction.

    :param model: model to be used
    :param graph: graph to predict

    :return: predictions, infers
    """
    graph = graph.to(device)

    node_true, edge_true = graph.y

    # predict
    (node_pred, edge_pred) = model(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        return_attention_weights=model_configs["return_attention_weights"],
    )
    (infers, corrects) = get_infers(node_pred, edge_pred, node_true, edge_true)

    node_acc = corrects["nodes"] / graph.num_nodes
    edge_acc = corrects["edges"] / graph.num_edges

    preds = {"nodes": node_pred, "edges": edge_pred}

    # print("--- Node ---")
    # print("Truth:     ", node_true.tolist())
    # print("Predicted: ", infers["nodes"].cpu().tolist())

    # print("\n--- Edge ---")
    # print("Truth:     ", edge_true.tolist())
    # print("Predicted: ", infers["edges"].cpu().tolist())

    print("\n--- Accuracies ---")
    print(f"Nodes: {corrects['nodes']}/{graph.num_nodes} = {node_acc}")
    print(f"Edges: {int(corrects['edges'])}/{graph.num_edges} = {edge_acc}")

    return preds, infers


if __name__ == "__main__":
    # load model
    model = SPGATNet(
        num_hidden_layers=model_configs["num_hidden_layers"],
        in_channels=model_configs["in_channels"],
        hidden_channels=model_configs["hidden_channels"],
        out_channels=model_configs["out_channels"],
        heads=model_configs["heads"],
        # dropout=model_configs[""],
        add_self_loops=model_configs["add_self_loops"],
        bias=model_configs["bias"],
        edge_hidden_channels=model_configs["edge_hidden_channels"],
        edge_out_channels=model_configs["edge_out_channels"],
        edge_num_layers=model_configs["edge_num_layers"],
        edge_bias=model_configs["edge_bias"],
        encode_node_in=model_configs["encode_node_in"],
        encode_edge_in=model_configs["encode_edge_in"],
        encode_node_out=model_configs["encode_node_out"],
        encode_edge_out=model_configs["encode_edge_out"],
    ).to(device)

    weight_base_path = osp.join(
        osp.dirname(__file__), train_configs["weight_base_path"]
    )

    model_path = osp.join(weight_base_path, which_model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    raw_path = dataset_path + "/raw"
    raw_file_paths = list(map(os.path.basename, glob(raw_path + "/test.*.json")))

    for raw_file_path in raw_file_paths:
        graphs_json = list(json.load(open(raw_path + "/" + raw_file_path)))
        for idx, graph_json in enumerate(graphs_json):
            print("\n\n")
            print("Graph idx: ", idx + 1)

            graph_nx = nx.node_link_graph(graph_json)
            (preds, infers) = predict(model, process_nx(graph_nx))
            (pred_graph_nx, truth_total_weight, pred_total_weight) = process_prediction(
                graph_nx, preds, infers
            )

            print(f"Truth weights: {truth_total_weight}")
            print(f"Pred weights: {pred_total_weight}")

            print("Drawing comparison...")
            fig = plt.figure(figsize=(80, 40))
            draw_networkx(
                f"Truth, total edge weights: {round(truth_total_weight, 2)}",
                fig,
                graph_nx,
                1,
                2,
            )
            draw_networkx(
                f"Prediction, total edge weights: {round(pred_total_weight, 2)}",
                fig,
                pred_graph_nx,
                2,
                2,
                "probability",
                "probability",
            )
            plot_name = predictions_path + f"/{raw_file_path}.{idx + 1}.jpg"
            plt.savefig(plot_name, pad_inches=0, bbox_inches="tight")
            plt.clf()
            print("Image saved at ", plot_name)

            input("Press enter to predict another graph")