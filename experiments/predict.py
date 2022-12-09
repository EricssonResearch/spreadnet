"""Use the trained model to do the prediction.

Args:
    --model "MPNN|DeepGCN|GAT|DeepCoGCN"
    --dataset-config dataset_config_path, defaults: ./dataset_configs.yaml
    --weight weight_file_name, defaults: model_weights_best.pth
"""
import argparse
from os import path as osp
import torch
import networkx as nx
import json
import os
from glob import glob
import matplotlib.pyplot as plt
import logging
from copy import deepcopy

from spreadnet.pyg_gnn.utils import get_correct_predictions
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.processor import process_nx
from spreadnet.datasets.data_utils.draw import draw_networkx
import spreadnet.utils.log_utils as log_utils
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.utils.post_processor import (
    process_prediction,
    swap_start_end,
    aggregate_results,
    probability_first_search,
    get_start_end_nodes,
    apply_path_on_graph,
)
from spreadnet.utils.model_loader import load_model

torch.multiprocessing.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser(description="Do predictions.")

parser.add_argument(
    "--model", help="Specify which model you want to train. ", required=True
)
parser.add_argument(
    "--dataset-config",
    default=osp.join(osp.dirname(__file__), "dataset_configs.yaml"),
    help="Specify the path of the dataset config file. ",
)
parser.add_argument(
    "--weight",
    default="model_weights_best.pth",
    help="Specify the weight we want to use.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_yaml_path = args.dataset_config
which_model = args.model
which_weight = args.weight

if which_model == "MPNN":
    folder = "encode_process_decode"
elif which_model == "DeepCoGCN":
    folder = "co_graph_conv_network"
elif which_model == "DeepGCN":
    folder = "deep_graph_conv_network"
elif which_model == "GAT":
    folder = "graph_attention_network"
else:
    print("Please check the model name. -h for more details.")
    exit()

yaml_path = os.path.join(os.path.dirname(__file__), folder, "configs.yaml")
weight_base_path = os.path.join(os.path.dirname(__file__), folder, "weights")
predictions_path = osp.join(osp.dirname(__file__), folder, "predictions").replace(
    "\\", "/"
)
log_save_path = osp.join(osp.dirname(__file__), folder, "logs").replace("\\", "/")

configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)
model_configs = configs.model
data_configs = dataset_configs.data
dataset_path = osp.join(osp.dirname(__file__), data_configs["dataset_path"]).replace(
    "\\", "/"
)
plot_size = 20

if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)


def predict(model, data):
    """Make prediction.

    :param model: model to be used
    :param data: graph data to predict

    :return: predictions
    """
    node_true, edge_true = data.y

    # predict
    if which_model == "GAT":
        (node_pred, edge_pred) = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            return_attention_weights=model_configs["return_attention_weights"],
        )
    else:
        (node_pred, edge_pred) = model(data.x, data.edge_index, data.edge_attr)

    (_, corrects) = get_correct_predictions(node_pred, edge_pred, node_true, edge_true)

    node_acc = corrects["nodes"] / data.num_nodes
    edge_acc = corrects["edges"] / data.num_edges

    preds = {"nodes": node_pred, "edges": edge_pred}

    print("\n--- Accuracies ---")
    print(f"Nodes: {corrects['nodes']}/{data.num_nodes} = {node_acc}")
    print(f"Edges: {int(corrects['edges'])}/{data.num_edges} = {edge_acc}\n")

    return preds


if __name__ == "__main__":

    predict_logger = log_utils.init_file_console_logger(
        "predict_logger", log_save_path, f"predict_{which_model}"
    )

    predict_logger.info(f"Using {device} device...")
    try:
        # load model
        print("Using model: ", which_model)
        model = load_model(which_model, model_configs, device)

        weight_path = osp.join(weight_base_path, which_weight)
        model.load_state_dict(
            torch.load(weight_path, map_location=torch.device(device))
        )
        model.eval()

        raw_path = dataset_path + "/raw"
        raw_file_paths = list(map(os.path.basename, glob(raw_path + "/test.*.json")))

        with torch.no_grad():
            for idx, raw_file_path in enumerate(raw_file_paths):
                graphs_json = list(json.load(open(raw_path + "/" + raw_file_path)))
                for iidx, graph_json in enumerate(graphs_json):
                    print("\n\n")
                    print("Graph: ", f"{idx + 1}.{iidx + 1}")
                    print(raw_file_path)
                    graph_nx = nx.node_link_graph(graph_json)
                    graph_nx_r = deepcopy(graph_nx)

                    (start_node, end_node) = get_start_end_nodes(
                        graph_nx.nodes(data=True)
                    )
                    swap_start_end(graph_nx_r, start_node, end_node)

                    graph_data = process_nx(graph_nx)
                    graph_data_r = process_nx(graph_nx_r)

                    graph_data.to(device)
                    graph_data_r.to(device)

                    preds = predict(model, graph_data)
                    preds_r = predict(model, graph_data_r)

                    (pred_graph_nx, truth_total_weight) = process_prediction(
                        graph_nx, preds
                    )
                    (pred_graph_nx_r, truth_total_weight_r) = process_prediction(
                        graph_nx_r, preds_r
                    )

                    aggregated_nx = aggregate_results(
                        deepcopy(pred_graph_nx), pred_graph_nx_r
                    )

                    print("Truth Edge Weights: ", round(truth_total_weight, 3))

                    (is_path_complete, prob_path) = probability_first_search(
                        deepcopy(pred_graph_nx), start_node, end_node
                    )

                    applied_nx, pred_edge_weights = apply_path_on_graph(
                        deepcopy(pred_graph_nx), prob_path, True
                    )

                    print(
                        "Prob Path on Pred: ",
                        is_path_complete,
                        round(pred_edge_weights, 3),
                        prob_path,
                    )

                    (is_path_complete_a, prob_path_a) = probability_first_search(
                        deepcopy(aggregated_nx), start_node, end_node
                    )

                    applied_nx_a, pred_edge_weights_a = apply_path_on_graph(
                        deepcopy(aggregated_nx), prob_path_a, True
                    )

                    print(
                        "Prob Path on Aggregated: ",
                        is_path_complete_a,
                        round(pred_edge_weights_a, 3),
                        prob_path_a,
                    )

                    plot_name = predictions_path + f"/pred.{raw_file_path}"

                    with open(f"{plot_name}.json", "w") as outfile:
                        json.dump(
                            [nx.node_link_data(pred_graph_nx)], outfile, cls=NpEncoder
                        )

                    with open(f"{plot_name}.r.json", "w") as outfile:
                        json.dump(
                            [nx.node_link_data(pred_graph_nx_r)], outfile, cls=NpEncoder
                        )

                    print("\nDrawing comparison...")
                    fig = plt.figure(figsize=(plot_size * 2, plot_size * 3))
                    draw_networkx(
                        f"Truth, Edge Weights: {truth_total_weight}",
                        fig,
                        graph_nx,
                        1,
                        6,
                        per_row=2,
                    )
                    draw_networkx(
                        "Pred",
                        fig,
                        pred_graph_nx,
                        2,
                        6,
                        "probability",
                        "probability",
                        per_row=2,
                    )
                    draw_networkx(
                        "Pred Rev",
                        fig,
                        pred_graph_nx_r,
                        3,
                        6,
                        "probability",
                        "probability",
                        per_row=2,
                    )

                    draw_networkx(
                        "Aggregated",
                        fig,
                        aggregated_nx,
                        4,
                        6,
                        "probability",
                        "probability",
                        per_row=2,
                    )

                    draw_networkx(
                        f"Prob Walk on Pred, Edge Weights: {pred_edge_weights}",
                        fig,
                        applied_nx,
                        5,
                        6,
                        "default",
                        "probability",
                        per_row=2,
                    )

                    draw_networkx(
                        f"Prob Walk on Aggregated, Edge Weights: {pred_edge_weights_a}",
                        fig,
                        applied_nx_a,
                        6,
                        6,
                        "default",
                        "probability",
                        per_row=2,
                    )

                    fig.tight_layout()

                    print("Saving image...")
                    plt.savefig(f"{plot_name}.jpg", pad_inches=0, bbox_inches="tight")
                    plt.clf()
                    print("Image saved at ", plot_name)

                    # input("Press enter to predict another graph")
    except Exception as e:
        predict_logger.exception(e)

    logging.shutdown()
