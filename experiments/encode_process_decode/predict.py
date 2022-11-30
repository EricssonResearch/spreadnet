"""Use the trained model to do the prediction.

Usage:
    python predict.py [--config config_file_path] [--model weight_file_path]

@Time    : 9/20/2022 8:28 PM
@Author  : Haodong Zhao
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
from joblib import Parallel, delayed
from copy import deepcopy

from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.pyg_gnn.utils import get_correct_predictions
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.processor import process_nx, process_prediction
from spreadnet.datasets.data_utils.draw import draw_networkx
import spreadnet.utils.log_utils as log_utils
from spreadnet.datasets.data_utils.encoder import NpEncoder

torch.multiprocessing.set_start_method("spawn", force=True)

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
log_save_path = osp.join(osp.dirname(__file__), "logs").replace("\\", "/")
plot_size = 80

if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)


def load_model(model_path):
    """load the entire model.

    :param model_path: the path of the model(entire model)
    :return: the loaded model
    """
    model = torch.load(model_path)
    return model


def swap_start_end(graph_nx: nx.DiGraph):
    """Swap start and end node for bidirectional inference.

    :param graph_nx: networkx graph to predict

    :return: swapped graph
    """

    swapped = 0
    for (_, data) in graph_nx.nodes(data=True):
        if data["is_start"]:
            data["is_end"] = True
            data["is_start"] = False
            swapped += 1
        elif data["is_end"]:
            data["is_start"] = True
            data["is_end"] = False
            swapped += 1

        if swapped == 2:
            break

    rev_edges = []

    for (u, v, d) in graph_nx.edges(data=True):
        if not graph_nx.has_edge(v, u):
            rev_edges.append((v, u, {"weight": d["weight"], "is_in_path": False}))

    graph_nx.add_edges_from(rev_edges)

    return graph_nx


def aggregate_results(g1: nx.DiGraph, g2: nx.DiGraph):
    """Merge two results together favoring nodes/edges with higher
    probabilities into g1.

    :pre: two graphs must have the same structure

    :param g1: inferred graph
    :param g2: inferred reversed graph

    :return: aggregated graph
    """

    g1_nodes = g1.nodes(data=True)
    g2_nodes = g2.nodes(data=True)
    g2_edges = g2.edges(data=True)

    for idx, (_, d2) in enumerate(g2_nodes):
        if d2["is_in_path"]:
            g1_nodes[idx]["is_in_path"] = True
        if d2["probability"] > g1_nodes[idx]["probability"]:
            g1_nodes[idx]["probability"] = d2["probability"]

    for idx, (u, v, d2) in enumerate(g2_edges):
        d1 = g1.get_edge_data(v, u)

        if d2["is_in_path"]:
            d1["is_in_path"] = True
        if d2["probability"] > d1["probability"]:
            d1["probability"] = d2["probability"]

    return g1


def predict(model, graph):
    """Make prediction.

    :param model: model to be used
    :param graph: graph to predict

    :return: predictions, infers
    """
    graph = graph.to(device)

    node_true, edge_true = graph.y

    # predict
    (node_pred, edge_pred) = model(graph.x, graph.edge_index, graph.edge_attr)
    (infers, corrects) = get_correct_predictions(
        node_pred, edge_pred, node_true, edge_true
    )

    node_acc = corrects["nodes"] / graph.num_nodes
    edge_acc = corrects["edges"] / graph.num_edges

    preds = {"nodes": node_pred, "edges": edge_pred}

    print("\n--- Accuracies ---")
    print(f"Nodes: {corrects['nodes']}/{graph.num_nodes} = {node_acc}")
    print(f"Edges: {int(corrects['edges'])}/{graph.num_edges} = {edge_acc}\n")

    return preds, infers


if __name__ == "__main__":

    predict_logger = log_utils.init_file_console_logger(
        "predict_logger", log_save_path, "predict_MPNN"
    )

    predict_logger.info(f"Using {device} device...")
    try:
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

        model_path = osp.join(weight_base_path, which_model)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()

        raw_path = dataset_path + "/raw"
        raw_file_paths = list(map(os.path.basename, glob(raw_path + "/test.*.json")))

        with torch.no_grad():
            for idx, raw_file_path in enumerate(raw_file_paths):
                graphs_json = list(json.load(open(raw_path + "/" + raw_file_path)))
                for iidx, graph_json in enumerate(graphs_json):
                    print("\n\n")
                    print("Graph idx: ", idx + 1, ".", iidx + 1)

                    [graph_nx, graph_nx_r] = Parallel(
                        n_jobs=2, backend="multiprocessing", batch_size=1
                    )(
                        [
                            delayed(nx.node_link_graph)(graph_json),
                            delayed(swap_start_end)(nx.node_link_graph(graph_json)),
                        ]
                    )

                    [(preds, infers), (preds_r, infers_r)] = Parallel(
                        n_jobs=2, backend="multiprocessing", batch_size=1
                    )(
                        [
                            delayed(predict)(model, process_nx(graph_nx)),
                            delayed(predict)(model, process_nx(graph_nx_r)),
                        ]
                    )

                    [
                        (
                            pred_graph_nx,
                            truth_total_weight,
                            pred_total_weight,
                        ),
                        (
                            pred_graph_nx_r,
                            truth_total_weight_r,
                            pred_total_weight_r,
                        ),
                    ] = Parallel(n_jobs=2, backend="multiprocessing", batch_size=1)(
                        [
                            delayed(process_prediction)(graph_nx, preds, infers),
                            delayed(process_prediction)(graph_nx_r, preds_r, infers_r),
                        ]
                    )

                    aggregated_nx = aggregate_results(
                        deepcopy(pred_graph_nx), pred_graph_nx_r
                    )

                    plot_name = (
                        predictions_path + f"/{raw_file_path}.{idx + 1}.{iidx + 1}"
                    )

                    with open(f"{plot_name}.json", "w") as outfile:
                        json.dump(
                            [nx.node_link_data(pred_graph_nx)], outfile, cls=NpEncoder
                        )

                    with open(f"{plot_name}_r.json", "w") as outfile:
                        json.dump(
                            [nx.node_link_data(pred_graph_nx_r)], outfile, cls=NpEncoder
                        )

                    print("Drawing comparison...")
                    fig = plt.figure(figsize=(plot_size, plot_size))
                    draw_networkx(
                        "Truth",
                        fig,
                        graph_nx,
                        1,
                        4,
                        per_row=2,
                    )
                    draw_networkx(
                        "Pred",
                        fig,
                        pred_graph_nx,
                        2,
                        4,
                        "probability",
                        "probability",
                        per_row=2,
                    )
                    draw_networkx(
                        "Pred Rev",
                        fig,
                        pred_graph_nx_r,
                        3,
                        4,
                        "probability",
                        "probability",
                        per_row=2,
                    )

                    draw_networkx(
                        "Aggregated",
                        fig,
                        aggregated_nx,
                        4,
                        4,
                        "probability",
                        "probability",
                        per_row=2,
                    )

                    plt.savefig(f"{plot_name}.jpg", pad_inches=0, bbox_inches="tight")
                    plt.clf()
                    print("Image saved at ", plot_name)

                    # input("Press enter to predict another graph")
    except Exception as e:
        predict_logger.exception(e)

    logging.shutdown(handlerList="train_local_logger")
