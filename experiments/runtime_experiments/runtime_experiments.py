# from dijkstra_algorithm import single_source_dijkstra
# from dijkstra_runner import shortest_path

from spreadnet.dijkstra_memoization import dijkstra_runner

# from spreadnet.datasets.data_utils.decoder import pt_decoder

import networkx as nx

# import webdataset as wds
# import timeit
from time import process_time, time

# import copy
import os
import json

# import sys
# import numpy as np
import pandas as pd
import argparse
from os import path as osp
import torch

# from glob import glob

# import matplotlib.pyplot as plt

from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.pyg_gnn.utils import get_correct_predictions
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.processor import process_nx, process_prediction

# from spreadnet.datasets.data_utils.draw import draw_networkx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_yaml_path = osp.join(
    osp.dirname(__file__), "../encode_process_decode/configs.yaml"
)
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
yaml_path = args.config
configs = yaml_parser(yaml_path)
model_configs = configs.model
train_configs = configs.train
which_model = args.model


# dataset_path = os.path.join(
#    os.path.dirname(__file__), "..", data_configs["dataset_path"]
# ).replace("\\", "/")


"""
JSON file layout example
    {
"directed": true,
    "multigraph": false,
    "graph": {},
    "nodes": [
        {
            "weight": 1.4213646962011073,
            "pos": [
                0.8442657485810173,
                0.8579456176227568
            ],
            "is_end": true,
            "is_start": false,
            "is_in_path": true,
            "id": 0  """


default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "../../experiments/dataset_configs.yaml"
)


def predict(model, graph):
    """Make prediction.

    :param model: model to be used
    :param graph: graph to predict

    :return: predictions, infers
    """
    graph = graph.to(device)

    node_true, edge_true = graph.y

    # predict
    t1_start = process_time()
    wall_time_start = time()
    (node_pred, edge_pred) = model(graph.x, graph.edge_index, graph.edge_attr)
    t1_stop = process_time()
    wall_time_stop = time()
    runtime_process = t1_stop - t1_start
    runtime_walltime = wall_time_stop - wall_time_start
    # stop
    (infers, corrects) = get_correct_predictions(
        node_pred, edge_pred, node_true, edge_true
    )

    # node_acc = corrects["nodes"] / graph.num_nodes
    # edge_acc = corrects["edges"] / graph.num_edges

    preds = {"nodes": node_pred, "edges": edge_pred}

    # Commenting out for time analysis
    # print("\n--- Accuracies ---")
    # print(f"Nodes: {corrects['nodes']}/{graph.num_nodes} = {node_acc}")
    # print(f"Edges: {int(corrects['edges'])}/{graph.num_edges} = {edge_acc}")

    return preds, infers, runtime_process, runtime_walltime


def divide_datasets(no_process=1):
    experiments_dir = "runtime_increasing_size_experiment_data"
    datasets = list()
    for path in os.listdir(experiments_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(experiments_dir, path)):
            datasets.append(path)
    # return np.array_split(datasets, no_process)
    return datasets


dataset_path = os.path.join(
    os.path.dirname(__file__),
    "../../experiments/runtime_experiments/runtime_increasing_size_experiment_data",
)
file_name_list = list()
hashing_runtime_test_list = list()
dijkstra_runtime_list = list()
dijksta_runtime_msp_memo_list = list()
dijkstra_runtime_memo_list = list()
message_passing_runtime_list = list()
networkx_dijkstra_runtime_list = list()

hashing_walltime_test_list = list()
dijkstra_walltime_list = list()
dijksta_walltime_msp_memo_list = list()
dijkstra_walltime_memo_list = list()
message_passing_walltime_list = list()
networkx_dijkstra_walltime_list = list()

walltime_test_list = list()
number_nodes_list = list()
number_edges_list = list()
start_node_list = list()
end_node_list = list()
function_name_list = list()
min_node_list = list()
max_node_list = list()
theta_list = list()
graph_data = {
    "file_name": file_name_list,
    "min_nodes": min_node_list,
    "max_nodes": max_node_list,
    "Hashing Graphs RunTime": hashing_runtime_test_list,
    "Dijkstra Full Path and Memoization Runtime": dijksta_runtime_msp_memo_list,
    "Dijkstra With Memoization Runtime": dijkstra_runtime_memo_list,
    "Message Passing GNN Runtime": message_passing_runtime_list,
    "Networkx Dijkstra Runtime": networkx_dijkstra_runtime_list,
    "Hashing Graphs Walltime": hashing_runtime_test_list,
    "Dijkstra Full Path and Memoization Walltime": dijksta_runtime_msp_memo_list,
    "Dijkstra With Memoization Walltime": dijkstra_runtime_memo_list,
    "Message Passing GNN Walltime": message_passing_runtime_list,
    "Networkx Dijkstra Walltime": networkx_dijkstra_walltime_list,
}


# def runTimeTest(graphs, start, end):
#    for g in graphs:
#        #print( nx.single_source_dijkstra(g, start, end) )
#        print( nx.single_source_dijkstra(g, 1, g.number_of_nodes() - 1) )
#    return


def main():
    ds_split = (
        divide_datasets()
    )  # list of all the .json files to be used in runtime tests

    for ds in ds_split:
        # print(ds)
        # f = open(dataset_path + "/raw/random.102.100-110.20.json")
        # path = dataset_path + "/" + ds
        # print(path)
        f = open(dataset_path + "/" + ds)
        dataset = json.load(f)
        all_graphs = list()
        file_parts = ds.split("_")
        min_node_list.append(file_parts[2])
        max_node_list.append(file_parts[3])
        theta_list.append(file_parts[4])

        # start_nodes = list()
        # end_nodes = list()
        # file_name_list = list()
        file_name_list.append(ds)
        # graph_data = {'file_name': file_name_list}
        # gd = pd.DataFrame(graph_data)
        # gd.to_csv
        # print(gd)

        for d in dataset:
            g = {}
            g["nxdata"] = nx.node_link_graph(d)
            g["hashed_graph"] = dijkstra_runner.hash_graph_weisfeiler(g["nxdata"])
            count_nodes = 0
            count_edges = 0
            g["min_node_graph"] = 0
            g["max_node_graph"] = 0
            g["min_edge_graph"] = 0
            g["max_edge_graph"] = 0

            for n in d["nodes"]:
                count_nodes = count_nodes + 1
                if n["is_start"]:
                    g["start"] = n["id"]
                if n["is_end"]:
                    g["end"] = n["id"]
            for e in d["links"]:
                count_edges = count_edges + 1
            # if count_edges < g["min_edge_graph"]:
            #    min_edge_graph = count_edges
            # if count_edges > g["max_edge_graph"]:
            #    max_edge_graph = count_edges
            if count_nodes < g["min_node_graph"]:
                g["min_node_graph"] = count_nodes
            if count_nodes > g["max_node_graph"]:
                g["max_node_graph"] = count_nodes

            all_graphs.append(g)

        print("Number of graphs: ", len(all_graphs))

        # hashing graphs only runtime test
        function_name_list.append("hashing_graphs_only")

        dijkstra_runner.clear_memo_table()
        for i in range(1):
            total_time = 0
            total_wall_time = 0
            for g in all_graphs:
                t1_start = process_time()
                wall_time_start = time()
                dijkstra_runner.hash_graph_weisfeiler(g["nxdata"])
                t1_stop = process_time()
                wall_time_stop = time()
                runtime = t1_stop - t1_start
                walltime = wall_time_stop - wall_time_start
                total_time = total_time + runtime
                total_wall_time = total_wall_time + walltime

            hashing_runtime_test_list.append(total_time)
            hashing_walltime_test_list.append(total_wall_time)

        dijkstra_runner.clear_memo_table()

        # dijkstra minimum spanning tree with memoization only
        # one path saved runtime test

        dijkstra_runner.clear_memo_table()
        for i in range(1):
            total_time = 0
            total_wall_time = 0
            for g in all_graphs:
                t1_start = process_time()
                wall_time_start = time()
                dijkstra_runner.shortest_path_single(
                    g["nxdata"], g["hashed_graph"], g["start"], g["end"]
                )
                t1_stop = process_time()
                wall_time_stop = time()
                runtime = t1_stop - t1_start
                walltime = wall_time_stop - wall_time_start
                total_time = total_time + runtime
                total_wall_time = total_wall_time + walltime

            dijkstra_runtime_memo_list.append(total_time)
            dijkstra_walltime_memo_list.append(total_wall_time)

        # networkx Dijkstra algorithm

        dijkstra_runner.clear_memo_table()
        for i in range(1):
            total_time = 0
            total_wall_time = 0
            for g in all_graphs:
                t1_start = process_time()
                wall_time_start = time()
                nx.single_source_dijkstra(g["nxdata"], g["start"], g["end"])
                t1_stop = process_time()
                wall_time_stop = time()
                runtime = t1_stop - t1_start
                walltime = wall_time_stop - wall_time_start
                total_time = total_time + runtime
                total_wall_time = total_wall_time + walltime

            networkx_dijkstra_runtime_list.append(total_time)
            networkx_dijkstra_walltime_list.append(total_wall_time)

        # dijkstra minimum spanning tree with memoization runtime test
        dijkstra_runner.clear_memo_table()
        for i in range(1):
            total_time = 0
            total_wall_time = 0
            for g in all_graphs:
                t1_start = process_time()
                wall_time_start = time()
                nx.single_source_dijkstra(g["nxdata"], g["start"], g["end"])
                t1_stop = process_time()
                wall_time_stop = time()
                runtime = t1_stop - t1_start
                walltime = wall_time_stop - wall_time_start
                total_time = total_time + runtime
                total_wall_time = total_wall_time + walltime

            dijksta_runtime_msp_memo_list.append(total_time)
            dijksta_walltime_msp_memo_list.append(total_wall_time)

        # for g in all_graphs:
        #    print( nx.single_source_dijkstra(g, 1, g.number_of_nodes() - 1) )

        # Encode Process Decode GNN runtime test
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
            osp.dirname(__file__),
            "..",
            "encode_process_decode",
            train_configs["weight_base_path"],
        )
        # print(weight_base_path)
        model_path = osp.join(weight_base_path, which_model)
        # print(model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        # print("Message Passing/Encode Process Decode")
        # raw_path = dataset_path + "/raw"
        # raw_file_paths = list(map(os.path.basename, glob(raw_path + "/test.*.json")))
        for i in range(1):
            total_time_process_all_runs = 0
            total_wall_time_all_runs = 0

            for g in all_graphs:
                (preds, infers, time_per_graph, wall_time_per_graph) = predict(
                    model, process_nx(g["nxdata"])
                )
                total_wall_time_all_runs = (
                    total_wall_time_all_runs + wall_time_per_graph
                )
                total_time_process_all_runs = (
                    total_time_process_all_runs + time_per_graph
                )
                (
                    pred_graph_nx,
                    truth_total_weight,
                    pred_total_weight,
                ) = process_prediction(g["nxdata"], preds, infers)
            # print( "Graph start:", g["start"], " End:", g["end"])
            message_passing_runtime_list.append(total_time_process_all_runs)
            message_passing_walltime_list.append(total_wall_time_all_runs)

    gd = pd.DataFrame(graph_data)
    gd.to_csv(r"runtime_test_results.csv", index=False, header=True)

    print(gd)

    return


if __name__ == "__main__":
    main()
