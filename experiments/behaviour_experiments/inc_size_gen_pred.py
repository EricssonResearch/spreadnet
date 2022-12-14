import sys


from multiprocessing import Pool, cpu_count

import json


from spreadnet.tf_gnn.model import gnn
import random
import networkx as nx
import numpy as np

from spreadnet.datasets.graph_generator import GraphGenerator
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.utils.experiment_utils import ExperimentUtils
import os


sys.modules["gnn"] = gnn


def inc_pred_infer(model, g):
    """Paralelize the inference calls. Remove if making it work is impossible.

    Would it work better to do it in batches? I suspect that the model can not
    fit in the cache anyway.
    """
    return nx.node_link_data(model.inferer_single_data(g))


def inc_pred(results_dir, model, graphs, ds):
    """Task for process paralelization. Currently unfeasible as the arguments
    are very large and we probably just keep overwriting the cache. Preliminary
    results show a decrease in performance. Could bring paralelization to just
    the inference part.

    Args:
        results_dir (_type_): _description_
        model (_type_): _description_
        graphs (_type_): _description_
        ds (_type_): _description_
    """
    output_name = ds.replace(
        ".json", "_out_" + model.show_model_used(ret=True, wh_w=True) + ".json"
    )
    output_graphs = list()
    for g in graphs:
        output_graphs.append(nx.node_link_data(model.inferer_single_data(g)))

    with open(results_dir + f"/{output_name}", "w") as outfile:
        json.dump(output_graphs, outfile, cls=NpEncoder)

    print(output_name, model.show_model_used(), " Done")


# @profile
def increasing_graph_size_experiment(datasets):
    """Predict for files in the increasing_size_experiment_data folder.

    Saves the results in the same folder as the
    """
    models_trained = [
        # ExperimentUtils(model_type="tf_gnn",
        # weights_model="pickled_2000_model.pickle"),
        ExperimentUtils(
            model_type="pyg_gnn",
            weights_model="no_weights_pyg_weights_best.pth",
            model_name="nobalancing-pyg",
        ),
        ExperimentUtils(
            model_type="pyg_gnn",
            weights_model="weights_pyg.pth",
            model_name="balancing-pyg",
        ),
    ]

    results_dir = "increasing_size_predictions"
    experiments_dir = "increasing_size_experiment_data"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for ds in datasets:

        raw_data_path = experiments_dir + "/" + ds
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        for model in models_trained:
            output_name = ds.replace(
                ".json", "_out_" + model.show_model_used(ret=True, wh_w=True) + ".json"
            )

            output_graphs = list()

            for g in graphs:
                output_graphs.append(nx.node_link_data(model.inferer_single_data(g)))

            with open(results_dir + f"/{output_name}", "w") as outfile:
                json.dump(output_graphs, outfile, cls=NpEncoder)

            print(os.getpid(), ": ", output_name, "Written ")


def inc_process(
    number_of_graphs,
    i,
    graph_size_start,
    graph_size_increment,
    graph_size_gap,
    seed,
    nodes_min_max,
    theta,
    path_length_increaser,
):
    print(os.getpid(), "Started ")
    generator = GraphGenerator(
        random_seed=seed,
        num_nodes_min_max=nodes_min_max,
        theta=theta,
        min_length=path_length_increaser,
    )
    graph_generator = generator.task_graph_generator()

    all_graphs = list()
    for idx in range(number_of_graphs):

        g = next(graph_generator)
        all_graphs.append(nx.node_link_data(g))
        generator.set_theta(theta)

    graph_name = (
        "increasing_size"
        + "_"
        + str(i)
        + "_"
        + str(i + graph_size_gap)
        + "_"
        + str(theta)
        + ".json"
    )
    graphs_folder = "increasing_size_experiment_data/"
    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)
    with open(graphs_folder + f"/{graph_name}", "w") as outfile:
        json.dump(all_graphs, outfile, cls=NpEncoder)
    print(
        "Process ID: ",
        os.getpid(),
        " Dataset: ",
        int((i - graph_size_start) / graph_size_increment),
    )


def increasing_graph_size_generator():
    """Gradually increasing the graph size while keeping theta at the same
    ratio."""

    """TODO: Generator should get the parameters through a function.
            The Graph Generator is not designed for taking different configs.

            We should make it easier to generate graphs with different
            characteristics
    """
    seed = random.seed()

    graph_size_increment = 15
    graph_size_gap = 20  # gap between the min and the max
    graph_size_start = 10
    max_min_graph_size = 2500
    theta = 20
    path_length_increaser = 3
    number_of_graphs = 50
    param_list = []

    for i in range(graph_size_start, max_min_graph_size, graph_size_increment):
        nodes_min_max = (i, i + graph_size_gap)
        if i % 250 == 0:
            path_length_increaser += 1

        theta += 10

        param_list.append(
            [
                number_of_graphs,
                i,
                graph_size_start,
                graph_size_increment,
                graph_size_gap,
                seed,
                nodes_min_max,
                theta,
                path_length_increaser,
            ]
        )
    pool = Pool(
        processes=cpu_count() - 1
    )  # Chose according to the number of processors
    pool.starmap(
        inc_process,
        param_list,
    )
    pool.close()


def divide_datasets(no_process=cpu_count() - 1):
    experiments_dir = "increasing_size_experiment_data"
    datasets = list()
    for path in os.listdir(experiments_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(experiments_dir, path)):
            datasets.append(path)
    return np.array_split(datasets, no_process)


if __name__ == "__main__":

    # increasing_graph_size_generator()

    ds_split = divide_datasets()
    ds_split_l = []
    for d in ds_split:
        ds_split_l.append([d.tolist()])

    pool = Pool(processes=cpu_count() - 2)
    pool.starmap(increasing_graph_size_experiment, ds_split_l)
    pool.close()
