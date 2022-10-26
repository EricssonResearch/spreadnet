import sys

# import json

from multiprocessing import Pool, cpu_count

# from spreadnet.utils.experiment_utils import ExperimentUtils
import json
import os
from spreadnet.tf_gnn.model import gnn
import random
import networkx as nx

from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.datasets.graph_generator import GraphGenerator
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.utils.experiment_utils import ExperimentUtils

sys.modules["EncodeProcessDecode"] = EncodeProcessDecode
sys.modules["gnn"] = gnn


def inc_pred(results_dir, experiments_dir, model, datasets):
    for ds in datasets:

        raw_data_path = experiments_dir + "/" + ds
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        output_graphs = list()
        output_name = ds.replace(".json", "_out.json")

    for g in graphs:
        print("Inference")
        output_graphs.append(model.inferer_single_data(g))

    with open(results_dir + f"/{output_name}", "w") as outfile:
        json.dump(output_graphs, outfile, cls=NpEncoder)

    print(output_name, model.show_model_used(), " Done")


def increasing_graph_size_experiment():
    """Predict for files in the increasing_size_experiment_data folder.

    Saves the results in the same folder as the
    """
    models_trained = [
        # ExperimentUtils(model_type="tf_gnn",
        # weights_model="pickled_2000_model.pickle"),
        ExperimentUtils(model_type="pyg_gnn", weights_model="model_weights_best.pth"),
    ]
    experiments_dir = "increasing_size_experiment_data"
    results_dir = "increasing_size_predictions"
    datasets = list()
    for path in os.listdir(experiments_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(experiments_dir, path)):
            datasets.append(path)
    # Chose according to the number of processors

    param_list = []
    for m in models_trained:
        param_list.append([results_dir, experiments_dir, m, datasets])

    pool = Pool(processes=cpu_count() - 1)

    pool.starmap(inc_pred, param_list)
    pool.close()

    # This one can be paralelied
    #     for g in tqdm(graphs, desc="Model: " + str(idx_m)):
    #         output_graphs.append(models_trained[idx_m].inferer_single_data(g))

    # with open(results_dir + f"/{output_name}", "w") as outfile:
    #     json.dump(output_graphs, outfile, cls=NpEncoder)


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
    with open(graphs_folder + f"/{graph_name}", "w") as outfile:
        json.dump(all_graphs, outfile, cls=NpEncoder)
    print("Graph: ", int((i - graph_size_start) / graph_size_increment))


def increasing_graph_size_generator():
    """Gradually increasing the graph size while keeping theta at the same
    ratio."""

    """TODO: Generator should get the parameters through a function.
            The Graph Generator is not designed for taking different configs.

            We should make it easier to generate graphs with different
            characteristics
    """
    seed = random.seed()

    graph_size_increment = 50
    graph_size_gap = 25  # gap between the min and the max
    graph_size_start = 10
    max_min_graph_size = 200
    theta = 20
    path_length_increaser = 3
    number_of_graphs = 10
    param_list = []

    for i in range(graph_size_start, max_min_graph_size, graph_size_increment):
        nodes_min_max = (i, i + graph_size_gap)
        if i % 500 == 0:
            path_length_increaser += 1
            theta += 1

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
    pool.starmap(inc_process, param_list, chunksize=100)
    pool.close()


if __name__ == "__main__":
    # increasing_graph_size_generator()
    increasing_graph_size_experiment()
