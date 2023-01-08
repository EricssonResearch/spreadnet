import os
import json
from codecarbon import OfflineEmissionsTracker
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import numpy as np
import wandb
import argparse

from spreadnet.utils import GraphGenerator, yaml_parser
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.datasets.data_utils.processor import process_raw_data_folder
from spreadnet.datasets.data_utils.draw import draw_networkx
from spreadnet.utils import log_utils
from spreadnet.datasets import dataset_statistics


# ------------------------------------------
# Params
yaml_path = os.path.join(os.path.dirname(__file__), "dataset_configs.yaml")
data_configs = yaml_parser(yaml_path).data
parser = argparse.ArgumentParser(description="Generate dataset.")

visualize_graph = int(data_configs["visualize_graph"])
random_seed = data_configs["random_seed_train"]
random_seed_test = data_configs["random_seed_test"]
min_path_length = int(data_configs["min_path_length"])
num_nodes_min_max = (
    data_configs["num_node_min_train"],
    data_configs["num_node_max_train"],
)
num_nodes_min_max_test = (
    data_configs["num_node_min_test"],
    data_configs["num_node_max_test"],
)
theta_cap = int(data_configs["theta_cap"])
theta_increase_rate = float(data_configs["theta_increase_rate"])
dataset_size = data_configs["dataset_size_train"]
dataset_path = os.path.join(os.path.dirname(__file__), data_configs["dataset_path"])
raw_path = dataset_path + "/raw"
log_save_path = dataset_path + "/logs"

parser.add_argument(
    "--wandb", help="Specify if generate dataset with wandb", action="store_true"
)

parser.add_argument(
    "--stats", help="Specify if generate dataset with statistics", action="store_true"
)
args = parser.parse_args()
use_wandb = args.wandb
generate_statistics = args.stats

if not os.path.exists(raw_path):
    os.makedirs(raw_path)


def generate_task_graph(
    gen_fn,
    starting_theta,
    file_name,
    seed,
    idx,
):
    """Generate graph.

    Args:
        gen_fn: generation function
        starting_theta: theta to be passed to graph generator
        file_name: output file name
        seed: graph seed to override
        idx: index
    Returns:
        None
    """
    json_file_name = raw_path + f"/{file_name}_{idx:06}.json"
    exists = os.path.exists(json_file_name)

    if exists:
        return

    increase_theta_after = 1 / theta_increase_rate
    theta = starting_theta + int(idx / increase_theta_after)

    if theta > theta_cap:
        theta = theta_cap

    g = gen_fn(seed, theta)
    graphs = [nx.node_link_data(g)]

    with open(json_file_name, "w") as outfile:
        json.dump(graphs, outfile, cls=NpEncoder)

    if visualize_graph and idx < visualize_graph:
        plot_size = 20
        fig = plt.figure(
            figsize=(
                plot_size,
                plot_size,
            )
        )

        draw_networkx(str(idx + 1), fig, g, 1, 1)
        fig.tight_layout()
        plt.savefig(
            raw_path + f"/{file_name}_{idx:06}.jpg", pad_inches=0, bbox_inches="tight"
        )
        plt.clf()


def generate(name, seed, size, nodes_min_max, starting_theta):
    """Generate dataset with config.

    Args:
        name: prefix of raw file name
        seed: random seed
        size: dataset size
        nodes_min_max: (min, max)
        starting_theta: theta to be passed to graph generator
        increase_theta_rate: gradual increase in theta after a certain iteration
    Returns:
        None
    """

    file_name = (
        f"{name}.{seed}." + f"{nodes_min_max[0]}-{nodes_min_max[1]}.{starting_theta}"
    )

    generator = GraphGenerator(
        random_seed=seed,
        num_nodes_min_max=nodes_min_max,
        theta=starting_theta,
        min_length=min_path_length,
    )

    loop = tqdm(
        range(size),
        unit="graph",
        total=size,
        leave=True,
    )

    random_state = np.random.RandomState(seed)
    seeds = random_state.randint(np.iinfo(np.int32).max, size=size)

    Parallel(n_jobs=-1, backend="multiprocessing", batch_size=4)(
        delayed(generate_task_graph)(
            generator.generate_task_graph,
            starting_theta,
            file_name,
            seeds[i],
            i,
        )
        for i in loop
    )


if __name__ == "__main__":

    dataset_logger = log_utils.init_file_console_logger(
        "dataset_logger", log_save_path, "generate_dataset"
    )
    # Train and validation set
    dataset_logger.info(f"use_wandb:{use_wandb}")
    dataset_logger.info("Generating training and validation set...")
    co2_emissions = OfflineEmissionsTracker(country_iso_code="SWE")
    co2_emissions.start()

    start_time = time.time()
    try:
        generate(
            "random",
            random_seed,
            dataset_size,
            num_nodes_min_max,
            data_configs["starting_theta_train"],
        )
        process_raw_data_folder(dataset_path, "all", "[!test.]")

        # Test set
        dataset_logger.info("Generating test set...")
        generate(
            "test.random",
            random_seed_test,
            data_configs["dataset_size_test"],
            num_nodes_min_max_test,
            data_configs["starting_theta_test"],
        )

        dataset_logger.info("Graph Generation Done...\nProcessing...")
        process_raw_data_folder(dataset_path, "test.all", "test.")

    except Exception as exception:
        dataset_logger.exception(exception)

    if generate_statistics:
        dataset_logger.info("Computing stats...")
        dataset_stat = dataset_statistics.DatasetStatistics(dataset_logger)
        dataset_stat.add_data("raw_path", raw_path)
        if use_wandb:
            wandb.login()
            json_file_paths = dataset_stat.get_dataset_path()
            dataset_stat.upload_dataset_statistics(json_file_paths)
            # artifact_location = dataset_stat.download_dataset_wandb(
            #                                 entity="pbs",
            #                                 project="artifacts-testing",
            #                                 job_type="data-download")

    co2_emissions_final = co2_emissions.stop()
    dataset_logger.info(f'Time elapsed = {(time.time()-start_time)} sec \n {"=":176s}')
    dataset_logger.info(
        f"Co2 Emissions: {co2_emissions_final} kg co2.eq/KWh"
        f"For more data see emissions.csv"
    )
