import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import time
import psutil

from spreadnet.utils import GraphGenerator, yaml_parser
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.datasets.data_utils.processor import process_raw_data_folder
from spreadnet.datasets.data_utils.draw import draw_networkx
from spreadnet.datasets import run_statistics
from spreadnet.utils import log_utils


process = psutil.Process(os.getpid())
# ------------------------------------------
# Params
yaml_path = os.path.join(os.path.dirname(__file__), "dataset_configs.yaml")
data_configs = yaml_parser(yaml_path).data

visualize_graph = int(data_configs["visualize_graph"])
random_seed = data_configs["random_seed"]
random_seed_test = data_configs["random_seed_test"]
min_path_length = int(data_configs["min_path_length"])
num_nodes_min_max = (data_configs["num_node_min"], data_configs["num_node_max"])
num_nodes_min_max_test = (
    data_configs["num_node_min_test"],
    data_configs["num_node_max_test"],
)
theta_cap = float(data_configs["theta_cap"])
dataset_size = data_configs["dataset_size"]
dataset_path = os.path.join(os.path.dirname(__file__), data_configs["dataset_path"])
raw_path = dataset_path + "/raw"
log_save_path = dataset_path + "/logs"
if not os.path.exists(raw_path):
    os.makedirs(raw_path)


def generate(name, seed, size, nodes_min_max, starting_theta, increase_theta_rate=15):
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

    theta = starting_theta
    increase_theta_after = (nodes_min_max[1] - nodes_min_max[0]) * increase_theta_rate

    generator = GraphGenerator(
        random_seed=seed,
        num_nodes_min_max=nodes_min_max,
        theta=theta,
        min_length=min_path_length,
    )

    graph_generator = generator.task_graph_generator()

    all_graphs = list()
    chunk_if_mem_usage_exceed = 2e9  # 2GB
    chunk_counter = 1

    if visualize_graph:
        graphs_to_be_drawn = visualize_graph
        plot_size = 20
        fig = plt.figure(
            figsize=(
                graphs_to_be_drawn * plot_size if graphs_to_be_drawn < 5 else 60,
                math.ceil(graphs_to_be_drawn / 5) * plot_size,
            )
        )

    for idx in tqdm(range(size)):
        g = next(graph_generator)
        all_graphs.append(nx.node_link_data(g))

        if visualize_graph and graphs_to_be_drawn > 0:
            draw_networkx(str(idx + 1), fig, g, idx + 1, visualize_graph)
            graphs_to_be_drawn -= 1

        if theta < theta_cap and (idx + 1) % increase_theta_after == 0:
            theta += 1
            generator.set_theta(theta)

        mem_usage = process.memory_info().rss
        dataset_logger.info(f"Mem usage: {round(mem_usage/1e9, 2)} GB")

        if mem_usage > chunk_if_mem_usage_exceed:
            with open(raw_path + f"/{file_name}_{chunk_counter}.json", "w") as outfile:
                json.dump(all_graphs, outfile, cls=NpEncoder)
                all_graphs.clear()
                chunk_counter += 1

    if visualize_graph:
        dataset_logger.info("Saving figure...")
        fig.tight_layout()
        plt.savefig(raw_path + f"/{file_name}.jpg", pad_inches=0, bbox_inches="tight")
        plt.clf()

    if len(all_graphs):
        with open(raw_path + f"/{file_name}_{chunk_counter}.json", "w") as outfile:
            json.dump(all_graphs, outfile, cls=NpEncoder)


if __name__ == "__main__":

    dataset_logger = log_utils.init_file_console_logger(
        "dataset_logger", log_save_path, "generate_dataset"
    )
    # Train and validation set
    dataset_logger.info("Generating training and validation set...")

    start_time = time.time()
    try:
        generate(
            "random",
            random_seed,
            dataset_size,
            num_nodes_min_max,
            data_configs["starting_theta"],
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

        run_stat = run_statistics.RunStatistics()
        run_stat.add_data("raw_path", raw_path)

    except Exception as exception:
        dataset_logger.exception(exception)

    dataset_logger.info(f'Time elapsed = {(time.time()-start_time)} sec \n {"=":176s}')
