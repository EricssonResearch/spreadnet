import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import wandb
from datetime import datetime

from spreadnet.utils import GraphGenerator, yaml_parser
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.datasets.data_utils.processor import process
from spreadnet.datasets.data_utils.draw import draw_networkx
from spreadnet.datasets import run_statistics


# ------------------------------------------
# Params
yaml_path = os.path.join(os.path.dirname(__file__), "dataset_configs.yaml")
data_configs = yaml_parser(yaml_path).data

visualize_graph = int(data_configs["visualize_graph"])
random_seed = data_configs["random_seed"]
min_path_length = int(data_configs["min_path_length"])
num_nodes_min_max = (data_configs["num_node_min"], data_configs["num_node_max"])
starting_theta = data_configs["starting_theta"]
dataset_size = data_configs["dataset_size"]
dataset_path = os.path.join(os.path.dirname(__file__), data_configs["dataset_path"])
raw_path = dataset_path + "/raw"

if not os.path.exists(raw_path):
    os.makedirs(raw_path)

# ------------------------------------------
if __name__ == "__main__":
    date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    now = datetime.now().strftime("%H:%M:%S_%d-%m-%y")
    wandb.init(
        project="gnn_pytorch_test_exp",
        name=f"generate_dataset_{now}",
        config=data_configs,  # type: ignore
    )
    theta = starting_theta
    increase_theta_after = (
        data_configs["num_node_max"] - data_configs["num_node_min"]
    ) * 15
    cap_theta = 60

    generator = GraphGenerator(
        random_seed=random_seed,
        num_nodes_min_max=num_nodes_min_max,
        theta=theta,
        min_length=min_path_length,
    )

    graph_generator = generator.task_graph_generator()

    all_graphs = list()

    if visualize_graph:
        graphs_to_be_drawn = visualize_graph
        fig = plt.figure(
            figsize=(
                graphs_to_be_drawn * 12 if graphs_to_be_drawn < 5 else 60,
                math.ceil(graphs_to_be_drawn / 5) * 12,
            )
        )

    for idx in tqdm(range(dataset_size)):
        g = next(graph_generator)
        all_graphs.append(nx.node_link_data(g))

        if visualize_graph and graphs_to_be_drawn > 0:
            draw_networkx(fig, g, idx + 1, visualize_graph)
            graphs_to_be_drawn -= 1

        if theta < cap_theta and (idx + 1) % increase_theta_after == 0:
            theta += 1
            generator.set_theta(theta)

    file_name = (
        f"random_.{random_seed}."
        + f"{num_nodes_min_max[0]}-{num_nodes_min_max[1]}.{starting_theta}"
    )

    if visualize_graph:
        print("Saving figure...")
        fig.tight_layout()
        plt.savefig(raw_path + f"/{file_name}.jpg", pad_inches=0, bbox_inches="tight")
        plt.clf()

    with open(raw_path + f"/{file_name}.json", "w") as outfile:
        json.dump(all_graphs, outfile, cls=NpEncoder)

    print("Graph Generation Done...\nProcessing...")
    process(dataset_path)

    run_stat = run_statistics.RunStatistics()
    run_stat.add_data("raw_path", raw_path)

    # Mark the run as finished
    wandb.finish()
