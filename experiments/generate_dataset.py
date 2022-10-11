import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from spreadnet.utils import GraphGenerator, yaml_parser
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.datasets.data_utils.processor import process
from spreadnet.datasets.data_utils.draw import draw_networkx


# ------------------------------------------
# Params
yaml_path = os.path.join(os.path.dirname(__file__), "dataset_configs.yaml")
configs = yaml_parser(yaml_path)
data_configs = configs.data

visualize_graph = int(data_configs["visualize_graph"])
random_seed = data_configs["random_seed"]
num_nodes_min_max = (data_configs["num_node_min"], data_configs["num_node_max"])
theta = data_configs["theta"]
dataset_size = data_configs["dataset_size"]
dataset_path = os.path.join(os.path.dirname(__file__), data_configs["dataset_path"])
raw_path = dataset_path + "/raw"

if not os.path.exists(raw_path):
    os.makedirs(raw_path)

# ------------------------------------------
if __name__ == "__main__":
    graph_generator = GraphGenerator(
        random_seed=random_seed,
        num_nodes_min_max=num_nodes_min_max,
        theta=theta,
    ).task_graph_generator()

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

        # print(str(idx + 1) + "/" + str(dataset_size) + " Done")

    file_name = f"random_{num_nodes_min_max[0]}-{num_nodes_min_max[1]}.{theta}"

    if visualize_graph:
        print("Saving figure...")
        fig.tight_layout()
        plt.savefig(raw_path + f"/{file_name}.jpg", pad_inches=0, bbox_inches="tight")
        plt.clf()

    with open(raw_path + f"/{file_name}.json", "w") as outfile:
        json.dump(all_graphs, outfile, cls=NpEncoder)

    print("Graph Generation Done...\nProcessing...")
    process(dataset_path)
