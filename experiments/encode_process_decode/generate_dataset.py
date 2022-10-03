import os
import json
import networkx as nx

from spreadnet.utils import GraphGenerator, yaml_parser
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.datasets.data_utils.processor import process

# ------------------------------------------
# Params
yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
configs = yaml_parser(yaml_path)
data_configs = configs.data

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

    for idx in range(dataset_size):
        all_graphs.append(nx.node_link_data(next(graph_generator)))
        print(str(idx + 1) + "/" + str(dataset_size) + " Done")

    output_file_name = (
        f"/random_{num_nodes_min_max[0]}-{num_nodes_min_max[1]}.{theta}.json"
    )
    with open(raw_path + output_file_name, "w") as outfile:
        json.dump(all_graphs, outfile, cls=NpEncoder)

    print("Graph Generation Done...\nProcessing...")
    process(dataset_path)
