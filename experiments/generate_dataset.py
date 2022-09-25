import os
import pickle

from spreadnet.utils import GraphGenerator, SPGraphDataset, yaml_parser

# ------------------------------------------
# Params
yaml_path = "./configs.yaml"
configs = yaml_parser(yaml_path)
data_configs = configs.data

random_seed = data_configs["random_seed"]
num_nodes_min_max = (data_configs["num_node_min"], data_configs["num_node_max"])
dataset_size = data_configs["dataset_size"]
dataset_path = data_configs["dataset_path"]

raw_path = dataset_path + "raw/"
if not os.path.exists(raw_path):
    os.makedirs(raw_path)
# raw_path = str(get_project_root()) + dataset_path + "raw/"
# ------------------------------------------
if __name__ == "__main__":
    graph_generator = GraphGenerator(
        random_seed=random_seed, num_nodes_min_max=num_nodes_min_max, theta=20
    ).task_graph_generator()

    for idx in range(dataset_size):
        graph_nx = next(graph_generator)
        pickle.dump(
            graph_nx, open(raw_path + f"raw_{idx}.pickle".format(idx=idx), "wb")
        )

    print("Graph Generation Done...\n")

    ds_generator = SPGraphDataset(root=os.path.join(raw_path, ".."))
    ds_generator.process()
