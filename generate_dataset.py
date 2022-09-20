import pickle

from utils import GraphGenerator, SPGraphDataset, get_project_root

# ------------------------------------------
# Params
random_seed = 0
num_nodes_min_max = (8, 17)
dataset_size = 1000

raw_path = str(get_project_root()) + "/dataset/raw/"
# ------------------------------------------
if __name__ == '__main__':
    graph_generator = GraphGenerator(random_seed=random_seed,
                                     num_nodes_min_max=num_nodes_min_max,
                                     theta=20).task_graph_generator()

    for idx in range(dataset_size):
        graph_nx = next(graph_generator)
        pickle.dump(graph_nx, open(raw_path + f'raw_{idx}.pickle'.format(idx=idx), 'wb'))

    print("Graph Generation Done...\n")

    ds_generator = SPGraphDataset(root=str(get_project_root()) + "/dataset/")
    ds_generator.process()

