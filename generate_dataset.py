import pickle

from utils import GraphGenerator, SPGraphDataset, get_project_root

# ------------------------------------------
# Params
random_seed = 5
num_nodes_min_max = (10, 20)
dataset_size = 100

raw_path = str(get_project_root()) + "/dataset/raw/"
# ------------------------------------------
if __name__ == '__main__':
    graph_generator = GraphGenerator(random_seed=random_seed,
                                     num_nodes_min_max=num_nodes_min_max).task_graph_generator()

    for idx in range(dataset_size):
        graph_nx = next(graph_generator)
        pickle.dump(graph_nx, open(raw_path + f'raw_{idx}.pickle'.format(idx=idx), 'wb'))

    print("Graph Generation Done...\n")

    ds_generator = SPGraphDataset(root=str(get_project_root()) + "/dataset/")
    ds_generator.process()

