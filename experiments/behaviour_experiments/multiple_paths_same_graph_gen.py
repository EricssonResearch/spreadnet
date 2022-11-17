from spreadnet.datasets.graph_generator import GraphGenerator
import networkx as nx
import itertools
import os
import json
from spreadnet.datasets.data_utils.encoder import NpEncoder

# from spreadnet.utils.visualization_utils import VisualUtils
# import matplotlib.pyplot as plt
from tqdm import tqdm


def _pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def append_sp(graph, start, end):
    path = nx.shortest_path(graph, source=start, target=end, weight="length")

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, is_start=True)
    digraph.add_node(end, is_end=True)
    digraph.add_nodes_from(_set_diff(digraph.nodes(), [start]), is_start=False)
    digraph.add_nodes_from(_set_diff(digraph.nodes(), [end]), is_end=False)
    digraph.add_nodes_from(_set_diff(digraph.nodes(), path), is_in_path=False)
    digraph.add_nodes_from(path, is_in_path=True)
    path_edges = list(_pairwise(path))
    digraph.add_edges_from(_set_diff(digraph.edges(), path_edges), is_in_path=False)
    digraph.add_edges_from(path_edges, is_in_path=True)

    return digraph


def remove_ground_truth(G: nx.Graph() = None) -> nx.Graph():
    """Removes the ground truth This means that only the weights attributes
    should remain."""
    for (n1, n2, d) in G.edges(data=True):
        del d["is_in_path"]
        del d["is_start"]
        del d["is_end"]
    for n in G.nodes(data=True):
        del d["is_in_path"]
        del d["is_start"]
        del d["is_end"]

    return G


def generate_base_graphs():
    """_summary_

    Args:
        list (_type_): _description_

    Returns:
        _type_: _description_
    """
    graph_sizes = []
    in_graph_gap = 20
    max_num_nodes = 1500
    seed = 543
    theta = 20
    path_length_increaser = 3
    graph_size_start = 20
    between_sets_gaps = 200
    file_base_name = "sgmp_"
    data_dir = "same_g_mult_paths/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_base_name = "sgmp_"
    graphs = []
    datasets_names = []
    for i in range(graph_size_start, max_num_nodes, between_sets_gaps):
        graph_sizes.append([i, i + in_graph_gap])
        theta += 13

        generator = GraphGenerator(
            random_seed=seed,
            num_nodes_min_max=(i, i + in_graph_gap),
            theta=theta,
            min_length=path_length_increaser,
        )
        graphs.append(generator.generate_base_graph())
        datasets_names.append(
            file_base_name + str(i) + "_" + str(i + in_graph_gap) + ".json"
        )

    return graphs, datasets_names


def get_graphs_from_dataset(dataset_folder=""):
    """

    Args:
        list (_type_): _description_
        dataset_name (_type_, optional): _description_. Defaults to "",
        dataset_folder="")->tuple(list(nx.Graph()).
    ?
    """
    file_base_name = "sgmp_"
    datasets = os.listdir(dataset_folder)
    for path in os.listdir(dataset_folder):
        # check if current path is a file
        if os.path.isfile(os.path.join(dataset_folder, path)):
            datasets.append(path)

    graphs = []
    datasets_names = []

    percentage_datasets_used = 0.20
    for i in range(int(len(datasets) * percentage_datasets_used)):

        raw_data_path = dataset_folder + datasets[i]
        file_raw = open(raw_data_path)

        g = nx.node_link_graph(json.load(file_raw))
        g = remove_ground_truth(g)
        graphs.append(g)
        file_base_name = "sgmp_"
        datasets_names.append(file_base_name + "10_100" + "_" + str(i) + ".json")
        # TODO construct dataset name here
    return graphs, datasets_names


def generate_sgmp():
    """Generates the base graphs.

    Those base graphs will later be used for having all the paths in between.

    TODO:
        1. Graph generator that just generates the base graphs.
        2. Path generator that takes each graph and then creates a dataset
        of the ground truths. Or simply something that applies them after
        they have been computed.
        3. Number of paths increases more than exponential. How do we deal with this?
            -  2 datasets :
                -  one with all paths but smaller graphs
                -  one with larger graphs but increasing a small ratio of paths
    """

    threshold = 2500  # threshold for deciding every x path that is added
    data_dir = "same_g_mult_paths/"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    threshold = 2500  # threshold for deciding every x path that is added
    # graphs = generate_base_graphs()
    graphs, datasets_names = get_graphs_from_dataset(
        dataset_folder="10-100_dataset_raw/"
    )

    for i in len(graphs):

        g = graphs[i]

        dataset_name = datasets_names[i]

        paths_of_g = []

        pair_to_length_dict = {}
        lengths = list(nx.all_pairs_shortest_path(g))

        for x, yy in lengths:
            for y, ln in yy.items():
                if len(ln) > 1:
                    pair_to_length_dict[x, y] = len(ln) - 1

        # Because the number of paths is too large we
        # sort the dict( :) ) and then we remove every x
        # element. We sort so that we remove equally from all
        # paths of the same length

        sorted_pair_to_length_dict = dict(
            sorted(pair_to_length_dict.items(), key=lambda x: x[1])
        )
        x_used = 1
        if len(sorted_pair_to_length_dict) > threshold:
            x_used = int(len(sorted_pair_to_length_dict) / threshold)

        counter = 0

        for x in tqdm(sorted_pair_to_length_dict.keys()):
            if counter % x_used == 0:  # every x element is used
                paths_of_g.append(append_sp(g, x[0], x[1]))
            counter += 1

            # # Visual check to see if the graphs generated are
            # the right types of graphs
            # #
            # # vis = VisualUtils()
            # # vis.new_nx_draw(paths_of_g[2], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[3], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[4], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[20], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[55], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[56], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[100], ground_truth=True)
            # # vis.new_nx_draw(paths_of_g[101], ground_truth=True)
            # # plt.show()
            # # quit()
            paths_of_g_converted = []

            for p in paths_of_g:
                paths_of_g_converted.append(nx.node_link_data(p))

            with open(data_dir + f"/{dataset_name}", "w") as outfile:
                json.dump(paths_of_g_converted, outfile, cls=NpEncoder)


if __name__ == "__main__":
    generate_sgmp()
