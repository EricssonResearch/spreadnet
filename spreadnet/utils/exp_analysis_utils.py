import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd
import networkx as nx
from tqdm import tqdm


class AccuracyMetrics:
    def get_start_end_nodes(self, G):
        for i in range(0, G.number_of_nodes()):
            # Construct node positions and find the source and target
            # of the query.
            if G.nodes[i]["is_start"] is True:
                start_node = i
            if G.nodes[i]["is_end"] is True:
                end_node = i

        return start_node, end_node

    def max_probability_walk(self, G, prob_treshold):
        """Takes an output graph with a start and end node, outputs the nodes
        and edges if we take the maximum probability, either node or edge.

        Args:
            G (_type_): Oruput Graph
            start_node int: Start node.
            end_node int: End node.
            prob_treshold: float (0,1]
        Returns:
            _type_: Max probability nodes and edges list.

        Notes: The path can be incomplete.
        """

        max_prob_walk_nodes = []
        max_prob_walk_edges = []

        start_node, end_node = self.get_start_end_nodes(G)

        current_node = start_node
        max_prob_walk_nodes.append(start_node)

        while current_node != end_node:
            edges = G.out_edges(current_node, data=True)

            max_probability_edge = prob_treshold
            chosen_edge = None
            for e in edges:

                probability_edge = np.round(tf.nn.softmax(e[2]["logits"]).numpy()[1], 2)
                if (
                    probability_edge > max_probability_edge
                    and e[1] not in max_prob_walk_nodes
                    and (e[0], e[1]) not in max_prob_walk_edges
                    and (e[1], e[0]) not in max_prob_walk_edges
                ):
                    max_probability_edge = probability_edge

                    chosen_edge = (e[0], e[1])

            if chosen_edge is None:
                chosen_node = None
                max_probability_node = prob_treshold
                for e in edges:

                    probability_node = np.round(
                        tf.nn.softmax(dict(G.nodes(data=True))[e[1]]["logits"]).numpy()[
                            1
                        ],
                        2,
                    )

                    if (
                        probability_node > max_probability_node
                        and e[1] not in max_prob_walk_nodes
                    ):
                        max_probability_node = probability_node
                        chosen_node = e[1]

                if chosen_node is not None:
                    max_prob_walk_edges.append((current_node, chosen_node))
                    max_prob_walk_nodes.append(chosen_node)
                    current_node = chosen_node
                else:
                    return max_prob_walk_nodes, max_prob_walk_edges
            else:
                max_prob_walk_edges.append(chosen_edge)
                max_prob_walk_nodes.append(chosen_edge[1])
                current_node = chosen_edge[1]

        return max_prob_walk_nodes, max_prob_walk_edges

    def prob_accuracy(
        self,
        file_name,
        only_path=False,
        use_nodes=True,
        use_edges=True,
        use_start_end=True,
    ):
        """Saves the accuracy to a pandas dataframe that is then saved as a
        csv.

        Args:
            file_name (_type_): name of the csv
            only_path (bool, optional): Only nodes/edges part of the path accuracy.
                Defaults to False.
            nodes (bool, optional): Nodes accuracy. Defaults to True.
            edges (bool, optional): Edges Accuracy. Defaults to True.

        Not using either nodes or edges for the accuracy returns an empty array.
        """
        pred_dir = "increasing_size_predictions"
        datasets = list()

        # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
        prob_treshold = [0.5, 0.25, 0.01]
        for path in os.listdir(pred_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(pred_dir, path)):
                datasets.append(path)
        accuracy_node_and_edge = []
        for ds in tqdm(datasets):
            raw_data_path = pred_dir + "/" + ds
            file_raw = open(raw_data_path)

            graphs = json.load(file_raw)

            for pt in prob_treshold:

                pred = []
                no_ground_truth_nodes = 0
                ground_truth_edges = 0
                for g in graphs:
                    g = nx.node_link_graph(g)  # TODO add probability for the edges also

                    edges_list = list(g.edges(data=True))

                    if only_path:
                        if use_nodes:

                            for i in range(0, g.number_of_nodes()):
                                if g.nodes[i]["is_in_path"]:
                                    no_ground_truth_nodes += 1
                            if not use_start_end:
                                no_ground_truth_nodes -= 2
                        if use_edges:
                            for e in edges_list:
                                if e[2]["is_in_path"]:
                                    ground_truth_edges += 1
                    if use_nodes:
                        if use_start_end:
                            for i in range(0, g.number_of_nodes()):
                                prob = np.round(
                                    tf.nn.softmax(g.nodes[i]["logits"])[1].numpy(), 2
                                )
                                if prob >= pt and g.nodes[i]["is_in_path"]:
                                    # print(prob, pt, g.nodes[i]["is_in_path"])

                                    pred.append(1)
                                elif (
                                    prob < pt
                                    and not g.nodes[i]["is_in_path"]
                                    and not only_path
                                ):
                                    pred.append(1)
                                else:
                                    pred.append(0)
                        else:
                            for i in range(0, g.number_of_nodes()):
                                prob = np.round(
                                    tf.nn.softmax(g.nodes[i]["logits"])[1].numpy(), 2
                                )
                                if (
                                    prob >= pt
                                    and g.nodes[i]["is_in_path"]
                                    and not g.nodes[i]["is_start"]
                                    and not g.nodes[i]["is_end"]
                                ):
                                    # print(prob, pt, g.nodes[i]["is_in_path"])

                                    pred.append(1)
                                elif (
                                    prob < pt
                                    and not g.nodes[i]["is_in_path"]
                                    and not only_path
                                ):
                                    pred.append(1)
                                else:
                                    pred.append(0)

                    if use_edges:
                        for e in edges_list:
                            prob = np.round(tf.nn.softmax(e[2]["logits"])[1].numpy(), 2)
                            if prob >= pt and e[2]["is_in_path"]:
                                pred.append(1)
                            elif prob < pt and not e[2]["is_in_path"] and not only_path:
                                pred.append(1)
                            else:
                                pred.append(0)

                if only_path:
                    accuracy_node_and_edge.append(
                        sum(pred) / (no_ground_truth_nodes + ground_truth_edges)
                    )
                else:
                    accuracy_node_and_edge.append(sum(pred) / len(pred))
        self.to_df_format(
            datasets,
            prob_threshold=prob_treshold,
            accuracy=accuracy_node_and_edge,
            name=file_name,
        )

    def max_prob_path_lengths(self, file_name=""):
        """Comapres the length of the path found to the ground truth length."""

        pred_dir = "increasing_size_predictions"
        datasets = list()

        # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
        prob_threshold = [0.5, 0.25, 0.01]
        for path in os.listdir(pred_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(pred_dir, path)):
                datasets.append(path)

        accuracy_path_length = []
        for ds in tqdm(datasets):
            raw_data_path = pred_dir + "/" + ds
            file_raw = open(raw_data_path)

            graphs = json.load(file_raw)

            for pt in prob_threshold:
                paths_correct = []
                for g in graphs:
                    g = nx.node_link_graph(g)  # TODO add probability for the edges also
                    (
                        max_prob_walk_nodes,
                        max_prob_walk_edges,
                    ) = self.max_probability_walk(g, pt)

                    start_node, end_node = self.get_start_end_nodes(g)

                    nodes_in_path = 0
                    for i in range(0, g.number_of_nodes()):
                        if g.nodes[i]["is_in_path"]:
                            nodes_in_path += 1

                    len_walk = len(max_prob_walk_nodes)

                    if (
                        len_walk == nodes_in_path
                        and max_prob_walk_nodes[len(max_prob_walk_nodes) - 1]
                        == end_node
                    ):
                        points = 1
                    elif len_walk < nodes_in_path:
                        points = 0

                    elif max_prob_walk_nodes[len(max_prob_walk_nodes) - 1] == end_node:
                        points = nodes_in_path / len_walk

                    else:
                        points = 0

                    paths_correct.append(points)
                accuracy_path_length.append(sum(paths_correct) / len(graphs))

        self.to_df_format(
            datasets=datasets,
            prob_threshold=prob_threshold,
            accuracy=accuracy_path_length,
            name=file_name,
        )

    def path_length_as_accuracy(self, file_name):
        """Compares the path length found by djikstra to the path length found
        by the GNN if it finds it. The metric is applied on the whole graph so
        the distance Can be computed as a percentage.

        This function should also return the percentage of incomplete paths.
        Also the Percentage over how much bigger is the path
        """
        pred_dir = "increasing_size_predictions"
        datasets = list()

        # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]

        for path in os.listdir(pred_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(pred_dir, path)):
                datasets.append(path)
        prob_threshold = [0.5, 0.25, 0.01]
        percentage_paths = []  # for each set of graphs in the dataset
        path_length_accuracy = []

        for ds in datasets:
            raw_data_path = pred_dir + "/" + ds
            file_raw = open(raw_data_path)

            graphs = json.load(file_raw)
            graphs_in_ds = len(graphs)

            for pt in prob_threshold:
                paths_found = 0
                graph_ratios = []
                for g in graphs:
                    g_edge_weights_gt = 0
                    g_edge_weights_pred = 0
                    g = nx.node_link_graph(g)

                    (
                        max_prob_walk_nodes,
                        max_prob_walk_edges,
                    ) = self.max_probability_walk(g, pt)
                    start_node, end_node = self.get_start_end_nodes(g)
                    if end_node == max_prob_walk_nodes[len(max_prob_walk_nodes) - 1]:
                        paths_found += 1

                        edges_list = list(g.edges(data=True))

                        for e in edges_list:
                            if e[2]["is_in_path"]:
                                g_edge_weights_gt += e[2]["weight"]

                        for e_mw in max_prob_walk_edges:
                            for e in edges_list:
                                if e[0] == e_mw[0] and e[1] == e_mw[1]:
                                    g_edge_weights_pred += e[2]["weight"]

                        graph_ratios.append(g_edge_weights_gt / g_edge_weights_pred)
                percentage_paths.append(paths_found / graphs_in_ds)

                if paths_found == 0:
                    path_length_accuracy.append(0)
                else:
                    path_length_accuracy.append(np.mean(graph_ratios))

        self.to_df_format(
            datasets,
            prob_threshold,
            percentage_paths,
            file_name.replace(".csv", "_percentage_paths.csv"),
        )

        self.to_df_format(
            datasets,
            prob_threshold,
            path_length_accuracy,
            file_name.replace(".csv", "_path_length_accuracy.csv"),
        )

    def to_df_format(self, datasets, prob_threshold, accuracy, name):
        df = pd.DataFrame()

        prob_treshold_df = []
        model_type = []
        g_size = []

        for ds in datasets:
            for pt in prob_threshold:

                prob_treshold_df.append(pt)
                # The following line is an abomination
                ds_clean = (
                    ds.replace("increasing_size_", "")
                    .replace(".json", "")
                    .replace("_out", "")
                )
                split = ds_clean.split("_")
                model_type.append(split[(len(split) - 2)])
                g_size.append(split[1])
                # g_size.append((split[0], split[1]))

        df["Graph Size"] = g_size
        df["Model Type"] = model_type
        df["Probabiltiy Threshold"] = prob_treshold_df

        df["Accuracy"] = accuracy

        df.to_csv(name, index=False)
