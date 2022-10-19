"""Utilities for visualizing graphs. The code was initially designed for
tf_gnn.

TODO: Add utilities for pyg_gnn if needed.
TODO: Combine ground truth and prediction graphs into a single plot./
        Add a title to the plot and a description of what the numbers mean.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


class VisualUtils:
    def __init__(self) -> None:
        pass

    def new_nx_draw(
        self, G, ground_truth=False, max_prob_path=False, title="Generic", save=False
    ):
        """Draws plot with prediction path. Or with max probability path.

        Args:
            G (_type_): Networkx graph as per documentation specifications.
                        (Documentation not written ask George)
            ground_truth (bool, optional): _description_.
                    Plot the Ground Truth or the logit class prediction.
        """

        is_start_index = 0
        is_end_index = 0

        sp_path = []
        sp_path_edges = []

        pos = {}

        for i in range(0, G.number_of_nodes()):
            # Construct node positions and find the source and target
            # of the query.
            if G.nodes[i]["is_start"] is True:
                is_start_index = i
            if G.nodes[i]["is_end"] is True:

                is_end_index = i
            pos[i] = G.nodes[i]["pos"]

        if ground_truth:

            for i in range(0, len(G.nodes)):
                # Get shortest path nodes ground truth.
                if G.nodes[i]["is_in_path"] is True:
                    sp_path.append(i)
            title = title + "_ground_truth"
            edges_list = list(G.edges(data=True))

            for e in edges_list:
                if e[2]["is_in_path"] is True:
                    sp_path_edges.append([e[0], e[1]])

            labels_edges = nx.get_edge_attributes(G, "weight")
            node_labels = {}

            for i in range(0, len(G.nodes)):
                node_labels[i] = round(G.nodes[i]["weight"], 2)

            for key in labels_edges:
                labels_edges[key] = np.round(labels_edges[key], 2)

        else:

            max_prob_sp_path, max_prob_sp_path_edges = self._max_probability_walk(
                G, start_node=is_start_index, end_node=is_end_index
            )

            sp_path, sp_path_edges = self._nodes_edges_in_path(G)

            node_labels, labels_edges = self.new_prob_labels(G)

        plt.figure(figsize=(15, 15))
        nx.draw_networkx_edges(
            G, pos=pos, style="-", alpha=0.5, arrows=False, width=0.7
        )
        nx.draw_networkx_nodes(
            G,
            pos=pos,
        )

        ax = plt.gca()

        ax.set_title(title)

        nx.draw_networkx_nodes(
            G, pos=pos, nodelist=sp_path, node_color="r", node_size=350
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=sp_path_edges,
            width=2,
            edge_color="r",
        )

        if not ground_truth:
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                nodelist=max_prob_sp_path,
                node_color="g",
                node_size=250,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=max_prob_sp_path_edges,
                width=2.5,
                style="dotted",
                edge_color="g",
            )
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edges)

            nx.draw_networkx_labels(G, pos, labels=node_labels)
        if save:

            plt.savefig(title + ".png")

    def new_prob_labels(self, G):
        """Takes nx Graph in standard format.

        A
        Args:
            G (_type_): nx Graph in standard format.



        Returns:
            node_labels: probability rounded to 2 decimals
            sorted_edges: sorted edges with probability rounded to 2 decimals.

        Note: Sorted edges as edges (A, B) and (B, A) have different probability
              but in drawing  the labels overlap.
        TODO: Represent probabilities for both ways of an edge.
        """
        edge_labels = {}
        node_labels = {}

        for i in range(0, G.number_of_nodes()):
            prob = np.round(tf.nn.softmax(G.nodes[i]["logits"])[1].numpy(), 2)
            if prob > 0.0:
                node_labels[i] = prob

        edges_list = list(G.edges(data=True))

        for e in edges_list:
            prob = np.round(tf.nn.softmax(e[2]["logits"]).numpy()[1], 2)
            if prob > 0.0:
                edge_labels[(e[0], e[1])] = prob

        sorted_values = sorted(edge_labels.values())  # Sort the values
        sorted_edges = {}

        for i in sorted_values:
            for k in edge_labels.keys():
                if edge_labels[k] == i:
                    sorted_edges[k] = edge_labels[k]

        return node_labels, sorted_edges

    def _nodes_edges_in_path(self, G):
        """Returns the nodes and edges where probability if +50%

        Args:
            G (_type_): The Graph in the defined output format.

        Returns:
            sp_path: list of nodes in the shortest path.
            sp_path_edges: list of edges(as tuples) in the shortest path.
        """
        sp_path = []
        sp_path_edges = []
        for i in range(0, len(G.nodes)):
            # Get shortest path nodes ground truth.
            if tf.argmax(G.nodes[i]["logits"], axis=-1) == 1:
                sp_path.append(i)

        edges_list = list(G.edges(data=True))

        # print(edges_list)
        for e in edges_list:
            # print(list(e[2]["logits"]))
            if tf.argmax(e[2]["logits"], axis=-1) == 1:

                sp_path_edges.append([e[0], e[1]])

        return sp_path, sp_path_edges

    def _max_probability_walk(self, G, start_node, end_node):
        """Takes an output graph with a start and end node, outputs the nodes
        and edges if we take the maximum probability, either node or edge.

        Args:
            G (_type_): Oruput Graph
            start_node int: Start node.
            end_node int: End node.

        Returns:
            _type_: Max probability nodes and edges list.

        Notes: The path can be incomplete.
        """

        max_prob_walk_nodes = []
        max_prob_walk_edges = []

        current_node = start_node
        max_prob_walk_nodes.append(start_node)

        while current_node != end_node:
            edges = G.out_edges(current_node, data=True)

            max_probability_edge = 0.01
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
                max_probability_node = 0.01
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
