"""
    Utilities for visualizing graphs. 
    The code was initially designed for tf_gnn. 

    TODO: Add utilities for pyg_gnn if needed. 
    TODO: Combine ground truth and prediction graphs into a single plot./
            Add a title to the plot and a decription of what the numbers mean.  

"""

from click import style
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_gnn as tfgnn
import tensorflow as tf


class VisualUtils:
    def __init__(self) -> None:
        pass

    def new_nx_draw(
        self, G, ground_truth=False, max_prob_path=False, title="Generic", save=False
    ):
        """Draws plot with prediction path. Or with max probability path.

        Args:
            G (_type_): Networkx graph as per documentation specifications. (Documentation not written ask George)
            ground_truth (bool, optional): _description_. Plot the Ground Truth or the logit class prediction.
        """

        is_start_index = 0
        is_end_index = 0

        sp_path = []
        sp_path_edges = []

        pos = {}

        for i in range(0, G.number_of_nodes()):
            # Construct node positions and find the source and target
            # of the querry.
            if G.nodes[i]["is_start"] == True:
                is_start_index = i
            if G.nodes[i]["is_end"] == False:
                is_end_index = i
            pos[i] = G.nodes[i]["pos"]

        if ground_truth:

            for i in range(0, len(G.nodes)):
                # Get shortest path nodes ground truth.
                if G.nodes[i]["is_in_path"] == True:
                    sp_path.append(i)
            title = title + "_ground_truth"
            edges_list = list(G.edges(data=True))

            for e in edges_list:
                if e[2]["is_in_path"] == True:
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
        nx.draw_networkx(G, pos=pos, labels=node_labels)
        ax = plt.gca()

        ax.set_title(title)

        nx.draw_networkx(
            G,
            pos=pos,
            nodelist=sp_path,
            edgelist=sp_path_edges,
            node_color="r",
            width=2,
            edge_color="r",
            # with_labels=True,
            labels=node_labels,
        )

        nx.draw_networkx(
            G,
            pos=pos,
            nodelist=max_prob_sp_path,
            edgelist=max_prob_sp_path_edges,
            node_color="g",
            width=2,
            edge_color="g",
            style="dotted",
            # with_labels=True,
            labels=node_labels,
        )

        # print("node_labes", node_labels, "\n\nlabels_edge", labels_edges)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edges)
        if save:

            plt.savefig(title + ".png")

        # plt.show()

    def new_prob_labels(self, G):
        edge_labels = {}
        node_labels = {}

        for i in range(0, G.number_of_nodes()):
            node_labels[i] = np.round(tf.nn.softmax(G.nodes[i]["logits"])[1].numpy(), 2)

        edges_list = list(G.edges(data=True))

        for e in edges_list:

            edge_labels[(e[0], e[1])] = np.round(
                tf.nn.softmax(e[2]["logits"])[1].numpy(), 2
            )

        sorted_values = sorted(edge_labels.values())  # Sort the values
        sorted_edges = {}

        for i in sorted_values:
            for k in edge_labels.keys():
                if edge_labels[k] == i:
                    sorted_edges[k] = edge_labels[k]

        return node_labels, sorted_edges

    def _nodes_edges_in_path(self, G):
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
        # Start from the start node.

        # Create the path based on the maximum probability.

        max_prob_walk_nodes = []
        max_prob_walk_edges = []

        current_node = start_node
        max_prob_walk_nodes.append(start_node)

        while current_node != end_node:
            edges = G.out_edges(start_node, data=True)
            max_probability_edge = 0.0
            chosen_edge = None
            for e in edges:
                probability = tf.nn.softmax(tf.argmax(e[2]["logits"]).numpy()[1])
                if (
                    probability > max_probability_edge
                    and (e[0], e[1]) not in max_prob_walk_edges
                ):
                    max_probability_edge = probability
                    chosen_edge = (e[0], e[1])

            max_prob_walk_edges.append(chosen_edge)
            max_prob_walk_nodes.append(chosen_edge[1])
            current_node = chosen_edge[1]

            ### if chosen edge is none then all edges has 0 probability
            ### Then we could look for the nodes with the highest probability
            ### When looking for neighbouring nodes they have to be unvisited nodes
            if chosen_edge == None:
                for e in edges:
                    probability_node = tf.nn.softmax(G.get_node_attributes(e[2]))
                    max_probability_node = 0
                    chosen_node = None
                    if (
                        probability_node > max_probability_node
                        and e[2] not in max_prob_walk_nodes
                    ):
                        max_prob_walk_nodes.append(e[2])
                        chosen_node = e[2]

                max_prob_walk_edges.append((current_node, chosen_node))
                max_prob_walk_nodes.append(chosen_node)
                current_node = chosen_edge[1]

            else:
                max_prob_walk_edges.append(chosen_edge)
                max_prob_walk_nodes.append(chosen_edge[1])
                current_node = chosen_edge[1]

        return max_prob_walk_nodes, max_prob_walk_edges
