"""
    Utilities for visualizing graphs. 
    The code was initially designed for tf_gnn. 

    TODO: Add utilities for pyg_gnn if needed. 
    TODO: Combine ground truth and prediction graphs into a single plot./
            Add a title to the plot and a decription of what the numbers mean.  

"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_gnn as tfgnn
import tensorflow as tf


class VisualUtils:
    def __init__(self) -> None:
        pass

    def new_nx_draw(self, G, ground_truth=False, max_prob_path=False, title="Generic"):
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
            pass
            # for i in range(0, len(G.nodes)):
            #     # Get shortest path nodes ground truth.
            #     if G.nodes[i]["is_in_path"] == True:
            #         sp_path.append(i)

            # edges_list = list(G.edges(data=True))

            # for e in edges_list:
            #     if e[2]["is_in_path"] == True:
            #         sp_path_edges.append([e[0], e[1]])

            # labels_edges = nx.get_edge_attributes(G, "weight")
            # node_labels = {}

            # for i in range(0, len(G.nodes)):
            #     node_labels[i] = round(G.nodes[i]["weight"], 2)

            # for key in labels_edges:
            #     labels_edges[key] = np.round(labels_edges[key], 2)

        else:
            if max_prob_path:
                pass  # Construct the path based on the maximum probability.
            else:
                sp_path, sp_path_edges = self._nodes_edges_in_path(G)

                node_labels, labels_edges = self.new_prob_labels(G)
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

        # print("node_labes", node_labels, "\n\nlabels_edge", labels_edges)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edges)

        plt.show()

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
