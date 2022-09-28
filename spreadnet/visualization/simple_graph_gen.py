import collections
from errno import EDEADLK
import functools
import itertools
from operator import truediv
from tracemalloc import start
from typing import Callable, Optional, Mapping, Tuple
from networkx.drawing.nx_pydot import write_dot

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import signal
from scipy import spatial
import tensorflow_gnn as tfgnn
import tensorflow as tf
import 


import pickle

class VisualG:
    def __init__(self):
        pass

    def pred_tensor_graph_to_nx_graph(self, graph_tensor):
        """Predicted tensor graph to nx graph.

        Args:
            graph_tensor (_type_): _description_

        Returns:
            _type_: _description_
        """

        tfgnn.check_scalar_graph_tensor(graph_tensor)
        assert graph_tensor.num_components == 1

        # Empty graph to be populated
        G = nx.Graph()

        node_set = graph_tensor.node_sets["cities"]
        node_positions = node_set["pos"].numpy()

        start_node_mask = node_set["is_start"].numpy()
        end_node_mask = node_set["is_end"].numpy()
        other_nodes_mask = ~(start_node_mask + end_node_mask)
        node_weights = node_set["weight"].numpy()
        in_path_node_mask = node_set["is_in_path"].numpy()

        # Add nodes with specifc atributes
        for i in range(len(node_positions)):
            path_flag = False
            start_flag = False
            end_flag = False
            if start_node_mask[i] == True:
                start_flag = True
            if in_path_node_mask[i] == True:
                path_flag = True
            if end_node_mask[i] == True:
                end_flag = True

            G.add_node(
                i,
                pos=node_positions[i],
                is_start=start_flag,
                is_end=end_flag,
                is_in_path=path_flag,
                weight=node_weights[i],
            )

        # Add edges from tensor

        in_path_edges_mask = graph_tensor.edge_sets["roads"]["is_in_path"].numpy()
        edge_weights = graph_tensor.edge_sets["roads"]["weight"].numpy()
        edge_links = np.stack(
            [
                graph_tensor.edge_sets["roads"].adjacency.source.numpy(),
                graph_tensor.edge_sets["roads"].adjacency.target.numpy(),
            ],
            axis=0,
        )

        for i in range(len(edge_links[0])):
            path_flag = False

            if in_path_edges_mask[i] == True:
                path_flag = True

            G.add_edge(
                edge_links[0][i],
                edge_links[1][i],
                weight=edge_weights[i],
                is_in_path=path_flag,
            )
        return G


    def draw(self, G, label_w_weights=False, output_graph=None) -> None:
        """_summary_

        Args:
            G (_type_): _description_
            label_w_weights (bool, optional): _description_. Defaults to False.
            output_graph (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
        """
        if label_w_weights == False and output_graph == None:
            # Make sure we can get use the output values for each node as
            # a label for the network.
            raise Exception(
                "output graph is needed if graph weights are not used for labels"
            )

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

        for i in range(0, len(G.nodes)):
            # Get shortest path nodes ground truth.
            if G.nodes[i]["is_in_path"] == True:
                sp_path.append(i)

        edges_list = list(G.edges(data=True))

        for e in edges_list:
            if e[2]["is_in_path"] == True:
                sp_path_edges.append([e[0], e[1]])

        if label_w_weights:
            # Use the graph weights or the output of the network.
            labels_edge = nx.get_edge_attributes(G, "weight")
            node_labels = {}

            for key in labels_edge:
                labels_edge[key] = np.round(labels_edge[key], 2)

            for i in range(0, len(G.nodes)):
                node_labels[i] = round(G.nodes[i]["weight"], 2)
        else:
            node_labels, labels_edge = prob_labels(output_graph)

        nx.draw_networkx(G, pos=pos, labels=node_labels)

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

        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edge)

        plt.show()


    def prob_labels(self, ot_graph) -> nx.Graph:
        """Output graph labels from  hidden to nx_graph.

        Outputs the labels from the output graph.

        Args:
            ot_graph (_type_): Instead of features the network has a
                hidden state that only contains the outpus.

        Returns:
            node_labels (dict): {node_num: prob_is_in_path 0..1}
            edge_labels (dict): {(s, t): prob_is_in_path 0..1}
        """

        node_labels = {}
        edge_labels = {}
        print(output_graph.node_sets["cities"][tfgnn.HIDDEN_STATE].numpy())
        node_logits = ot_graph.node_sets["cities"][tfgnn.HIDDEN_STATE]
        node_prob = tf.nn.softmax(node_logits)  # assume nodes are in order

        edge_logits = ot_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE]
        edge_prob = tf.nn.softmax(edge_logits)

        for n in range(0, len(ot_graph.node_sets["cities"][tfgnn.HIDDEN_STATE])):
            node_labels[n] = np.round(node_prob[n][1].numpy(), 2)  # prob is in path

        edge_links = np.stack(
            [
                ot_graph.edge_sets["roads"].adjacency.source.numpy(),
                ot_graph.edge_sets["roads"].adjacency.target.numpy(),
            ],
            axis=0,
        )

        for i in range(0, len(ot_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE])):
            edge_labels[(edge_links[0][i], edge_links[1][i])] = np.round(
                edge_prob[i][1].numpy(), 2
            )

        return node_labels, edge_labels


if __name__ == "__main__":

    build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=_set_initial_node_state,
        edge_sets_fn=_set_initial_edge_state,
        context_fn=_set_initial_context_state,
    )

    num_nodes_min_max = (10, 15)
    random_seed = 12
    random_state = np.random.RandomState(random_seed)

    dimensions = 2
    theta = 25
    rate = 1.0
    min_length = 3

    graph = _generate_base_graph(
        random_state,
        num_nodes_min_max=num_nodes_min_max,
        dimensions=dimensions,
        theta=theta,
        rate=rate,
    )
    # Randomly adds a shortest path as they do in the normal code. This is not really nedded unless we want to modify a graph that has been u
    # used for training
    graph_sp = _add_shortest_path(random_state, graph, min_length=min_length)

    tensor_graph_sp = _convert_to_graph_tensor(graph_sp)

    with (open("pickled_200_model_50_60_g_size", "rb")) as openfile:

        trained_gnn = pickle.load(openfile)

    input_graph = build_initial_hidden_state(tensor_graph_sp)
    output_graph = trained_gnn(input_graph)

    node_logits = output_graph.node_sets["cities"][tfgnn.HIDDEN_STATE]
    edge_logits = output_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE]

    # print(output_graph.edge_sets["roads"].get_features_dict())

    """
    Extract Confindence from the output Graph. 
    
    """
    # print("Edge outputs: ", output_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE])
    # print("Node outputs: ", output_graph.node_sets["cities"][tfgnn.HIDDEN_STATE])

    predicted_task_graph = predict_from_final_hidden_state(
        tensor_graph_sp, output_graph
    )

    # # 	print(predicted_task_graph.node_sets["cities"]["is_in_path"])

    predicted_graph_nx = pred_tensor_graph_to_nx_graph(predicted_task_graph)
    # otuput_graph_nx = pred_tensor_graph_to_nx_graph(output_graph)

    # 	print(predicted_graph_nx)
    draw(graph_sp, label_w_weights=True)
    draw(predicted_graph_nx, output_graph=output_graph)
