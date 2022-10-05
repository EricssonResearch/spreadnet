"""
Plots the prediction with prediction probabilities for a single graph.    
single graphs. This could help us understand how the network behaves. 


TODO:
    1. Connect to the data loader. 
    2. Connect to the same graph generator PyG uses for new graph 
        generation.
    3. Finish defining the tests and their purpose in the
        context.
    4. Connect to utils so that both PyG and tensorflow can be tested.    

Reminder:
    Same test should work for both implementations. 
    Implementation specifc operation should be dealt with
    in the utils not here. 
"""

import sys
from spreadnet.utils.experiment_utils import ExperimentUtils
from spreadnet.utils.visualization_utils import VisualUtils
from spreadnet.utils.tf_utils import TfGNNUtils
from spreadnet.tf_gnn.model import gnn
import matplotlib.pyplot as plt
from spreadnet.pyg_gnn.models import EncodeProcessDecode

sys.modules["gnn"] = gnn


import tensorflow_gnn as tfgnn
import tensorflow as tf
import numpy as np
import pickle
import json
import os


def single_graph_implementation_test_helper(
    trained_gnn, model_used: str = True, graph_nx=None
):
    """

    Temporary test to make sure that the visualization integration works.
    It uses a trained pickled tf_gnn model and it generates a graph using
    the utils from the tf_gnn repository that came with the paper.

    TODO: remove after integration with the data loader and PyG
          new data should not be generated during the experiments.

    """

    vis = VisualUtils()
    tf_utils = TfGNNUtils()
    if model_used == "tf_gnn":
        build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=tf_utils._set_initial_node_state,
            edge_sets_fn=tf_utils._set_initial_edge_state,
            context_fn=tf_utils._set_initial_context_state,
        )

        graph_sp = graph_nx

        tensor_graph_sp = tf_utils._convert_to_graph_tensor(graph_sp)
        input_graph = build_initial_hidden_state(tensor_graph_sp)
        output_graph = trained_gnn(input_graph)

        predicted_task_graph = tf_utils.predict_from_final_hidden_state(
            tensor_graph_sp, output_graph
        )

        predicted_graph_nx = tf_utils.tf_pred_tensor_graph_to_nx_graph(
            predicted_task_graph
        )

    elif model_used == "pyg_gnn":
        """
        Data Format changed need to pull from main before I can use the new data format.
        """
        pass

    plt.figure(1)

    vis.nx_draw(graph_sp, label_w_weights=True)
    plt.figure(2)

    vis.nx_draw(predicted_graph_nx, output_graph=output_graph)


def single_graph_vis_pyg_test0(trained_gnn):
    pass


def single_graph_vis_test1():
    """
    Unseen Graphs

    Pass a new unsen graph to the trainable GNN and see if it finds the correct shortest path.

    """


def single_graph_vis_test2():
    """
    Take a graph that has been seen before and check the behaviour if you have different start and end nodes.

    """
    pass


def single_graph_vis_test3():
    """
    The demo report only shows accuracy and loss on the graphs that the network has been trained on.
    What about the accuracy on graphs that the network has not been trained on?

    """

    pass


def gen_local_graph():
    # TODO: Remove this graph generating part after the dataset handling is done.
    tf_utils = TfGNNUtils()

    num_nodes_min_max = (30, 40)
    random_seed = 34567
    random_state = np.random.RandomState(random_seed)

    dimensions = 2
    theta = 25
    rate = 0.5
    min_length = 6

    graph = tf_utils._generate_base_graph(
        random_state,
        num_nodes_min_max=num_nodes_min_max,
        dimensions=dimensions,
        theta=theta,
        rate=rate,
    )

    graph_sp = tf_utils._add_shortest_path(random_state, graph, min_length=min_length)

    return graph_sp


if __name__ == "__main__":
    models_trained = [ExperimentUtils("tf_gnn"), ExperimentUtils("pyg_gnn")]

    trained_gnn = ts.load_model("pickled_2000_model.pickle", "tf_gnn")

    yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
    configs = yaml_parser(yaml_path)
    data_configs = configs.data

    random_seed = data_configs["random_seed"]
    num_nodes_min_max = (data_configs["num_node_min"], data_configs["num_node_max"])
    dataset_size = data_configs["dataset_size"]
    dataset_path = os.path.join(os.path.dirname(__file__), data_configs["dataset_path"])
    raw_path = dataset_path + "/raw"

    graphs_data = json.load(raw_path + "random.json")
    print(graphs_data)

    # Locally generated graph, get the graphs from the

    # single_graph_implementation_test_helper(trained_gnn, model_used="tf_gnn", graph_nx=graph_w_sp)
