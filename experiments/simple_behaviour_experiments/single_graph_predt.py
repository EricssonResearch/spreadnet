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

from spreadnet.utils.testing_utils import TestingUtils
from spreadnet.utils.visualization_utils import VisualUtils
from spreadnet.utils.tf_utils import TfGNNUtils

import tensorflow_gnn as tfgnn
import tensorflow as tf
import numpy as np
import pickle


def single_graph_implementation_test_helper(trained_gnn):
    """

    Temporary test to make sure that the visualization integration works.
    It uses a trained pickled tf_gnn model and it generates a graph using
    the utils from the tf_gnn repository that came with the paper.

    TODO: remove after integration with the data loader and PyG
          new data should not be generated during the experiments.

    """

    vis = VisualUtils()
    tf_utils = TfGNNUtils()

    build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=tf_utils._set_initial_node_state,
        edge_sets_fn=tf_utils._set_initial_edge_state,
        context_fn=tf_utils._set_initial_context_state,
    )

    num_nodes_min_max = (10, 15)
    random_seed = 12
    random_state = np.random.RandomState(random_seed)

    dimensions = 2
    theta = 25
    rate = 1.0
    min_length = 3

    graph = tf_utils._generate_base_graph(
        random_state,
        num_nodes_min_max=num_nodes_min_max,
        dimensions=dimensions,
        theta=theta,
        rate=rate,
    )
    # Randomly adds a shortest path as they do in the normal code. This is not really nedded unless we want to modify a graph that has been u
    # used for training
    graph_sp = tf_utils._add_shortest_path(random_state, graph, min_length=min_length)

    tensor_graph_sp = tf_utils._convert_to_graph_tensor(graph_sp)

    with (open("../pickled_200_model_50_60_g_size", "rb")) as openfile:

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
    vis.nx_draw(graph_sp, label_w_weights=True)
    vis.nx_draw(predicted_graph_nx, output_graph=output_graph)


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


if __name__ == "__main__":
    yaml_path = "configs.yaml"
    ts = TestingUtils(yaml_path="../configs.yaml")
    trained_gnn = ts.load("pickled_2000_model.pickle")
    single_graph_implementation_test_helper(trained_gnn)
