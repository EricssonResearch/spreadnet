"""Plots the prediction with prediction probabilities for a single graph.
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
    Implementation specific operation should be dealt with
    in the utils not here.
"""


import sys
import json

from spreadnet.utils.experiment_utils import ExperimentUtils
from spreadnet.utils.visualization_utils import VisualUtils
from spreadnet.tf_gnn.model import gnn


from spreadnet.pyg_gnn.models import MPNN

sys.modules["MPNN"] = MPNN
sys.modules["gnn"] = gnn


def single_graph_implementation_test_helper():
    """Temporary test to make sure that the visualization integration works.

    It uses a trained pickled tf_gnn model and it generates a graph using the
    utils from the tf_gnn repository that came with the paper.
    """

    vis = VisualUtils()
    models_trained = [
        ExperimentUtils(model_type="tf_gnn", weights_model="pickled_2000_model.pickle"),
        ExperimentUtils(model_type="pyg_gnn", weights_model="model_weights_best.pth"),
    ]

    # TODO fix this path thing with the configs

    datasets = [
        "random.json",
        "random_25-35.20.json",
        "random_100-101.20.json",
        "random_100-101.20.json",
        "random_250-251.100.json",
    ]
    for d in datasets:

        raw_data_path = "../dataset/raw/" + d
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        single_graph = graphs[1]  # std format

        output_graph_tf = models_trained[0].inferer_single_data(single_graph)
        output_graph_pyg = models_trained[1].inferer_single_data(single_graph)

        # print("\n\n\nOutput TFF_edges", output_graph_tf.edges.data())
        # print("\n\n\nOutput TFF_nodes", output_graph_tf.nodes.data())

        # print("\n\n\n output pygg_edges", output_graph_pyg.edges.data())
        # print("\n\n\n output pygg_nodes", output_graph_pyg.nodes.data())

        vis.new_nx_draw(output_graph_pyg, ground_truth=True, title=d, save=True)
        vis.new_nx_draw(
            output_graph_tf, ground_truth=False, title="Tensorflow" + d, save=True
        )
        vis.new_nx_draw(
            output_graph_pyg, ground_truth=False, title="Pyg GNN" + d, save=True
        )


if __name__ == "__main__":
    single_graph_implementation_test_helper()
