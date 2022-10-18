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

from statistics import mode
import sys

from black import out
from spreadnet.utils.experiment_utils import ExperimentUtils
from spreadnet.utils.visualization_utils import VisualUtils
from spreadnet.tf_gnn.tf_utils.tf_utils import TfGNNUtils
from spreadnet.tf_gnn.model import gnn
import matplotlib.pyplot as plt
from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.utils import yaml_parser
from spreadnet.pyg_gnn.models import *

from spreadnet.pyg_gnn.models import EncodeProcessDecode

sys.modules["EncodeProcessDecode"] = EncodeProcessDecode


sys.modules["gnn"] = gnn

import webdataset as wds
import argparse
import tensorflow_gnn as tfgnn
import tensorflow as tf
import numpy as np
import pickle
import json
from os import path as osp
import networkx as nx


"""
    The following is boilerplate code and should be dealt with.
    It is ugly and keeps beeing reproduced in different formats
    all over the placed.

"""
# parser = argparse.ArgumentParser(description="Do predictions.")
# args = parser.parse_args()
# yaml_path = args.config
# which_model = args.model
# configs = yaml_parser(yaml_path)
# train_configs = configs.train
# model_configs = configs.model
# data_configs = configs.data
# dataset_path = osp.join(osp.dirname(__file__), data_configs["dataset_path"]).replace(
#     "\\", "/"
# )
# raw_path = dataset_path + "/raw"


def single_graph_implementation_test_helper():
    """

    Temporary test to make sure that the visualization integration works.
    It uses a trained pickled tf_gnn model and it generates a graph using
    the utils from the tf_gnn repository that came with the paper.

    """

    vis = VisualUtils()
    models_trained = [
        ExperimentUtils(model_type="tf_gnn", weights_model="pickled_2000_model.pickle"),
        ExperimentUtils(model_type="pyg_gnn", weights_model="model_weights_best.pth"),
    ]

    # TODO fix this path thing with the configs
    # graphs = json.load(raw_path + "random.json")
    # Broken window priciple applies tho
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


if __name__ == "__main__":

    single_graph_implementation_test_helper()
