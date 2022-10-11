"""
Class for utilities required by the testing suite. 
Repetitive functions to be included. 


Assumes that the code is run in a folder placed in experiments. 
It goes one folder back to pick up the weights. 


This testing utils needs to know where the weights are. 
TODO: Add the weights path 
TODO: make the class not change the working dirrectory
      durring ussage. 


"""
# from spreadnet.tf_gnn.gnn import *
from black import out
from spreadnet.utils.config_parser import yaml_parser
from spreadnet.pyg_gnn.models import *
import tensorflow_gnn as tfgnn

# from spreadnet.pyg_gnn.models.encode_process_decode.models import (
#     Encoder,
#     Decoder,
#     EncodeProcessDecode,
#     Processor,
# )

# from spreadnet.pyg_gnn.models.encode_process_decode.models import EncodeProcessDecode
from spreadnet.utils.tf_utils import TfGNNUtils
import pickle
import os
from os import path as osp
import torch
import sys
import argparse
import networkx as nx
import tensorflow as tf
from spreadnet.datasets.data_utils.convertor import graphnx_to_dict_spec
from torch_geometric.data import Data


class ExperimentUtils:
    def __init__(self, model_type: str = "", weights_model: str = "") -> None:
        """Load the weights path during training.

        Raise: Exception if the folder does not exist.
        """
        self.current_folder = os.getcwd()
        self.trained_model = None
        self.implemented_models = ["tf_gnn", "pyg_gnn"]
        self.model_type = model_type
        self.model_weights = weights_model  # name of the weights file

        self._load_model()

        """
        TODO: add in configs the trained models for each model type
        those models are taken from the weights folder. But chosing one is harder as it should require some kind
         of standardized naming scheme.  

        Current solution during the experiment we chose the specific model. Deciding on the trained model weights
        to be used is done during the experiment. 

        """

        self.model_type = model_type
        if model_type not in self.implemented_models:
            sys.exit("The model type given is not part of the implemented models.")

    def _goto_weights(self):
        """
        Changes the current directory to the weights directory.

        Assumens that it is inside a directory in experiments and that the weights are in a
        weights directory in the same folder.

        Always call _return
        """
        os.chdir("../weights")

    def _return_from_weights(self):
        """Goes back to the initial directory."""

        os.chdir(self.current_folder)

    def show_available_models(self, ret=False):
        """
            Prints or returns a list of the available models.
        Args:
            ret (bool, optional): Return the list of implemented models. Defaults to False.


        Returns:
            array[str]: _description_
        """

        if ret:
            return self.implemented_models
        else:
            print("Models Implemented: ", self.implemented_models)
            return []

    def show_model_used(self, ret=False):
        """Print or/and return the name of the model the class instance is using.

        Args:
            ret (bool, optional): To return or not to return. Defaults to False.

        Returns:
            str: String with the model used for this class instance.
        """
        if ret:
            return (self.model_type, self.model_weights)
        else:
            print(self.model_type, self.model_weights)

    def _load_model(
        self,
    ):
        """Loads a specifc model.
        If it is a tensorflow model it loads it from a pickle.
        The class pickled needs to exist when unpickling.
        If it is a PyG model it loads it from the specific
        TODO: Load pyG model.
        Args:
            name (str): dataset name
            pickled(bool): Is pickled
        """
        self._goto_weights()  # TODO Can replace the directory traversal with the weights base path making.

        if self.model_type == "tf_gnn":
            print(self.model_weights)
            with (open(self.model_weights, "rb")) as openfile:
                trained_model = pickle.load(openfile)

        else:
            # self.model_type == "pyg_gnn":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            default_yaml_path = osp.join(osp.dirname(__file__), "configs.yaml")

            parser = argparse.ArgumentParser(description="Do predictions.")
            parser.add_argument(
                "--config",
                default=default_yaml_path,
                help="Specify the path of the config file. ",
            )
            parser.add_argument(
                "--model",
                default="model_weights_best.pth",
                help="Specify the model we want to use.",
            )
            args = parser.parse_args()
            yaml_path = args.config
            configs = yaml_parser(yaml_path)

            train_configs = configs.train
            model_configs = configs.model
            trained_model = EncodeProcessDecode(
                node_in=model_configs["node_in"],
                edge_in=model_configs["edge_in"],
                node_out=model_configs["node_out"],
                edge_out=model_configs["edge_out"],
                latent_size=model_configs["latent_size"],
                num_message_passing_steps=model_configs["num_message_passing_steps"],
                num_mlp_hidden_layers=model_configs["num_mlp_hidden_layers"],
                mlp_hidden_size=model_configs["mlp_hidden_size"],
            ).to(device)
            print("\n\n\n", os.getcwd(), "\n\n\n\n")
            trained_model.load_state_dict(
                torch.load(self.model_weights, map_location=torch.device(device))
            )

        self._return_from_weights()
        self.trained_model = trained_model

    def inferer_single_data(self, input_graph):
        """
            Return model inference for an input depending on the model type.
        Assumes that the input already is in the format required by the model.

        Args:
            model      (_type_): Trained model.
            model_name      str: tf_gnn or pyg_gnn currently
            input      (_type_): Input in nx internal standard format
            prediction_type str: Probability or class as output.
        Returns:
            The output from the model. In a standard node and edge labels.


        Notes:  Some models output a graph instead of the node and edge labels.
                This function returns the output values.
        """

        """
        Convert the data to the format specific for the model.

        """
        input_graph = nx.node_link_graph(input_graph, directed=True, multigraph=False)
        if self.model_type == "tf_gnn":

            # Predict
            tf_ut = TfGNNUtils()
            graph_tensor = tf_ut.convert_to_graph_tensor(input_graph)
            tensor_input_graph = tf_ut.build_initial_hidden_state(graph_tensor)
            output_graph = self.trained_model(tensor_input_graph)

            std_output_graph = tf_ut.nx_standard_format_from_tensor(
                input_graph, output_graph
            )
            # print("What: ", std_output_graph.edges(data=True))
        elif self.model_type == "pyg_gnn":
            # Convert networkx graph to pyg_gnn input Graph
            processed_graph = self.process(input_graph)
            # make prediction

            # print("\n\nProcessed Graph", processed_graph, "\n\n")

            nodes_output, edges_output = self.trained_model(
                processed_graph.x, processed_graph.edge_index, processed_graph.edge_attr
            )
            # convert to common format
            # print(nodes_output, edges_output)

            std_output_graph = self._std_from_pyg_gnn(
                input_graph=input_graph,
                prediction_edges=edges_output,
                prediction_nodes=nodes_output,
            )

            # Append prediction to pyg_gnn

        return std_output_graph

    def _std_from_pyg_gnn(self, input_graph, prediction_edges, prediction_nodes):
        graph_updated = input_graph
        prediction_nodes = prediction_nodes.detach().numpy()
        prediction_edges = prediction_edges.detach().numpy()

        node_labels = {}
        for i in range(len(prediction_nodes)):
            node_labels[i] = prediction_nodes[i]
        nx.set_node_attributes(graph_updated, node_labels, "logits")

        edge_index_data = [list(tpl) for tpl in input_graph.edges]
        edge_index_t = torch.tensor(edge_index_data, dtype=torch.long)
        edge_index = edge_index_t.t().contiguous()

        edge_labels = {}
        for i in range(0, len(edge_index[0])):
            edge_labels[(edge_index.numpy()[0][i], edge_index.numpy()[1][i])] = {
                "logits": prediction_edges[i]
            }

        nx.set_edge_attributes(graph_updated, edge_labels)

        return graph_updated

    def process(self, graph_nx):

        graph_dict = graphnx_to_dict_spec(graph_nx)
        # Get ground truth labels.
        node_tensor = torch.tensor(graph_dict["nodes_feature"]["is_in_path"])
        node_labels = node_tensor.type(torch.int64)

        edge_tensor = torch.tensor(graph_dict["edges_feature"]["is_in_path"])
        edge_labels = edge_tensor.type(torch.int64)

        nodes_data = [data for _, data in graph_nx.nodes(data=True)]
        nodes_weight = torch.tensor(
            [data["weight"] for data in nodes_data], dtype=torch.float
        ).view(-1, 1)
        nodes_is_start = torch.tensor(
            [data["is_start"] for data in nodes_data], dtype=torch.int
        ).view(-1, 1)
        nodes_is_end = torch.tensor(
            [data["is_end"] for data in nodes_data], dtype=torch.int
        ).view(-1, 1)
        nodes_pos = torch.tensor(
            [data["pos"] for data in nodes_data], dtype=torch.float
        )
        x = torch.cat((nodes_weight, nodes_is_start, nodes_is_end), 1)

        _, _, edges_data = zip(*graph_nx.edges(data=True))
        edges_weight = torch.tensor(
            [data["weight"] for data in edges_data], dtype=torch.float
        ).view(-1, 1)

        # get edge_index from graph_nx
        edge_index_data = [list(tpl) for tpl in graph_nx.edges]
        edge_index_t = torch.tensor(edge_index_data, dtype=torch.long)
        edge_index = edge_index_t.t().contiguous()

        data = Data(
            edge_index=edge_index,
            pos=nodes_pos,
            x=x,
            edge_attr=edges_weight,
            y=(node_labels, edge_labels),
        )

        return data
