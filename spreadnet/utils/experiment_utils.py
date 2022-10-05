"""
Class for utilities required by the testing suite. 
Repetitive functions to be included. 


Assumes that the code is run in a folder placed in experiments. 
It goes one folder back to pick up the weights. 


This testing utils needs to know where the weights are. 
TODO: Add the weights path 
TODO: make the class not change the working dirrectory
      durring ussage. 
TODO: The testing util should decide on the model type at the start instead of
      needing to specify it at every function call.

"""
# from spreadnet.tf_gnn.gnn import *
from spreadnet.utils.config_parser import yaml_parser

from spreadnet.pyg_gnn.models.encode_process_decode.models import EncodeProcessDecode
from spreadnet.utils.tf_utils import TfGNNUtils
import pickle
import os
import torch
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentUtils:
    def __init__(self, model_type: str = "", weights_model: str = "") -> None:
        """Load the weights path during training.

        Raise: Exception if the folder does not exist.
        """
        self.current_folder = os.getcwd()
        self.trained_model = None
        self.implemented_models = ["tf_gnn", "pyg_gnn"]
        self.model_type = model_type
        self.model_weights = weights_model
        self._goto_weights()  # TODO Can replace the directory traversal with the weights base path making.
        self._load_model()
        self._return_from_weights()

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

        self.load_model()

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
        path_initial_dir = "../" + self.current_folder
        os.chdir(path_initial_dir)

    def show_available_models(self, ret=False) -> list[str]:
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

    def show_model_used(self, ret=False) -> list(str):
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
        if self.model_type == "tf_gnn":

            with (open(self.model_weights, "rb")) as openfile:
                trained_model = pickle.load(openfile)

        else:
            # self.model_type == "pyg_gnn":
            trained_model = EncodeProcessDecode(
                node_in=self.model_configs["node_in"],
                edge_in=self.model_configs["edge_in"],
                node_out=self.model_configs["node_out"],
                edge_out=self.model_configs["edge_out"],
                latent_size=self.model_configs["latent_size"],
                num_message_passing_steps=self.model_configs[
                    "num_message_passing_steps"
                ],
                num_mlp_hidden_layers=self.model_configs["num_mlp_hidden_layers"],
                mlp_hidden_size=self.model_configs["mlp_hidden_size"],
            ).to(self.device)

            trained_model.load_state_dict(
                torch.load(self.model_weights, map_location=torch.device(device))
            )

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

        if self.model_type == "tf_gnn":
            tf_ut = TfGNNUtils()
            graph_tensor = tf_ut._convert_to_graph_tensor(input_graph)
            input_graph = tf_ut.build_initial_hidden_state(graph_tensor)
            output_graph = self.trained_model(input_graph)
            
        else:


        return

    def _convert_to_standard_nx_format(self):
        """Takes the output from an inference and returns the internal nx standard data format."""
        pass
