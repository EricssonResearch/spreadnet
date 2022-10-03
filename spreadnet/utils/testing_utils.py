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
import pickle
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestingUtils:
    def __init__(self) -> None:
        """Load the weights path during training.

        Raise: Exception if the folder does not exist.
        """
        os.chdir("../weights")

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # yaml_path = "configs.yaml"
        # configs = yaml_parser(yaml_path)
        # self.train_configs = configs.train
        # self.model_configs = configs.model

    def load_model(
        self,
        model_name: str,
        model_type: str = "",
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
        if model_type == "tf_gnn":

            with (open(model_name, "rb")) as openfile:
                trained_model = pickle.load(openfile)

        elif model_type == "pyg_gnn":
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

            weight_base_path = self.train_configs["weight_base_path"]
            # model_path = weight_base_path + "model_weights_best.pth"
            model_path = weight_base_path + model_name
            trained_model.load_state_dict(
                torch.load(model_path, map_location=torch.device(device))
            )

        return trained_model

    def inferer_single_data(self, model, model_type, input, prediction_type):
        """
            Return model inference for an input depending on the model type.
        Assumes that the input already is in the format required by the model.

        Args:
            model      (_type_): Trained model.
            model_name      str: tf_gnn or pyg_gnn currently
            input      (_type_): tensor graph for tf_gnn and ...... for pyg_gnn
            prediction_type str: Probability or class as output.
        Returns:
            The output from the model. In a standard node and edge labels.


        Notes:  Some models output a graph instead of the node and edge labels.
                This function returns the output values.
        """

        return
