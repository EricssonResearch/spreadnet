"""
Class for utilities required by the testing suite. 
Repetitive functions to be included. 

"""
# from config_parser import yaml_parser
import pickle
import argparse

import yaml


def yaml_parser(yaml_path: str):
    """
    Parse `.yaml` config file.

    Args:
        yaml_path: The path of the `.yaml` config file.

    Returns:
        A dictionary that contains the configs.

    """
    with open(yaml_path, "r") as file:
        configs = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))

    return configs


class TestingUtils:
    def __init__(self, yaml_path) -> None:
        """Load the weights path during training.

        Raise: Exception if the folder does not exist.
        """
        configs = yaml_parser(yaml_path)
        weights_train = configs.train
        self.weights_path = weights_train["weights"]

        print(self.weights_path)

    def load_model(self, name: str, pickled: bool = False):
        """Loads a specifc model.
        If it is a tensorflow model it loads it from a pickle
        If it is a PyG model it loads it from the specific
        TODO: Load pyG model.
        Args:
            name (str): dataset name
            pickled(bool): Is pickled
        """

        with (open(self.weights_path + "pickled_2000_model.pickle", "rb")) as openfile:
            trained_gnn = pickle.load(openfile)

        return trained_gnn
