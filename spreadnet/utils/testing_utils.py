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
from spreadnet.utils.config_parser import yaml_parser
import pickle
import os


class TestingUtils:
    def __init__(self) -> None:
        """Load the weights path during training.

        Raise: Exception if the folder does not exist.
        """
        os.chdir("../weights")
        curdir = os.getcwd()
        print("\n\nWorking dir:", curdir, "\n\n")
        print(os.listdir())
        # configs = yaml_parser(yaml_path)
        # weights_train = configs.train
        # self.weights_path = weights_train["weights"]

        # print(self.weights_path)

    def load_model(self, name: str, pickled: bool = False):
        """Loads a specifc model.
        If it is a tensorflow model it loads it from a pickle.
        The class pickled needs to exist when unpickling.
        If it is a PyG model it loads it from the specific
        TODO: Load pyG model.
        Args:
            name (str): dataset name
            pickled(bool): Is pickled
        """

        with (open("pickled_2000_model.pickle", "rb")) as openfile:
            trained_gnn = pickle.load(openfile)

        return trained_gnn
