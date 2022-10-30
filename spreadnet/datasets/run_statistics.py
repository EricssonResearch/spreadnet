import csv
import json
import glob
import pandas as pd
from datetime import datetime
import os

from spreadnet.utils import yaml_parser


class RunStatistics:
    """Captures the data's statistics for every run."""

    def __init__(self):
        """Initializes and checks for the file for statistics and headers."""
        self.statistics_file = "statistics.csv"
        project_folder = os.path.sep.join(
            os.path.abspath(os.path.realpath(__file__)).split(os.path.sep)[:-3]
        )
        open(self.statistics_file, "a")
        self.add_headers()
        self.add_time()
        self.parse_yaml(
            [
                project_folder + "/experiments/dataset_configs.yaml",
                project_folder + "/experiments/encode_process_decode/configs.yaml",
            ]
        )
        self.check_nodes_and_edges()

    def add_headers(self):
        """Adds and checks for headers in the statistics file."""
        with open(self.statistics_file, "r+") as csv_file:
            dict_reader = csv.DictReader(csv_file)
            writer = csv.writer(csv_file)
            header = dict_reader.fieldnames
            if header is None:
                header_list = [
                    "time",
                    "random_seed",
                    "num_node_min",
                    "num_node_max",
                    "num_nodes",
                    "starting_theta",
                    "max_edges_per_node",
                    "avg_edges_per_node",
                    "dataset_size",
                    "dataset_path",
                    "raw_path",
                    "weighted_(nodes)",
                    "weighted_(edges)",
                    "weight_save_freq",
                    "weight_base_path",
                    "best_weight_name",
                    "epochs",
                    "batch_size",
                    "shuffle",
                    "avg_hops",
                    "min_hops",
                    "max_hops",
                    "node_in",
                    "edge_in",
                    "node_out",
                    "edge_out",
                    "latent_size",
                    "num_of_message_passing_steps",
                    "num_of_mlp_hidden_layers",
                    "mlp_hidden_size",
                    "adam_lr",
                    "adam_weight_decay",
                ]
                writer.writerow(header_list)

    def check_nodes_and_edges(self):
        """Checks if the nodes and edges are weighted."""
        project_folder = os.path.sep.join(
            os.path.abspath(os.path.realpath(__file__)).split(os.path.sep)[:-3]
        )
        json_dataset_path = os.path.join(project_folder, "experiments/dataset/raw")
        json_file_path = list(
            map(os.path.basename, glob.glob(json_dataset_path + "/*.json"))
        )
        json_file_path = os.path.join(json_dataset_path, json_file_path[0])
        with open(json_file_path, "r") as json_file:
            json_data = json.load(json_file)
            self.add_data("weighted_(nodes)", "True") if json_data[0]["nodes"][0][
                "weight"
            ] != "" else self.add_data("weighted_(nodes)", "False")
            self.add_data("weighted_(edges)", "True") if json_data[0]["links"][0][
                "weight"
            ] != "" else self.add_data("weighted_(edges)", "False")

    def add_time(self):
        """Creates a new row and initializes current time onto it."""
        statistics = pd.read_csv(self.statistics_file)
        row_length = len(statistics.index)
        current_time = datetime.now().strftime(r"%d/%m/%Y %H:%M:%S")
        statistics.loc[row_length, "time"] = current_time
        statistics.to_csv(self.statistics_file, encoding="utf-8", index=False)

    def add_data(self, column_name, data):
        """Add data into the statistics.csv file
        Args:
            column_name:
                Creates a column of the given name or if adds 'data'
                to the column if it exists
            data:
                Data to be added to the 'column_name'
        """
        statistics = pd.read_csv(self.statistics_file)
        row_length = len(statistics.index)
        statistics.loc[row_length - 1, column_name] = data
        statistics.to_csv(self.statistics_file, encoding="utf-8", index=False)

    def parse_yaml(self, file_paths):
        """Parse the configs.yaml files.

        Args:
            file_paths:
                list of file paths of the config files being analyzed
                The path being passed should be from the project folder
                to the config file and pass it in a list
        """
        statistics = pd.read_csv(self.statistics_file)
        row_length = len(statistics.index)

        try:
            data_configs = yaml_parser(file_paths[0])
            configs = yaml_parser(file_paths[1])

            data_configs = data_configs.data
            train_configs = configs.train
            model_configs = configs.model

        except FileNotFoundError:
            print(
                """Configs file could not be read properly. Please
                check the sections inside the configs.yaml file."""
            )

        try:
            for data in data_configs:
                statistics.loc[row_length - 1, data] = data_configs[data]

            for train in train_configs:
                statistics.loc[row_length - 1, train] = train_configs[train]

            for model in model_configs:
                statistics.loc[row_length - 1, model] = model_configs[model]

            statistics.to_csv(self.statistics_file, encoding="utf-8", index=False)

        except TypeError:
            print("Contents of the config file are not iterable.")
