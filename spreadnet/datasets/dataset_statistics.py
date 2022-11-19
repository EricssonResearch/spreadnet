import csv
import json
import glob
import pandas as pd
from datetime import datetime
import os
from itertools import groupby, islice
import wandb

from spreadnet.utils import yaml_parser
from experiments import generate_dataset


class DatasetStatistics:
    """Captures the data's statistics for every dataset generated."""

    def __init__(self, dataset_logger):
        """Initializes and checks for the file for statistics and headers.

        Args:
            dataset_logger:
                Pass a logger object to help log the errors and exceptions
        """
        self.dataset_logger = dataset_logger
        self.statistics_file = "statistics.csv"
        self.dataset_type = ["train", "test"]
        project_folder = self.get_project_folder_path()
        try:
            open(self.statistics_file, "a")
        except PermissionError:
            self.dataset_logger.exception(
                f"Close {self.statistics_file} and try again."
            )
        self.training_dataset, self.testing_dataset = self.open_file()
        self.add_headers()
        self.add_time()
        self.parse_yaml(
            [
                project_folder + "/experiments/dataset_configs.yaml",
                project_folder + "/experiments/encode_process_decode/configs.yaml",
            ]
        )

        self.check_nodes_and_edges()
        self.check_hops_in_dataset()
        self.check_avg_num_nodes()
        self.check_number_of_edges_per_node()
        self.use_wandb = generate_dataset.use_wandb

    def add_headers(self):
        """Adds and checks for headers in the statistics file."""
        with open(self.statistics_file, "r+") as csv_file:
            dict_reader = csv.DictReader(csv_file)
            writer = csv.writer(csv_file)
            header = dict_reader.fieldnames
            if header is None:
                header_list = ["time", "raw_path"]
                writer.writerow(header_list)

    def get_project_folder_path(self):
        """Get the project folder's path."""
        return os.path.sep.join(
            os.path.abspath(os.path.realpath(__file__)).split(os.path.sep)[:-3]
        )

    def open_file(self):
        """Open dataset_statistics file."""
        json_file_paths = self.get_dataset_path()
        training_dataset = []
        testing_dataset = []
        for json_file_path in json_file_paths:
            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)
                if "test" in json_file_path:
                    testing_dataset.append(json_data)
                else:
                    training_dataset.append(json_data)
            json_file.close()
        return training_dataset, testing_dataset

    def get_dataset_path(self):
        """Get a list of path of datasets."""
        project_folder = self.get_project_folder_path()
        json_dataset_path = os.path.join(project_folder, "experiments/dataset/raw")
        json_file_paths = list(
            map(os.path.basename, glob.glob(json_dataset_path + "/*.json"))
        )
        for idx, file_path in enumerate(json_file_paths):
            json_file_paths[idx] = os.path.join(json_dataset_path, file_path)
        return json_file_paths

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

    def check_nodes_and_edges(self):
        """Checks if the nodes and edges are weighted.

        Args:
            json_file_paths: A list of file paths of the datasets you
            want to check the nodes and edges for.
        """
        self.add_data("weighted_(nodes)_train", "True") if self.training_dataset[0][0][
            "nodes"
        ][0]["weight"] != "" else self.add_data("weighted_(nodes)", "False")
        self.add_data("weighted_(edges)_train", "True") if self.training_dataset[0][0][
            "links"
        ][0]["weight"] != "" else self.add_data("weighted_(edges)", "False")
        self.add_data("weighted_(nodes)_test", "True") if self.testing_dataset[0][0][
            "links"
        ][0]["weight"] != "" else self.add_data("weighted_(nodes)", "False")
        self.add_data("weighted_(edges)_test", "True") if self.testing_dataset[0][0][
            "links"
        ][0]["weight"] != "" else self.add_data("weighted_(edges)", "False")

    def check_hops_in_dataset(self):
        """Checks the number of hops in the dataset.

        Args:
            json_file_paths: A list of file paths of the datasets you
            want to check the shortest path distance for.
        """
        train_hops = []
        test_hops = []
        hop = 0

        for graphs in self.training_dataset:
            for link in graphs[0]["links"]:
                if link["is_in_path"] is True:
                    hop += 1
            train_hops.append(hop)
            hop = 0
        for graphs in self.testing_dataset:
            for link in graphs[0]["links"]:
                if link["is_in_path"] is True:
                    hop += 1
            test_hops.append(hop)
            hop = 0

        self.add_data("avg_hops_train", round(sum(train_hops) / len(train_hops), 3))
        self.add_data("min_hops_train", min(train_hops))
        self.add_data("max_hops_train", max(train_hops))
        self.add_data("avg_hops_test", round(sum(test_hops) / len(test_hops), 3))
        self.add_data("min_hops_test", min(test_hops))
        self.add_data("max_hops_test", max(test_hops))

    def check_avg_num_nodes(self):
        """Checks the average number of nodes in the dataset.

        Args:
            json_file_paths: A list of file paths of the datasets you
            want to check the average number of nodes for.
        """
        train_nodes = []
        test_nodes = []

        for graphs in self.training_dataset:
            train_nodes.append(len(graphs[0]["nodes"]))

        for graphs in self.testing_dataset:
            test_nodes.append(len(graphs[0]["nodes"]))

        self.add_data("avg_nodes_train", round(sum(train_nodes) / len(train_nodes), 3))
        self.add_data("avg_nodes_test", round(sum(test_nodes) / len(test_nodes), 3))

    def check_number_of_edges_per_node(self):
        """Checks the average and the max number of edges per node in the
        dataset.

        Args:
            json_file_paths: A list of file paths of the datasets you
            want to check the average number of edges per node for.
        """
        train_edges = []
        test_edges = []
        train_source = []
        test_source = []
        test_max_edge = 0
        train_max_edge = 0

        for graph in self.training_dataset:
            for link in graph[0]["links"]:
                train_source.append(link["source"])
            train_graph_edges = [
                len(list(group)) for key, group in groupby(train_source)
            ]
            train_edges.append(
                round(sum(train_graph_edges) / len(train_graph_edges), 3)
            )
            if max(train_graph_edges) > train_max_edge:
                train_max_edge = max(train_graph_edges)
            train_source = []

        for graph in self.testing_dataset:
            for link in graph[0]["links"]:
                test_source.append(link["source"])
            test_graph_edges = [len(list(group)) for key, group in groupby(test_source)]
            test_edges.append(round(sum(test_graph_edges) / len(test_graph_edges), 3))
            if max(test_graph_edges) > test_max_edge:
                test_max_edge = max(test_graph_edges)
            test_source = []
        self.add_data(
            "avg_edges_per_node_train", round(sum(train_edges) / len(train_edges), 3)
        )
        self.add_data("max_edges_per_node_train", train_max_edge)
        self.add_data(
            "avg_edges_per_node_test", round(sum(test_edges) / len(test_edges), 3)
        )
        self.add_data("max_edges_per_node_test", test_max_edge)

    def upload_dataset_statistics(self, json_file_paths):
        if self.use_wandb:
            statistics = pd.read_csv(self.statistics_file)
            row_length = len(statistics.index)
            with open(self.statistics_file, "r+") as csv_file:
                dict_reader = csv.DictReader(csv_file)
                row = next(islice(dict_reader, row_length - 1, row_length))

            for type in self.dataset_type:
                with wandb.init(
                    entity="pbs",
                    project="artifacts-testing",
                    job_type="data-versioning",
                ) as version:
                    raw_data = wandb.Artifact(
                        name=f"{type}_graph",
                        type="dataset",
                        description=f"{type} dataset for shortest path",
                        metadata={
                            "time": row["time"],
                            "random_seed": row[f"random_seed_{type}"],
                            "num_node_min": row[f"num_node_min_{type}"],
                            "num_node_max": row[f"num_node_max_{type}"],
                            "batch_size": row["batch_size"],
                            "shuffle": row["shuffle"],
                            "latent_size": row["latent_size"],
                            "min_path_length": row["min_path_length"],
                            "visualize_graph": row["visualize_graph"],
                            "train_ratio": row["train_ratio"],
                            "plot_after_epochs": row["plot_after_epochs"],
                            "weighted_(nodes)": row[f"weighted_(nodes)_{type}"],
                            "weighted_(edges)": row[f"weighted_(edges)_{type}"],
                            "avg_hops": row[f"avg_hops_{type}"],
                            "min_hops": row[f"min_hops_{type}"],
                            "max_hops": row[f"max_hops_{type}"],
                            "avg_nodes": row[f"avg_nodes_{type}"],
                            "avg_edges_per_node": row[f"avg_edges_per_node_{type}"],
                            "max_edges_per_node": row[f"max_edges_per_node_{type}"],
                        },
                    )

                    version.log_artifact(raw_data)
        else:
            self.dataset_logger.error(
                "Method using wandb called without initializing wandb flag"
            )

    def download_dataset_wandb(self, project, job_type, entity="uu-spreadnet"):
        """Download the testing and the training dataset from wandb.

        Args:
            entity: the team on wandb you want to run the program in
                    default value is 'uu-spreadnet'
                    please change if you are testing
            project: name of the project
            job_type: type of task being performed
        """
        if self.use_wandb:
            with wandb.init(entity=entity, project=project, job_type=job_type) as run:
                for type in self.dataset_type:
                    artifact = wandb.Artifact(
                        f"{type}_graph",
                        type="dataset",
                        description=f"{type} dataset for shortest path",
                    )

                    raw_data_artifact = run.use_artifact(f"{type}_graph:latest")

                    raw_data_artifact.download()

                    run.log_artifact(artifact)
            project_folder_path = self.get_project_folder_path()
            return os.path.join(project_folder_path, "artifacts")
        else:
            self.dataset_logger.error(
                "Method using wandb called without initializing wandb flag"
            )
