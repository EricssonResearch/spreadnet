"""Model Trainer.

@Time    : 10/27/2022 6:17 PM
@Author  : Haodong Zhao
"""
import copy
import os
from datetime import datetime
from itertools import islice

import torch
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import webdataset as wds

from spreadnet.datasets.data_utils.decoder import pt_decoder
from spreadnet.datasets.data_utils.draw import plot_training_graph
from spreadnet.pyg_gnn.models import SPCoDeepGCNet, EncodeProcessDecode


class ModelTrainer:
    def __init__(
        self,
        model_name: str,
        model_configs: dict,
        train_configs: dict,
        dataset_path: str,
        dataset_configs: dict,
        model_save_path: str,
    ):

        self.model_name = model_name
        self.model_configs = model_configs
        self.train_configs = train_configs

        self.dataset_path = dataset_path
        self.dataset_configs = dataset_configs

        self.model_save_path = model_save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None

        # plot settings
        self.plots_save_path = os.path.join(self.model_save_path, "training")
        self.steps_curve = []
        self.losses_curve = []
        self.validation_losses_curve = []
        self.accuracies_curve = []
        self.validation_accuracies_curve = []

    def construct_model(self):
        if self.model_name == "MPNN":
            self.model = EncodeProcessDecode(
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
        elif self.model_name == "DEEPCOGCN":
            self.model = SPCoDeepGCNet(
                node_in=self.model_configs["node_in"],
                edge_in=self.model_configs["edge_in"],
                gcn_hidden_channels=self.model_configs["gcn_hidden_channels"],
                gcn_num_layers=self.model_configs["gcn_num_layers"],
                mlp_hidden_channels=self.model_configs["mlp_hidden_channels"],
                mlp_hidden_layers=self.model_configs["mlp_hidden_layers"],
                node_out=self.model_configs["node_out"],
                edge_out=self.model_configs["edge_out"],
            ).to(self.device)
        elif self.model_name == "GAT":
            self.model = None
            pass

    def create_plot(self, plot_name):
        plot_training_graph(
            self.steps_curve,
            self.losses_curve,
            self.validation_losses_curve,
            self.accuracies_curve,
            self.validation_accuracies_curve,
            self.plots_save_path + f"{plot_name}",
        )

    def data_preprocessor(self, data):
        """Preprocess the data from dataset.

        Args:
            data: Pytorch Geometric data

        Returns:
            1. the inputs for the GCN model
            2. the ground-truth labels
        """

        (node_true, edge_true) = data.y
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr

        return (x, edge_index, edge_attr), (node_true, edge_true)

    def construct_dataloader(self):

        dataset = (
            wds.WebDataset("file:" + self.dataset_path + "/processed/all_000000.tar")
            .decode(pt_decoder)
            .to_tuple(
                "pt",
            )
        )

        dataset_size = len(list(dataset))
        train_size = int(self.dataset_configs["train_ratio"] * dataset_size)

        if bool(self.train_configs["shuffle"]):
            dataset.shuffle(dataset_size * 10)

        train_dataset = list(islice(dataset, 0, train_size))
        validation_dataset = list(islice(dataset, train_size, dataset_size + 1))
        train_loader = DataLoader(
            train_dataset, batch_size=self.train_configs["batch_size"]
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.train_configs["batch_size"]
        )
        return train_loader, validation_loader

    def epoch_execute(self, mode, epoch, total_epoch, dataloader, loss_func):
        is_training = mode == "train"
        if is_training:
            pb_str = "Train"
            self.model.train()
        else:
            pb_str = "validation"
            self.model.eval()

        nodes_loss, edges_loss = 0.0, 0.0
        nodes_corrects, edges_corrects = 0, 0
        dataset_nodes_size, dataset_edges_size = 0, 0

        with torch.enable_grad() if is_training else torch.no_grad():
            for batch, (data,) in tqdm(
                enumerate(dataloader),
                unit="batch",
                total=len(list(dataloader)),
                desc=f"[Epoch: {epoch + 1:4} / {total_epoch:4} | {pb_str} ]",
                leave=False,
            ):
                data = data.to(self.device)
                (x, edge_index, edge_attr), (
                    node_true,
                    edge_true,
                ) = self.data_preprocessor(data)

                if is_training:
                    self.optimizer.zero_grad()

                (node_pred, edge_pred) = self.model(x, edge_index, edge_attr)

                # Losses
                (losses, corrects) = loss_func(
                    node_pred, edge_pred, node_true, edge_true
                )
                nodes_loss += losses["nodes"].item() * data.num_graphs
                edges_loss += losses["edges"].item() * data.num_graphs

                if is_training:
                    losses["nodes"].backward(retain_graph=True)
                    losses["edges"].backward()
                    self.optimizer.step()

                # Accuracies
                nodes_corrects += corrects["nodes"]
                edges_corrects += corrects["edges"]
                dataset_nodes_size += data.num_nodes
                dataset_edges_size += data.num_edges

        nodes_loss /= len(dataloader.dataset)
        edges_loss /= len(dataloader.dataset)
        nodes_acc = (nodes_corrects / dataset_nodes_size).cpu().numpy()
        edges_acc = (edges_corrects / dataset_edges_size).cpu().numpy()
        return nodes_loss, edges_loss, nodes_acc, edges_acc

    def execute(self, mode, epoch, total_epoch, dataloader, loss_func):
        """
        Execute training or validating.
        Args:
            epoch: current epoch
            total_epoch: the number of total epoch
            mode: train | validation
            dataloader: dataloader
            loss_func: loss function

        Returns:
            accuracy
        """

        is_training = mode == "train"
        nodes_loss, edges_loss, nodes_acc, edges_acc = self.epoch_execute(
            mode, epoch, total_epoch, dataloader, loss_func
        )

        (self.losses_curve if is_training else self.validation_losses_curve).append(
            {"nodes": nodes_loss, "edges": edges_loss}
        )

        (
            self.accuracies_curve if is_training else self.validation_accuracies_curve
        ).append({"nodes": nodes_acc, "edges": edges_acc})

        return (nodes_acc + edges_acc) / 2

    def train(self, print_info: bool = True):
        print(f"Using {self.device} device...")
        date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)

        dataset_size = self.dataset_configs["dataset_size"]
        plot_name = f"training-size-{dataset_size}-at-{date}.jpg"

        train_loader, validation_loader = self.construct_dataloader()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_configs["adam_lr"],
            weight_decay=self.train_configs["adam_weight_decay"],
        )

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        epochs = self.train_configs["epochs"]
        plot_after_epochs = self.train_configs["plot_after_epochs"]

        if print_info:
            print(self.model)
            print("\n")

        for epoch in range(epochs):
            self.steps_curve.append(epoch + 1)

            _ = self.execute("train", epoch, epochs, train_loader)
            validation_acc = self.execute(
                "validation", epoch, epochs, validation_loader
            )

            if validation_acc > best_acc:
                best_acc = validation_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            if epoch % self.train_configs["weight_save_freq"] == 0:
                weight_name = f"model_weights_ep_{epoch}.pth"
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_path, weight_name),
                )

            if epoch % 10 == 0:
                if print_info:
                    print(
                        "\n  Epoch   "
                        + "Train Loss (Node,Edge)     Validation Loss        "
                        + "Train Acc (Node,Edge)      Validation Acc"
                    )

            if print_info:
                print(f"{epoch + 1:4}/{epochs}".ljust(10), end="")
                print(
                    "{:2.8f}, {:2.8f}  {:2.8f}, {:2.8f}    ".format(
                        self.losses_curve[-1]["nodes"],
                        self.losses_curve[-1]["edges"],
                        self.validation_losses_curve[-1]["nodes"],
                        self.validation_losses_curve[-1]["edges"],
                    ),
                    end="",
                )
                print(
                    "{:2.8f}, {:2.8f}  {:2.8f}, {:2.8f}".format(
                        self.accuracies_curve[-1]["nodes"],
                        self.accuracies_curve[-1]["edges"],
                        self.validation_accuracies_curve[-1]["nodes"],
                        self.validation_accuracies_curve[-1]["edges"],
                    )
                )

            if (epoch + 1) % plot_after_epochs == 0:
                self.create_plot(plot_name)

        weight_name = self.train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(self.model_save_path, weight_name))
        self.create_plot(plot_name)


class WAndBModelTrainer(ModelTrainer):
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        model_name: str,
        model_configs: dict,
        train_configs: dict,
        dataset_path: str,
        dataset_configs: dict,
        model_save_path: str,
    ):
        super(WAndBModelTrainer, self).__init__(
            model_name,
            model_configs,
            train_configs,
            dataset_path,
            dataset_configs,
            model_save_path,
        )
        self.project_name = project_name
        self.experiment_name = f"experiment_{model_name}"
        self.wandb_configs = train_configs

    def train(self):
        wandb.login()
        wandb.init(
            # Set the project where this run will be logged
            project=self.project_name,
            name=f"{self.experiment_name}",
            # Track hyperparameters and run metadata
            config=self.wandb_configs,
        )

        print(f"Using {self.device} device...")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)

        train_loader, validation_loader = self.construct_dataloader()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_configs["adam_lr"],
            weight_decay=self.train_configs["adam_weight_decay"],
        )

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        epochs = self.train_configs["epochs"]

        for epoch in range(epochs):
            (
                train_nodes_loss,
                train_edges_loss,
                train_nodes_acc,
                train_edges_acc,
            ) = self.epoch_execute("train", epoch, epochs, train_loader)

            train_metrics = {
                "train/train_nodes_loss": train_nodes_loss,
                "train/train_edges_loss": train_edges_loss,
                "train/train_nodes_acc": train_nodes_acc,
                "train/train_edges_acc": train_edges_acc,
            }

            (
                validation_nodes_loss,
                validation_edges_loss,
                validation_nodes_acc,
                validation_edges_acc,
            ) = self.epoch_execute("validation", epoch, epochs, validation_loader)

            validation_metrics = {
                "validation/validation_nodes_loss": validation_nodes_loss,
                "validation/validation_edges_loss": validation_edges_loss,
                "validation/validation_nodes_acc": validation_nodes_acc,
                "validation/validation_edges_acc": validation_edges_acc,
            }

            wandb.log({{**train_metrics, **validation_metrics}})

            validation_acc = (validation_nodes_acc + validation_edges_acc) / 2

            if validation_acc > best_acc:
                best_acc = validation_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())

            if epoch % self.train_configs["weight_save_freq"] == 0:
                weight_name = f"model_weights_ep_{epoch}.pth"
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_path, weight_name),
                )

        weight_name = self.train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(self.model_save_path, weight_name))

        # Mark the run as finished
        wandb.finish()
