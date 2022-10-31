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
from spreadnet.pyg_gnn.loss import hybrid_loss
from spreadnet.pyg_gnn.models import SPCoDeepGCNet, EncodeProcessDecode
from spreadnet.pyg_gnn.models.graph_attention_network.sp_gat import SPGATNet


class ModelTrainer:
    def __init__(
        self,
        model_configs: dict,
        train_configs: dict,
        dataset_path: str,
        dataset_configs: dict,
        model_save_path: str,
    ):
        self.model_configs = model_configs
        self.model_name = self.model_configs["model_name"]
        self.train_configs = train_configs

        self.dataset_path = dataset_path
        self.dataset_configs = dataset_configs

        self.model_save_path = model_save_path
        self.checkpoint_path = os.path.join(model_save_path, "train_state.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None

        # plot settings
        self.plots_save_path = os.path.join(model_save_path, "../", "trainings")
        self.steps_curve = []
        self.losses_curve = []
        self.validation_losses_curve = []
        self.accuracies_curve = []
        self.validation_accuracies_curve = []

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)

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
        elif self.model_name == "DeepCoGCN":
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
            self.model = SPGATNet(
                num_hidden_layers=self.model_configs["num_hidden_layers"],
                in_channels=self.model_configs["in_channels"],
                hidden_channels=self.model_configs["hidden_channels"],
                out_channels=self.model_configs["out_channels"],
                heads=self.model_configs["heads"],
                # dropout=model_configs[""],
                add_self_loops=self.model_configs["add_self_loops"],
                bias=self.model_configs["bias"],
                edge_hidden_channels=self.model_configs["edge_hidden_channels"],
                edge_out_channels=self.model_configs["edge_out_channels"],
                edge_num_layers=self.model_configs["edge_num_layers"],
                edge_bias=self.model_configs["edge_bias"],
                encode_node_in=self.model_configs["encode_node_in"],
                encode_edge_in=self.model_configs["encode_edge_in"],
                encode_node_out=self.model_configs["encode_node_out"],
                encode_edge_out=self.model_configs["encode_edge_out"],
            ).to(self.device)

        return self.model

    def create_plot(self, plot_name):
        plot_training_graph(
            self.steps_curve,
            self.losses_curve,
            self.validation_losses_curve,
            self.accuracies_curve,
            self.validation_accuracies_curve,
            os.path.join(self.plots_save_path, f"{plot_name}"),
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
        """Construct dataloaders: train loader and validation loader.

        Returns:
            train_loader, validation_loader
        """
        dataset = (
            wds.WebDataset("file:" + self.dataset_path + "/processed/all_000000.tar")
            .decode(pt_decoder)
            .to_tuple(
                "pt",
            )
        )

        dataset_size = len(list(dataset))
        train_size = int(self.train_configs["train_ratio"] * dataset_size)

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

    def save_training_state(
        self,
        epoch,
        best_model_wts,
        best_acc,
    ):
        print("Saving state...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_model_state_dict": best_model_wts,
                "best_acc": best_acc,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_curve": self.steps_curve,
                "losses_curve": self.losses_curve,
                "validation_losses_curve": self.validation_losses_curve,
                "accuracies_curve": self.accuracies_curve,
                "validation_accuracies_curve": self.validation_accuracies_curve,
            },
            self.checkpoint_path,
        )

    def sub_execute(self, mode, epoch, total_epoch, dataloader, loss_func):
        """sub execution: train or validation in one epoch.

        Args:
            mode: train | validation
            epoch: the current epoch
            total_epoch: the total epoch
            dataloader: dataloader
            loss_func: loass function

        Returns:
            nodes_loss, edges_loss, nodes_acc, edges_acc
        """
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
                desc=f"[Epoch: {epoch:4} / {total_epoch:4} | {pb_str} ]",
                leave=False,
            ):
                data = data.to(self.device)
                (x, edge_index, edge_attr), (
                    node_true,
                    edge_true,
                ) = self.data_preprocessor(data)

                if is_training:
                    self.optimizer.zero_grad()

                if self.model_name == "GAT":
                    (node_pred, edge_pred) = self.model(
                        data.x,
                        data.edge_index,
                        data.edge_attr,
                        return_attention_weights=self.model_configs[
                            "return_attention_weights"
                        ],
                    )
                else:
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
        nodes_acc = (nodes_corrects / dataset_nodes_size).cpu().numpy().item()
        edges_acc = (edges_corrects / dataset_edges_size).cpu().numpy().item()
        return nodes_loss, edges_loss, nodes_acc, edges_acc

    def execute(self, epoch, total_epoch, train_loader, valid_loader, loss_func):
        """
        Execute training or validating.
        Args:

            epoch: current epoch
            total_epoch: the number of total epoch
            valid_loader:
            train_loader:
            loss_func: loss function

        Returns:
            accuracy
        """
        (
            train_nodes_loss,
            train_edges_loss,
            train_nodes_acc,
            train_edges_acc,
        ) = self.sub_execute("train", epoch, total_epoch, train_loader, loss_func)

        (
            validation_nodes_loss,
            validation_edges_loss,
            validation_nodes_acc,
            validation_edges_acc,
        ) = self.sub_execute("validation", epoch, total_epoch, valid_loader, loss_func)

        self.losses_curve.append({"nodes": train_nodes_loss, "edges": train_edges_loss})
        self.validation_losses_curve.append(
            {"nodes": validation_nodes_loss, "edges": validation_edges_loss}
        )

        self.accuracies_curve.append(
            {"nodes": train_nodes_acc, "edges": train_edges_acc}
        )
        self.validation_accuracies_curve.append(
            {"nodes": validation_nodes_acc, "edges": validation_edges_acc}
        )

        validation_acc = (validation_nodes_acc + validation_edges_acc) / 2

        return validation_acc

    def train(self):
        print(f"Using {self.device} device...")
        date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)

        dataset_size = self.dataset_configs["dataset_size"]
        plot_name = f"training-size-{dataset_size}-at-{date}.jpg"

        train_loader, validation_loader = self.construct_dataloader()

        self.construct_model()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_configs["adam_lr"],
            weight_decay=self.train_configs["adam_weight_decay"],
        )

        epoch = 1
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        epochs = self.train_configs["epochs"]
        plot_after_epochs = self.train_configs["plot_after_epochs"]

        print(self.model)
        print("\n")

        try:
            exists = os.path.exists(self.checkpoint_path)

            if exists:
                answer = input(
                    "Previous training state found. Enter y/Y to continue training: "
                )

                if answer.capitalize() == "Y":
                    checkpoint = torch.load(self.checkpoint_path)
                    print("Resume training...")
                    epoch = checkpoint["epoch"] + 1
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    best_model_wts = checkpoint["best_model_state_dict"]
                    best_acc = checkpoint["best_acc"]
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self.steps_curve = checkpoint["steps_curve"]
                    self.losses_curve = checkpoint["losses_curve"]
                    self.validation_losses_curve = checkpoint["validation_losses_curve"]
                    self.accuracies_curve = checkpoint["accuracies_curve"]
                    self.validation_accuracies_curve = checkpoint[
                        "validation_accuracies_curve"
                    ]
        except Exception as exception:
            print(exception)

        print("Start training...")

        for epoch in range(epoch, epochs + 1):
            self.steps_curve.append(epoch)

            validation_acc = self.execute(
                epoch, epochs, train_loader, validation_loader, hybrid_loss
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
                self.save_training_state(
                    epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
                )

            if epoch % 10 == 0:
                print(
                    "\n  Epoch   "
                    + "Train Loss (Node,Edge)     Validation Loss        "
                    + "Train Acc (Node,Edge)      Validation Acc"
                )

            print(f"{epoch:4}/{epochs}".ljust(10), end="")
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

            if epoch % plot_after_epochs == 0:
                self.create_plot(plot_name)

        weight_name = self.train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(self.model_save_path, weight_name))
        print("Finalizing training plot...")
        self.create_plot(plot_name)
        self.save_training_state(
            epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
        )


class WAndBModelTrainer(ModelTrainer):
    def __init__(
        self,
        project_name: str,
        model_configs: dict,
        train_configs: dict,
        dataset_path: str,
        dataset_configs: dict,
        model_save_path: str,
    ):
        super(WAndBModelTrainer, self).__init__(
            model_configs,
            train_configs,
            dataset_path,
            dataset_configs,
            model_save_path,
        )

        self.project_name = project_name
        self.wandb_configs = train_configs

        self.loss_data = []
        self.acc_data = []

        self.epoch_lst = []
        self.train_nodes_loss = []
        self.train_edges_loss = []
        self.train_nodes_acc = []
        self.train_edges_acc = []

        self.validation_nodes_loss = []
        self.validation_edges_loss = []
        self.validation_nodes_acc = []
        self.validation_edges_acc = []

    def execute(self, epoch, total_epoch, train_loader, valid_loader, loss_func):
        """
        Execute training or validating.
        Args:

            epoch: current epoch
            total_epoch: the number of total epoch
            valid_loader:
            train_loader:
            loss_func: loss function

        Returns:
            accuracy
        """
        (
            train_nodes_loss,
            train_edges_loss,
            train_nodes_acc,
            train_edges_acc,
        ) = self.sub_execute("train", epoch, total_epoch, train_loader, loss_func)

        (
            validation_nodes_loss,
            validation_edges_loss,
            validation_nodes_acc,
            validation_edges_acc,
        ) = self.sub_execute("validation", epoch, total_epoch, valid_loader, loss_func)

        # simple log
        train_metrics = {
            "Train/epoch": epoch + 1,
            "Train/train_nodes_loss": train_nodes_loss,
            "Train/train_edges_loss": train_edges_loss,
            "Train/train_nodes_acc": train_nodes_acc,
            "Train/train_edges_acc": train_edges_acc,
        }

        validation_metrics = {
            "Train/epoch": epoch + 1,
            "Validation/validation_nodes_loss": validation_nodes_loss,
            "Validation/validation_edges_loss": validation_edges_loss,
            "Validation/validation_nodes_acc": validation_nodes_acc,
            "Validation/validation_edges_acc": validation_edges_acc,
        }

        wandb.log({**train_metrics, **validation_metrics})

        # line plot
        self.train_nodes_loss.append(train_nodes_loss)
        self.train_edges_loss.append(train_edges_loss)
        self.train_nodes_acc.append(train_nodes_acc)
        self.train_edges_acc.append(train_edges_acc)

        self.validation_nodes_loss.append(validation_nodes_loss)
        self.validation_edges_loss.append(validation_edges_loss)
        self.validation_nodes_acc.append(validation_nodes_acc)
        self.validation_edges_acc.append(validation_edges_acc)

        wandb.log(
            {
                "train_valid_loss": wandb.plot.line_series(
                    xs=self.epoch_lst,
                    ys=[
                        self.train_nodes_loss,
                        self.train_edges_loss,
                        self.validation_nodes_loss,
                        self.validation_edges_loss,
                    ],
                    keys=[
                        "train_nodes_loss",
                        "train_edges_loss",
                        "validation_nodes_loss",
                        "validation_edges_loss",
                    ],
                    title="Loss",
                    xname="epoch",
                )
            }
        )

        wandb.log(
            {
                "train_valid_acc": wandb.plot.line_series(
                    xs=self.epoch_lst,
                    ys=[
                        self.train_nodes_acc,
                        self.train_edges_acc,
                        self.validation_nodes_acc,
                        self.validation_edges_acc,
                    ],
                    keys=[
                        "train_nodes_acc",
                        "train_edges_acc",
                        "validation_nodes_acc",
                        "validation_edges_acc",
                    ],
                    title="Accuracy",
                    xname="epoch",
                )
            }
        )

        # table log
        cur_loss_data = [
            train_nodes_loss,
            train_edges_loss,
            validation_nodes_loss,
            validation_edges_loss,
        ]
        self.loss_data.append(cur_loss_data)
        wandb.log(
            {
                "loss": wandb.Table(
                    data=self.loss_data,
                    columns=[
                        "train_nodes_loss",
                        "train_edges_loss",
                        "validation_nodes_loss",
                        "validation_edges_loss",
                    ],
                )
            }
        )

        cur_acc_data = [
            train_nodes_acc,
            train_edges_acc,
            validation_nodes_acc,
            validation_edges_acc,
        ]
        self.acc_data.append(cur_acc_data)
        wandb.log(
            {
                "accuracy": wandb.Table(
                    data=self.acc_data,
                    columns=[
                        "train_nodes_acc",
                        "train_edges_acc",
                        "validation_nodes_acc",
                        "validation_edges_acc",
                    ],
                )
            }
        )

        validation_acc = (validation_nodes_acc + validation_edges_acc) / 2

        return validation_acc

    def wandb_model_log(self, run, artifact_name, weight_name):
        model_artifact = wandb.Artifact(
            artifact_name,
            type="model",
            description=self.model_name,
            metadata={
                "train_configs": self.train_configs,
                "model_configs": self.model_configs,
            },
        )

        torch.save(
            self.model.state_dict(), os.path.join(self.model_save_path, weight_name)
        )
        model_artifact.add_file(os.path.join(self.model_save_path, weight_name))
        run.log_artifact(model_artifact)

    def train(self):
        date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
        experiment_name = f"train_{self.model_name}_{date}"
        wandb.login()

        # TODO: specify artifact_name.
        #       Maybe we can extract some features from dataset
        #       and specify artifact_name like: MPNN_nodes_8-17
        artifact_name = f"{self.model_name}"

        with wandb.init(
            # Set the project where this run will be logged
            project=self.project_name,
            name=f"{experiment_name}",
            # Track hyperparameters and run metadata
            config=self.wandb_configs,
            job_type="training",
        ) as run:

            print(f"Using {self.device} device...")

            # TODO:
            #  We can override `construct_model()` after setting the dataset in wandb
            self.construct_model()

            train_loader, validation_loader = self.construct_dataloader()

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_configs["adam_lr"],
                weight_decay=self.train_configs["adam_weight_decay"],
            )

            epoch = 1
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_acc = 0.0

            epochs = self.train_configs["epochs"]

            for epoch in range(epoch, epochs + 1):
                self.epoch_lst.append(epoch)

                validation_acc = self.execute(
                    epoch, epochs, train_loader, validation_loader, hybrid_loss
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

                    self.wandb_model_log(run, artifact_name, weight_name)

            weight_name = self.train_configs["best_weight_name"]
            torch.save(best_model_wts, os.path.join(self.model_save_path, weight_name))
            self.wandb_model_log(run, artifact_name, weight_name)
