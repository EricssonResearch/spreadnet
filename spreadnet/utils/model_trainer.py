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
import shutil
from tqdm import tqdm
import webdataset as wds
import logging
import time
from glob import glob
from torch_geometric.data import Batch
import gc

from spreadnet.datasets.data_utils.decoder import pt_decoder
from spreadnet.datasets.data_utils.draw import plot_training_graph
from spreadnet.pyg_gnn.models.deepGCN.sp_deepGCN import SPDeepGCN
from spreadnet.pyg_gnn.utils import hybrid_loss
from spreadnet.pyg_gnn.models import SPCoDeepGCNet, EncodeProcessDecode
from spreadnet.pyg_gnn.models.graph_attention_network.sp_gat import SPGATNet
from spreadnet.pyg_gnn.utils.metrics import (
    get_precise_and_f_score,
    get_corrects_in_path,
    get_correct_predictions,
)


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
        self.in_path_accuracies_curve = []
        self.validation_in_path_accuracies_curve = []
        self.precise_accuracies_curve = []
        self.validation_precise_accuracies_curve = []
        self.score_curve = []
        self.validation_score_curve = []

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
        elif self.model_name == "DeepGCN":
            self.model = SPDeepGCN(
                node_in=self.model_configs["node_in"],
                edge_in=self.model_configs["edge_in"],
                encoder_hidden_channels=self.model_configs["encoder_hidden_channels"],
                encoder_layers=self.model_configs["encoder_layers"],
                gcn_hidden_channels=self.model_configs["gcn_hidden_channels"],
                gcn_layers=self.model_configs["gcn_layers"],
                decoder_hidden_channels=self.model_configs["decoder_hidden_channels"],
                decoder_layers=self.model_configs["decoder_layers"],
            ).to(self.device)

        return self.model

    def create_plot(self, plot_name):
        plot_training_graph(
            self.steps_curve,
            self.losses_curve,
            self.validation_losses_curve,
            self.accuracies_curve,
            self.validation_accuracies_curve,
            self.in_path_accuracies_curve,
            self.validation_in_path_accuracies_curve,
            self.precise_accuracies_curve,
            self.validation_precise_accuracies_curve,
            self.score_curve,
            self.validation_score_curve,
            os.path.join(self.plots_save_path, f"{plot_name}"),
        )

    def data_preprocessor(self, data):
        """Preprocess the data from dataset.

        Args:
            data: PyTorch Geometric data

        Returns:
            1. the inputs for the GCN model
            2. the ground-truth labels
        """

        (node_true, edge_true) = data.y
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr

        return (x, edge_index, edge_attr), (node_true, edge_true)

    def construct_dataset(self):
        """Construct datasets.

        Returns:
            dataset, dataset_size
        """
        tar_length = len(list(glob(self.dataset_path + "/processed/[!test.]*.tar"))) - 1
        last_tar = f"{tar_length:06}"

        dataset = (
            wds.WebDataset(
                "file:"
                + self.dataset_path
                + "/processed/all_{000000.."
                + last_tar
                + "}.tar"
            )
            .decode(pt_decoder)
            .to_tuple(
                "pt",
            )
        )
        dataset_size = sum(1 for _ in dataset)

        if bool(self.train_configs["shuffle"]):
            dataset.shuffle(dataset_size * 10)

        return dataset, dataset_size

    def save_training_state(
        self,
        epoch,
        best_model_wts,
        best_acc,
    ):
        vipac = self.validation_in_path_accuracies_curve
        vpac = self.validation_precise_accuracies_curve
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
                "in_path_accuracies_curve": self.in_path_accuracies_curve,
                "validation_in_path_accuracies_curve": vipac,
                "precise_accuracies_curve": self.precise_accuracies_curve,
                "validation_precise_accuracies_curve": vpac,
                "score_curve": self.score_curve,
                "validation_score_curve": self.validation_score_curve,
            },
            self.checkpoint_path,
        )

    def sub_execute(self, mode, epoch, total_epoch, dataset, dataset_size, loss_func):
        """sub execution: train or validation in one epoch.

        Args:
            mode: train | validation
            epoch: the current epoch
            total_epoch: the total epoch
            dataset: train or validation dataset
            dataset_size: dataset size
            loss_func: loss function

        Returns:
            nodes_loss,
            edges_loss,
            nodes_acc,
            edges_acc,
            node_in_path_acc,
            edge_in_path_acc,
            nodes_precise_acc,
            edges_precise_acc,
            nodes_score_avg,
            edges_score_avg
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
        nodes_in_path_corrects, edges_in_path_corrects = 0, 0
        dataset_nodes_in_path_size, dataset_edges_in_path_size = 0, 0
        nodes_precise_corrects, edges_precise_corrects = 0, 0
        nodes_score, edges_score = 0.0, 0.0

        with torch.enable_grad() if is_training else torch.no_grad():
            batch = []
            batch_size = self.train_configs["batch_size"]
            graph_sizes = []

            for idx, (data,) in tqdm(
                enumerate(dataset),
                unit="batch",
                total=dataset_size,
                desc=f"[Epoch: {epoch:4} / {total_epoch:4} | {pb_str} ]",
                leave=False,
            ):
                batch.append(data)
                graph_sizes.append({"nodes": len(data.y[0]), "edges": len(data.y[1])})

                if (
                    len(batch) < batch_size
                    and len(batch) < dataset_size
                    and idx + 1 != dataset_size
                ):
                    continue

                num_graphs = len(batch)
                data = Batch.from_data_list(batch).to(self.device)
                batch.clear()
                gc.collect()

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
                losses = loss_func(node_pred, edge_pred, node_true, edge_true)
                nodes_loss += float(losses["nodes"].item()) * num_graphs
                edges_loss += float(losses["edges"].item()) * num_graphs

                if is_training:
                    losses["nodes"].backward(retain_graph=True)
                    losses["edges"].backward()
                    self.optimizer.step()

                infers, corrects = get_correct_predictions(
                    node_pred, edge_pred, node_true, edge_true
                )

                node_in_path, edge_in_path = get_corrects_in_path(
                    node_pred, edge_pred, node_true, edge_true
                )
                node_correct_in_path, total_node_in_path = (
                    node_in_path["in_path"].cpu().numpy().item(),
                    node_in_path["total"].cpu().numpy().item(),
                )
                edge_correct_in_path, total_edge_in_path = (
                    edge_in_path["in_path"].cpu().numpy().item(),
                    edge_in_path["total"].cpu().numpy().item(),
                )

                # Accuracies
                nodes_corrects += int(corrects["nodes"].cpu().numpy().item())
                edges_corrects += int(corrects["edges"].cpu().numpy().item())
                dataset_nodes_size += int(data.num_nodes)
                dataset_edges_size += int(data.num_edges)

                nodes_in_path_corrects += node_correct_in_path
                edges_in_path_corrects += edge_correct_in_path
                dataset_nodes_in_path_size += total_node_in_path
                dataset_edges_in_path_size += total_edge_in_path

                (
                    nodes_precise_corrects_batch,
                    edges_precise_corrects_batch,
                    nodes_score_batch,
                    edges_score_batch,
                ) = get_precise_and_f_score(infers, data.y, graph_sizes)

                nodes_precise_corrects += nodes_precise_corrects_batch
                edges_precise_corrects += edges_precise_corrects_batch
                nodes_score += nodes_score_batch
                edges_score += edges_score_batch

                # python auto gc is not good enough to free the memory
                del (
                    node_pred,
                    edge_pred,
                    data,
                    losses,
                    infers,
                    corrects,
                    node_in_path,
                    edge_in_path,
                    node_correct_in_path,
                    total_node_in_path,
                    edge_correct_in_path,
                    total_edge_in_path,
                )
                graph_sizes.clear()
                gc.collect()

        nodes_loss /= dataset_size
        edges_loss /= dataset_size
        nodes_acc = nodes_corrects / dataset_nodes_size
        edges_acc = edges_corrects / dataset_edges_size

        node_in_path_acc = nodes_in_path_corrects / dataset_nodes_in_path_size
        edge_in_path_acc = edges_in_path_corrects / dataset_edges_in_path_size

        nodes_precise_acc = nodes_precise_corrects / dataset_size
        edges_precise_acc = edges_precise_corrects / dataset_size
        nodes_score_avg = nodes_score / dataset_size
        edges_score_avg = edges_score / dataset_size

        return (
            nodes_loss,
            edges_loss,
            nodes_acc,
            edges_acc,
            node_in_path_acc,
            edge_in_path_acc,
            nodes_precise_acc,
            edges_precise_acc,
            nodes_score_avg,
            edges_score_avg,
        )

    def execute(self, epoch, total_epoch, dataset, dataset_size, loss_func):
        """
        Execute training or validating.
        Args:

            epoch: current epoch
            total_epoch: the number of total epoch
            dataset: train and validation dataset
            dataset_size: dataset size
            loss_func: utils function

        Returns:
            accuracy
        """
        train_size = int(self.train_configs["train_ratio"] * dataset_size)
        validation_size = dataset_size - train_size
        train_dataset = islice(dataset, 0, train_size)
        validation_dataset = islice(dataset, train_size, dataset_size + 1)
        (
            train_nodes_loss,
            train_edges_loss,
            train_nodes_acc,
            train_edges_acc,
            train_node_in_path_acc,
            train_edge_in_path_acc,
            train_nodes_precise_acc,
            train_edges_precise_acc,
            train_nodes_score,
            train_edges_score,
        ) = self.sub_execute(
            "train", epoch, total_epoch, train_dataset, train_size, loss_func
        )

        (
            validation_nodes_loss,
            validation_edges_loss,
            validation_nodes_acc,
            validation_edges_acc,
            validation_node_in_path_acc,
            validation_edge_in_path_acc,
            validation_nodes_precise_acc,
            validation_edges_precise_acc,
            validation_nodes_score,
            validation_edges_score,
        ) = self.sub_execute(
            "validation",
            epoch,
            total_epoch,
            validation_dataset,
            validation_size,
            loss_func,
        )

        self.losses_curve.append({"nodes": train_nodes_loss, "edges": train_edges_loss})
        self.validation_losses_curve.append(
            {"nodes": validation_nodes_loss, "edges": validation_edges_loss}
        )

        self.accuracies_curve.append(
            {
                "nodes": train_nodes_acc,
                "edges": train_edges_acc,
            }
        )
        self.validation_accuracies_curve.append(
            {
                "nodes": validation_nodes_acc,
                "edges": validation_edges_acc,
            }
        )

        self.in_path_accuracies_curve.append(
            {"nodes": train_node_in_path_acc, "edges": train_edge_in_path_acc}
        )

        self.validation_in_path_accuracies_curve.append(
            {"nodes": validation_node_in_path_acc, "edges": validation_edge_in_path_acc}
        )

        self.precise_accuracies_curve.append(
            {"nodes": train_nodes_precise_acc, "edges": train_edges_precise_acc}
        )

        self.validation_precise_accuracies_curve.append(
            {
                "nodes": validation_nodes_precise_acc,
                "edges": validation_edges_precise_acc,
            }
        )

        self.score_curve.append(
            {"nodes": train_nodes_score, "edges": train_edges_score}
        )

        self.validation_score_curve.append(
            {"nodes": validation_nodes_score, "edges": validation_edges_score}
        )

        validation_acc = (validation_nodes_score + validation_edges_score) / 2

        return validation_acc

    def train(self, resume):
        train_local_logger = logging.getLogger("train_local_logger")
        train_local_logger.info(f"Using {self.device} device...")
        date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        start_time = time.time()

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if not os.path.exists(self.plots_save_path):
            os.makedirs(self.plots_save_path)

        dataset, dataset_size = self.construct_dataset()

        plot_name = f"training-size-{dataset_size}-at-{date}.jpg"

        self.construct_model()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_configs["adam_lr"],
            weight_decay=self.train_configs["adam_weight_decay"],
        )

        epoch = 1
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = -1.0

        epochs = self.train_configs["epochs"]
        plot_after_epochs = self.train_configs["plot_after_epochs"]

        train_local_logger.info(self.model)

        try:
            exists = os.path.exists(self.checkpoint_path)

            if exists:
                answer = resume or input(
                    "Previous training state found. Enter y/Y to continue training: "
                )

                if resume or answer.capitalize() == "Y":
                    checkpoint = torch.load(self.checkpoint_path)
                    train_local_logger.info(
                        f'Resume training from {checkpoint["epoch"]} epoch'
                    )
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
                    self.in_path_accuracies_curve = checkpoint[
                        "in_path_accuracies_curve"
                    ]
                    self.validation_in_path_accuracies_curve = checkpoint[
                        "validation_in_path_accuracies_curve"
                    ]
                    self.precise_accuracies_curve = checkpoint[
                        "precise_accuracies_curve"
                    ]
                    self.validation_precise_accuracies_curve = checkpoint[
                        "validation_precise_accuracies_curve"
                    ]
                    self.score_curve = checkpoint["score_curve"]
                    self.validation_score_curve = checkpoint["validation_score_curve"]
        except Exception as exception:
            train_local_logger.exception(exception)

        train_local_logger.info("Start training...")
        try:
            for epoch in range(epoch, epochs + 1):
                self.steps_curve.append(epoch)

                validation_acc = self.execute(
                    epoch, epochs, dataset, dataset_size, hybrid_loss
                )

                if validation_acc > best_acc:
                    best_acc = validation_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    weight_name = self.train_configs["best_weight_name"]
                    torch.save(
                        best_model_wts, os.path.join(self.model_save_path, weight_name)
                    )

                if epoch % self.train_configs["weight_save_freq"] == 0:
                    weight_name = f"model_weights_ep_{epoch}.pth"
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.model_save_path, weight_name),
                    )
                    self.save_training_state(
                        epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
                    )
                    train_local_logger.info("Saving state...")

                if epoch % 10 == 1:
                    train_local_logger.info(
                        f'{"Epoch":^7s}|{"Train Loss (Node,Edge)":^22s}|'
                        f'{"Validation Loss":^22s}|'
                        f"Train Acc (Node,Edge,NodeInPath,EdgeInPath,"
                        f"NodePrecise,EdgePrecise)|"
                        f'{"Validation Acc":^67s}|'
                        f'{"Train F-Score":^22s}|'
                        f'{"Validation F-Score":^22s} \n {"=" * 266}'
                    )
                lcn = self.losses_curve[-1]["nodes"]
                lce = self.losses_curve[-1]["edges"]
                vlcn = self.validation_losses_curve[-1]["nodes"]
                vlce = self.validation_losses_curve[-1]["edges"]
                accn = self.accuracies_curve[-1]["nodes"]
                ace = self.accuracies_curve[-1]["edges"]
                ipacn = self.in_path_accuracies_curve[-1]["nodes"]
                ipace = self.in_path_accuracies_curve[-1]["edges"]
                vacn = self.validation_accuracies_curve[-1]["nodes"]
                vace = self.validation_accuracies_curve[-1]["edges"]
                vipacn = self.validation_in_path_accuracies_curve[-1]["nodes"]
                vipace = self.validation_in_path_accuracies_curve[-1]["edges"]
                pacn = self.precise_accuracies_curve[-1]["nodes"]
                pace = self.precise_accuracies_curve[-1]["edges"]
                vpacn = self.validation_precise_accuracies_curve[-1]["nodes"]
                vpace = self.validation_precise_accuracies_curve[-1]["edges"]
                scn = self.score_curve[-1]["nodes"]
                sce = self.score_curve[-1]["edges"]
                vscn = self.validation_score_curve[-1]["nodes"]
                vsce = self.validation_score_curve[-1]["edges"]

                train_local_logger.info(
                    f"{epoch:3}/{epochs}|"
                    f"{lcn:2.8f},{lce:2.8f} |"
                    f"{vlcn:2.8f},{vlce:2.8f} |"
                    f" {accn:2.8f},{ace:2.8f} {ipacn:2.8f},{ipace:2.8f} "
                    f"{pacn:2.8f},{pace:2.8f} |"
                    f" {vacn:2.8f},{vace:2.8f} {vipacn:2.8f},{vipace:2.8f} "
                    f"{vpacn:2.8f},{vpace:2.8f} |"
                    f"{scn:2.8f},{sce:2.8f} |"
                    f"{vscn:2.8f},{vsce:2.8f}"
                )
                if epoch % plot_after_epochs == 0:
                    self.create_plot(plot_name)

            train_local_logger.info("Finalizing training plot...")
            self.create_plot(plot_name)
            self.save_training_state(
                epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
            )
        except Exception as exception:
            train_local_logger.exception(exception)

        train_local_logger.info(
            f'Time elapsed = {(time.time() - start_time)} sec \n {"=":176s}'
        )
        logging.shutdown()


class WAndBModelTrainer(ModelTrainer):
    def __init__(
        self,
        entity_name: str,
        project_name: str,
        model_configs: dict,
        train_configs: dict,
        dataset_path: str,
        dataset_configs: dict,
        model_save_path: str,
    ):
        super(WAndBModelTrainer, self).__init__(
            model_configs, train_configs, dataset_path, dataset_configs, model_save_path
        )
        self.checkpoint_path = os.path.join(model_save_path, "wandb_train_state.pth")
        self.wandb_id = 0
        self.entity_name = entity_name
        self.project_name = project_name
        self.wandb_configs = train_configs
        self.wandb_checkpoint_name = "wandb_train_state.pth"

        self.loss_data = []
        self.acc_data = []

        self.epoch_lst = []
        self.train_nodes_loss = []
        self.train_edges_loss = []
        self.train_nodes_acc = []
        self.train_edges_acc = []
        self.train_nodes_in_path_acc = []
        self.train_edges_in_path_acc = []
        self.train_nodes_precise_acc = []
        self.train_edges_precise_acc = []
        self.train_nodes_score = []
        self.train_edges_score = []

        self.validation_nodes_loss = []
        self.validation_edges_loss = []
        self.validation_nodes_acc = []
        self.validation_edges_acc = []
        self.validation_nodes_in_path_acc = []
        self.validation_edges_in_path_acc = []
        self.validation_nodes_precise_acc = []
        self.validation_edges_precise_acc = []
        self.validation_nodes_score = []
        self.validation_edges_score = []

    def save_training_state(
        self,
        epoch,
        best_model_wts,
        best_acc,
    ):
        torch.save(
            {
                "wandb_id": self.wandb_id,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_model_state_dict": best_model_wts,
                "best_acc": best_acc,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch_lst": self.epoch_lst,
                "loss_data": self.loss_data,
                "acc_data": self.acc_data,
                "train_nodes_loss": self.train_nodes_loss,
                "train_edges_loss": self.train_edges_loss,
                "train_nodes_acc": self.train_nodes_acc,
                "train_edges_acc": self.train_edges_acc,
                "validation_nodes_loss": self.validation_nodes_loss,
                "validation_edges_loss": self.validation_edges_loss,
                "validation_nodes_acc": self.validation_nodes_acc,
                "validation_edges_acc": self.validation_edges_acc,
                "train_nodes_in_path_acc": self.train_nodes_in_path_acc,
                "train_edges_in_path_acc": self.train_edges_in_path_acc,
                "validation_nodes_in_path_acc": self.validation_nodes_in_path_acc,
                "validation_edges_in_path_acc": self.validation_edges_in_path_acc,
                "train_nodes_precise_acc": self.train_nodes_precise_acc,
                "train_edges_precise_acc": self.train_edges_precise_acc,
                "validation_nodes_precise_acc": self.validation_nodes_precise_acc,
                "validation_edges_precise_acc": self.validation_edges_precise_acc,
                "train_nodes_score": self.train_nodes_score,
                "train_edges_score": self.train_edges_score,
                "validation_nodes_score": self.validation_nodes_score,
                "validation_edges_score": self.validation_edges_score,
            },
            self.checkpoint_path,
        )

    def execute(self, epoch, total_epoch, dataset, dataset_size, loss_func):
        """
        Execute training or validating.
        Args:

            epoch: current epoch
            total_epoch: the number of total epoch
            dataset: train and validation dataset
            dataset_size: dataset size
            loss_func: utils function

        Returns:
            accuracy
        """
        train_size = int(self.train_configs["train_ratio"] * dataset_size)
        validation_size = dataset_size - train_size
        train_dataset = islice(dataset, 0, train_size)
        validation_dataset = islice(dataset, train_size, dataset_size + 1)
        (
            train_nodes_loss,
            train_edges_loss,
            train_nodes_acc,
            train_edges_acc,
            train_node_in_path_acc,
            train_edge_in_path_acc,
            train_node_precise_acc,
            train_edge_precise_acc,
            train_node_score,
            train_edge_score,
        ) = self.sub_execute(
            "train", epoch, total_epoch, train_dataset, train_size, loss_func
        )

        (
            validation_nodes_loss,
            validation_edges_loss,
            validation_nodes_acc,
            validation_edges_acc,
            validation_node_in_path_acc,
            validation_edge_in_path_acc,
            validation_node_precise_acc,
            validation_edge_precise_acc,
            validation_node_score,
            validation_edge_score,
        ) = self.sub_execute(
            "validation",
            epoch,
            total_epoch,
            validation_dataset,
            validation_size,
            loss_func,
        )

        # simple log
        train_metrics = {
            "epoch": epoch,
            "train_nodes_loss": train_nodes_loss,
            "train_edges_loss": train_edges_loss,
            "train_nodes_acc": train_nodes_acc,
            "train_edges_acc": train_edges_acc,
            "train_nodes_in_path_acc": train_node_in_path_acc,
            "train_edges_in_path_acc": train_edge_in_path_acc,
            "train_nodes_precise_acc": train_node_precise_acc,
            "train_edges_precise_acc": train_edge_precise_acc,
            "train_nodes_score": train_node_score,
            "train_edges_score": train_edge_score,
        }

        validation_metrics = {
            "epoch": epoch,
            "validation_nodes_loss": validation_nodes_loss,
            "validation_edges_loss": validation_edges_loss,
            "validation_nodes_acc": validation_nodes_acc,
            "validation_edges_acc": validation_edges_acc,
            "validation_nodes_in_path_acc": validation_node_in_path_acc,
            "validation_edges_in_path_acc": validation_edge_in_path_acc,
            "validation_nodes_precise_acc": validation_node_precise_acc,
            "validation_edges_precise_acc": validation_edge_precise_acc,
            "validation_nodes_score": validation_node_score,
            "validation_edges_score": validation_edge_score,
        }

        wandb.log({**train_metrics, **validation_metrics})

        # line plot
        self.train_nodes_loss.append(train_nodes_loss)
        self.train_edges_loss.append(train_edges_loss)
        self.train_nodes_acc.append(train_nodes_acc)
        self.train_edges_acc.append(train_edges_acc)
        self.train_nodes_in_path_acc.append(train_node_in_path_acc)
        self.train_edges_in_path_acc.append(train_edge_in_path_acc)
        self.train_nodes_precise_acc.append(train_node_precise_acc)
        self.train_edges_precise_acc.append(train_edge_precise_acc)
        self.train_nodes_score.append(train_node_score)
        self.train_edges_score.append(train_edge_score)

        self.validation_nodes_loss.append(validation_nodes_loss)
        self.validation_edges_loss.append(validation_edges_loss)
        self.validation_nodes_acc.append(validation_nodes_acc)
        self.validation_edges_acc.append(validation_edges_acc)
        self.validation_nodes_in_path_acc.append(validation_node_in_path_acc)
        self.validation_edges_in_path_acc.append(validation_edge_in_path_acc)
        self.validation_nodes_precise_acc.append(validation_node_precise_acc)
        self.validation_edges_precise_acc.append(validation_edge_precise_acc)
        self.validation_nodes_score.append(validation_node_score)
        self.validation_edges_score.append(validation_edge_score)

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

        wandb.log(
            {
                "train_valid_in_path_acc": wandb.plot.line_series(
                    xs=self.epoch_lst,
                    ys=[
                        self.train_nodes_in_path_acc,
                        self.train_edges_in_path_acc,
                        self.validation_nodes_in_path_acc,
                        self.validation_edges_in_path_acc,
                    ],
                    keys=[
                        "train_nodes_in_path_acc",
                        "train_edges_in_path_acc",
                        "validation_nodes_in_path_acc",
                        "validation_edges_in_path_acc",
                    ],
                    title="In-path Accuracy",
                    xname="epoch",
                )
            }
        )

        wandb.log(
            {
                "train_valid_precise_acc": wandb.plot.line_series(
                    xs=self.epoch_lst,
                    ys=[
                        self.train_nodes_precise_acc,
                        self.train_edges_precise_acc,
                        self.validation_nodes_precise_acc,
                        self.validation_edges_precise_acc,
                    ],
                    keys=[
                        "train_nodes_precise_acc",
                        "train_edges_precise_acc",
                        "validation_nodes_precise_acc",
                        "validation_edges_precise_acc",
                    ],
                    title="Precise Accuracy",
                    xname="epoch",
                )
            }
        )

        wandb.log(
            {
                "train_valid_score": wandb.plot.line_series(
                    xs=self.epoch_lst,
                    ys=[
                        self.train_nodes_score,
                        self.train_edges_score,
                        self.validation_nodes_score,
                        self.validation_edges_score,
                    ],
                    keys=[
                        "train_nodes_score",
                        "train_edges_score",
                        "validation_nodes_score",
                        "validation_edges_score",
                    ],
                    title="F-Score",
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
                "utils": wandb.Table(
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
            train_node_in_path_acc,
            train_edge_in_path_acc,
            train_node_precise_acc,
            train_edge_precise_acc,
            train_node_score,
            train_edge_score,
            validation_nodes_acc,
            validation_edges_acc,
            validation_node_in_path_acc,
            validation_edge_in_path_acc,
            validation_node_precise_acc,
            validation_edge_precise_acc,
            validation_node_score,
            validation_edge_score,
        ]
        self.acc_data.append(cur_acc_data)
        wandb.log(
            {
                "accuracy": wandb.Table(
                    data=self.acc_data,
                    columns=[
                        "train_nodes_acc",
                        "train_edges_acc",
                        "train_nodes_in_path_acc",
                        "train_edges_in_path_acc",
                        "train_nodes_precise_acc",
                        "train_edges_precise_acc",
                        "train_nodes_score",
                        "train_edges_score",
                        "validation_nodes_acc",
                        "validation_edges_acc",
                        "validation_nodes_in_path_acc",
                        "validation_edges_in_path_acc",
                        "validation_nodes_precise_acc",
                        "validation_edges_precise_acc",
                        "validation_nodes_score",
                        "validation_edges_score",
                    ],
                )
            }
        )

        validation_acc = (validation_node_score + validation_edge_score) / 2

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

    def train(self, resume):
        date = datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
        experiment_name = f"train_{self.model_name}_{date}"

        wandb.login()

        train_console_logger = logging.getLogger("train_console_logger")

        start_time = time.time()

        # TODO: specify artifact_name.
        #       Maybe we can extract some features from dataset
        #       and specify artifact_name like: MPNN_nodes_8-17
        artifact_name = f"{self.model_name}"

        self.construct_model()

        # TODO:
        #  We can override `construct_dataset()` after setting the dataset in wandb
        dataset, dataset_size = self.construct_dataset()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_configs["adam_lr"],
            weight_decay=self.train_configs["adam_weight_decay"],
        )

        epoch = 1
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = -1.0

        epochs = self.train_configs["epochs"]

        exists = os.path.exists(self.checkpoint_path)

        if exists:
            answer = resume or input(
                "Previous wandb training state found. Enter y/Y to continue training: "
            )

            if resume or answer.capitalize() == "Y":
                checkpoint = torch.load(self.checkpoint_path)
                print("Resume training...")

                self.wandb_id = checkpoint["wandb_id"]

                run = wandb.init(
                    entity=self.entity_name,
                    project=self.project_name,
                    id=self.wandb_id,
                    resume=True,
                )

                epoch = checkpoint["epoch"] + 1
                self.model.load_state_dict(checkpoint["model_state_dict"])
                best_model_wts = checkpoint["best_model_state_dict"]
                best_acc = checkpoint["best_acc"]
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.loss_data = checkpoint["loss_data"]
                self.acc_data = checkpoint["acc_data"]
                self.epoch_lst = checkpoint["epoch_lst"]
                self.train_nodes_loss = checkpoint["train_nodes_loss"]
                self.train_edges_loss = checkpoint["train_edges_loss"]
                self.train_nodes_acc = checkpoint["train_nodes_acc"]
                self.train_edges_acc = checkpoint["train_edges_acc"]

                self.validation_nodes_loss = checkpoint["validation_nodes_loss"]
                self.validation_edges_loss = checkpoint["validation_edges_loss"]
                self.validation_nodes_acc = checkpoint["validation_nodes_acc"]
                self.validation_edges_acc = checkpoint["validation_edges_acc"]

                self.train_nodes_in_path_acc = checkpoint["train_nodes_in_path_acc"]
                self.train_edges_in_path_acc = checkpoint["train_edges_in_path_acc"]
                self.validation_nodes_in_path_acc = checkpoint[
                    "validation_nodes_in_path_acc"
                ]
                self.validation_edges_in_path_acc = checkpoint[
                    "validation_edges_in_path_acc"
                ]
                self.train_nodes_precise_acc = checkpoint["train_nodes_precise_acc"]
                self.train_edges_precise_acc = checkpoint["train_edges_precise_acc"]
                self.validation_nodes_precise_acc = checkpoint[
                    "validation_nodes_precise_acc"
                ]
                self.validation_edges_precise_acc = checkpoint[
                    "validation_edges_precise_acc"
                ]
                self.train_nodes_score = checkpoint["train_nodes_score"]
                self.train_edges_score = checkpoint["train_edges_score"]
                self.validation_nodes_score = checkpoint["validation_nodes_score"]
                self.validation_edges_score = checkpoint["validation_edges_score"]
            else:
                self.wandb_id = wandb.util.generate_id()
                run = wandb.init(
                    # Set the project where this run will be logged
                    entity=self.entity_name,
                    project=self.project_name,
                    name=f"{experiment_name}",
                    id=self.wandb_id,
                    # Track hyperparameters and run metadata
                    config=self.wandb_configs,
                    job_type="training",
                )
        else:
            self.wandb_id = wandb.util.generate_id()
            run = wandb.init(
                # Set the project where this run will be logged
                entity=self.entity_name,
                project=self.project_name,
                name=f"{experiment_name}",
                id=self.wandb_id,
                # Track hyperparameters and run metadata
                config=self.wandb_configs,
                job_type="training",
            )

        train_console_logger.info(f"Using {self.device} device...")
        train_console_logger.info("Start training")

        for epoch in range(epoch, epochs + 1):
            self.epoch_lst.append(epoch)

            validation_acc = self.execute(
                epoch, epochs, dataset, dataset_size, hybrid_loss
            )

            if validation_acc > best_acc:
                best_acc = validation_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                weight_name = self.train_configs["best_weight_name"]
                torch.save(
                    best_model_wts, os.path.join(self.model_save_path, weight_name)
                )
                self.wandb_model_log(run, artifact_name, weight_name)

            if epoch % self.train_configs["weight_save_freq"] == 0:
                weight_name = f"model_weights_ep_{epoch}.pth"
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_path, weight_name),
                )
                # log weights
                self.wandb_model_log(run, artifact_name, weight_name)

            # log states each epoch
            self.save_training_state(
                epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
            )
            # wandb.save(self.wandb_checkpoint_path)
            #   symlink error on Windows, copy manually
            shutil.copy(
                self.checkpoint_path,
                os.path.join(wandb.run.dir, self.wandb_checkpoint_name),
            )

        self.save_training_state(
            epoch=epoch, best_model_wts=best_model_wts, best_acc=best_acc
        )
        # wandb.save(self.wandb_checkpoint_path)
        #   symlink error on Windows, copy manually
        shutil.copy(
            self.checkpoint_path,
            os.path.join(wandb.run.dir, self.wandb_checkpoint_name),
        )
        train_console_logger.info(
            f'Time elapsed = {(time.time() - start_time)} sec \n {"=":176s}'
        )
