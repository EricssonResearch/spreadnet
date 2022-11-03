"""Train the GCN model.

Usage:
    python train.py [--config config_file_path]

    @Time    : 10/20/2022 8:54 PM
    @Author  : Haodong Zhao
"""
import copy
import os
import argparse
from typing import Optional
import torch
import webdataset as wds
from torch_geometric.loader import DataLoader
from datetime import datetime
from itertools import islice

from tqdm import tqdm

from spreadnet.pyg_gnn.loss import hybrid_loss
from spreadnet.pyg_gnn.models import SPCoDeepGCNet
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder
from spreadnet.datasets.data_utils.draw import plot_training_graph

default_yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "../dataset_configs.yaml"
)

parser = argparse.ArgumentParser(description="Train the model.")
parser.add_argument(
    "--config", default=default_yaml_path, help="Specify the path of the config file. "
)
parser.add_argument(
    "--dataset-config",
    default=default_dataset_yaml_path,
    help="Specify the path of the dataset config file. ",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_path = args.config
dataset_yaml_path = args.dataset_config
configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)
train_configs = configs.train
epochs = train_configs["epochs"]
plot_after_epochs = train_configs["plot_after_epochs"]
data_configs = dataset_configs.data
dataset_path = os.path.join(
    os.path.dirname(__file__), "..", data_configs["dataset_path"]
).replace("\\", "/")

weight_base_path = train_configs["weight_base_path"]
model_configs = configs.model
log_folder = "trainings"
print("Prepare to train the GCN model...")

weight_base_path = os.path.join(os.path.dirname(__file__), weight_base_path)

trainings_plots_path = os.path.join(os.path.dirname(__file__), log_folder)
train_ratio = train_configs["train_ratio"]

if not os.path.exists(weight_base_path):
    os.makedirs(weight_base_path)

if not os.path.exists(trainings_plots_path):
    os.makedirs(trainings_plots_path)

# For plotting learning curves.
steps_curve = []
losses_curve = []
validation_losses_curve = []
accuracies_curve = []
validation_accuracies_curve = []


def create_plot(plot_name):
    plot_training_graph(
        steps_curve,
        losses_curve,
        validation_losses_curve,
        accuracies_curve,
        validation_accuracies_curve,
        os.path.dirname(__file__) + f"/trainings/{plot_name}",
    )


def data_preprocessor(data):
    """Preprocessor for CoGCNet Preprocess the data from dataset.

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


def execute(
    mode,
    epoch,
    total_epoch,
    dataloader,
    preprocessor,
    model,
    loss_func,
    optimizer: Optional[str] = None,
):
    """
    Execute training or validating.
    Args:
        mode: train | validation
        dataloader: dataloader
        preprocessor: preprocess
        model: model
        loss_func: loss function
        optimizer: optional optimizer for validation mode

    Returns:
        accuracy
    """
    is_training = mode == "train"
    if is_training:
        pb_str = "Train"
        model.train()
    else:
        pb_str = "validation"
        model.eval()

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
            data = data.to(device)
            (x, edge_index, edge_attr), (node_true, edge_true) = preprocessor(data)

            if is_training:
                optimizer.zero_grad()

            (node_pred, edge_pred) = model(x, edge_index, edge_attr)

            # Losses
            (losses, corrects) = loss_func(node_pred, edge_pred, node_true, edge_true)
            nodes_loss += losses["nodes"].item() * data.num_graphs
            edges_loss += losses["edges"].item() * data.num_graphs

            if is_training:
                losses["nodes"].backward(retain_graph=True)
                losses["edges"].backward()
                optimizer.step()

            # Accuracies
            nodes_corrects += corrects["nodes"]
            edges_corrects += corrects["edges"]
            dataset_nodes_size += data.num_nodes
            dataset_edges_size += data.num_edges

    nodes_loss /= len(dataloader.dataset)
    edges_loss /= len(dataloader.dataset)

    (losses_curve if is_training else validation_losses_curve).append(
        {"nodes": nodes_loss, "edges": edges_loss}
    )

    nodes_acc = (nodes_corrects / dataset_nodes_size).cpu().numpy()
    edges_acc = (edges_corrects / dataset_edges_size).cpu().numpy()
    (accuracies_curve if is_training else validation_accuracies_curve).append(
        {"nodes": nodes_acc, "edges": edges_acc}
    )

    return (nodes_acc + edges_acc) / 2


if __name__ == "__main__":
    print(f"Using {device} device...")
    date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    dataset = (
        wds.WebDataset("file:" + dataset_path + "/processed/all_000000.tar")
        .decode(pt_decoder)
        .to_tuple(
            "pt",
        )
    )

    dataset_size = len(list(dataset))
    train_size = int(train_ratio * dataset_size)

    plot_name = f"training-size-{dataset_size}-at-{date}.jpg"

    if bool(train_configs["shuffle"]):
        dataset.shuffle(dataset_size * 10)

    train_dataset = list(islice(dataset, 0, train_size))
    validation_dataset = list(islice(dataset, train_size, dataset_size + 1))
    train_loader = DataLoader(train_dataset, batch_size=train_configs["batch_size"])
    validation_loader = DataLoader(
        validation_dataset, batch_size=train_configs["batch_size"]
    )

    model = SPCoDeepGCNet(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        gcn_hidden_channels=model_configs["gcn_hidden_channels"],
        gcn_num_layers=model_configs["gcn_num_layers"],
        mlp_hidden_channels=model_configs["mlp_hidden_channels"],
        mlp_hidden_layers=model_configs["mlp_hidden_layers"],
        node_out=model_configs["node_out"],
        edge_out=model_configs["edge_out"],
    ).to(device)

    print("\n" + str(model) + "\n")

    print("Start training...\n")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_configs["adam_lr"],
        weight_decay=train_configs["adam_weight_decay"],
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        steps_curve.append(epoch + 1)

        execute(
            "train",
            epoch,
            epochs,
            train_loader,
            data_preprocessor,
            model,
            hybrid_loss,
            opt,
        )
        validation_acc = execute(
            "validation",
            epoch,
            epochs,
            validation_loader,
            data_preprocessor,
            model,
            hybrid_loss,
        )

        if validation_acc > best_acc:
            best_acc = validation_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if weight_base_path is not None:
            if epoch % train_configs["weight_save_freq"] == 0:
                weight_name = f"model_weights_ep_{epoch}.pth"
                torch.save(
                    model.state_dict(), os.path.join(weight_base_path, weight_name)
                )

        if epoch % 10 == 0:
            print(
                "\n  Epoch   "
                + "Train Loss (Node,Edge)     Validation Loss        "
                + "Train Acc (Node,Edge)      Validation Acc"
            )

        print(f"{epoch + 1:4}/{epochs}".ljust(10), end="")
        print(
            "{:2.8f}, {:2.8f}  {:2.8f}, {:2.8f}    ".format(
                losses_curve[-1]["nodes"],
                losses_curve[-1]["edges"],
                validation_losses_curve[-1]["nodes"],
                validation_losses_curve[-1]["edges"],
            ),
            end="",
        )
        print(
            "{:2.8f}, {:2.8f}  {:2.8f}, {:2.8f}".format(
                accuracies_curve[-1]["nodes"],
                accuracies_curve[-1]["edges"],
                validation_accuracies_curve[-1]["nodes"],
                validation_accuracies_curve[-1]["edges"],
            )
        )

        if (epoch + 1) % plot_after_epochs == 0:
            create_plot(plot_name)

    if weight_base_path is not None:
        weight_name = train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(weight_base_path, weight_name))
        create_plot(plot_name)