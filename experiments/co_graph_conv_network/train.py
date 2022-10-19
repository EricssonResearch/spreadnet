"""Train the GCN model.

Usage:
    python train.py [--config config_file_path]

@Time    : 10/3/2022 2:16 PM
@Author  : Haodong Zhao
"""
import copy
import os
import argparse
from datetime import datetime

import torch

import webdataset as wds
from torch_geometric.loader import DataLoader
from typing import Optional

from torch_geometric.transforms import LineGraph
from tqdm import tqdm

from spreadnet.datasets.data_utils.draw import plot_training_graph
from spreadnet.pyg_gnn.loss.loss import hybrid_loss
from spreadnet.pyg_gnn.models import SPCoDeepGCNet, SPCoGCNet
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

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

# yaml_path = str(get_project_root()) + "/configs.yaml"
yaml_path = args.config
dataset_yaml_path = args.dataset_config
configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)

train_configs = configs.train
data_configs = dataset_configs.data
dataset_path = os.path.join(
    os.path.dirname(__file__), "..", data_configs["dataset_path"]
).replace("\\", "/")

# For plotting learning curves.
steps_curve = []
losses_curve = []
accuracies_curve = []


def train(
    epoch_num,
    dataloader,
    trainable_model,
    loss_func,
    optimizer,
    save_path: Optional[str],
    trainings_plots_path,
):
    dataset_size = len(list(dataloader.dataset))  # for accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # record the best accuracies
    line_graph = LineGraph(force_directed=True)

    for epoch in range(epoch_num):
        nodes_loss, edges_loss = 0.0, 0.0
        nodes_corrects, edges_corrects = 0, 0
        dataset_nodes_size, dataset_edges_size = 0, 0  # for accuracy

        for batch, (data,) in tqdm(
            enumerate(dataloader),
            unit="batch",
            total=len(list(dataloader)),
            desc=f"[Epoch: {epoch + 1:4}/{epoch_num}] ",
        ):
            n_data = data.to(device)
            e_data = n_data.clone()
            e_data = line_graph(data=e_data)
            # print(n_data)
            # print(e_data)

            (node_true, edge_true) = data.y
            n_index = n_data.edge_index
            e_index = e_data.edge_index
            n_feats = n_data.x
            e_feats = e_data.x

            node_pred, edge_pred = trainable_model(n_feats, n_index, e_feats, e_index)
            losses, corrects = loss_func(node_pred, edge_pred, node_true, edge_true)
            optimizer.zero_grad()
            losses["nodes"].backward(retain_graph=True)
            losses["edges"].backward()

            # losses["nodes"].backward()
            # # losses["edges"].backward()

            optimizer.step()

            assert data.num_nodes >= corrects["nodes"]
            assert data.num_edges >= corrects["edges"]
            dataset_nodes_size += data.num_nodes
            dataset_edges_size += data.num_edges
            nodes_loss += losses["nodes"].item() * data.num_graphs
            edges_loss += losses["edges"].item() * data.num_graphs
            nodes_corrects += corrects["nodes"]
            edges_corrects += corrects["edges"]

        # get epoch losses and accuracies
        nodes_loss /= dataset_size
        edges_loss /= dataset_size
        nodes_acc = nodes_corrects / dataset_nodes_size
        edges_acc = edges_corrects / dataset_edges_size

        cur_acc = nodes_acc

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f" Losses: {{'nodes': {nodes_loss}, 'edges': {edges_loss} }}\n"
            f" Accuracies: {{'nodes': {nodes_acc}, 'edges': {edges_acc}}}"
        )

        steps_curve.append(epoch + 1)
        losses_curve.append({"nodes": nodes_loss, "edges": edges_loss})
        accuracies_curve.append(
            {"nodes": nodes_acc.cpu().numpy(), "edges": edges_acc.cpu().numpy()}
        )

        if save_path is not None:
            if epoch % train_configs["weight_save_freq"] == 0:
                weight_name = "model_weights_ep_{ep}.pth".format(ep=epoch)
                torch.save(model.state_dict(), os.path.join(save_path, weight_name))

    if save_path is not None:
        weight_name = train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(save_path, weight_name))

        date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plot_name = f"training-size-{dataset_size}-at-{date}.jpg"
        plot_training_graph(
            steps_curve,
            losses_curve,
            accuracies_curve,
            trainings_plots_path + f"/{plot_name}",
        )


if __name__ == "__main__":
    print(f"Using {device} device...")

    epochs = train_configs["epochs"]

    dataset = (
        wds.WebDataset("file:" + dataset_path + "/processed/all_000000.tar")
        .decode(pt_decoder)
        .to_tuple(
            "pt",
        )
    )

    if bool(train_configs["shuffle"]):
        dataset.shuffle(len(list(dataset)) * 10)

    loader = DataLoader(dataset, batch_size=train_configs["batch_size"])

    which_model = train_configs["which_model"]
    weight_base_path = train_configs["weight_base_path"]
    model_configs = configs.model
    log_folder = "trainings"
    model = SPCoGCNet(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        hidden_channels=model_configs["hidden_channels"],
        num_layers=model_configs["num_layers"],
        node_out=model_configs["node_out"],
        edge_out=model_configs["edge_out"],
    ).to(device)

    if which_model == "deep":
        print("Prepare to train deep GCN model...")
        weight_base_path = train_configs["deep_weight_base_path"]
        log_folder = "deep_trainings"
        model_configs = configs.deep_model
        model = SPCoDeepGCNet(
            node_in=model_configs["node_in"],
            edge_in=model_configs["edge_in"],
            hidden_channels=model_configs["hidden_channels"],
            num_layers=model_configs["num_layers"],
            node_out=model_configs["node_out"],
            edge_out=model_configs["edge_out"],
        ).to(device)
    else:
        print("Prepare to train GCN model...")

    print(model, "\n")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_configs["adam_lr"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")

    weight_base_path = os.path.join(os.path.dirname(__file__), weight_base_path)

    if not os.path.exists(weight_base_path):
        os.makedirs(weight_base_path)

    trainings_plots_path = os.path.join(os.path.dirname(__file__), log_folder)

    if not os.path.exists(trainings_plots_path):
        os.makedirs(trainings_plots_path)

    train(
        epoch_num=epochs,
        dataloader=loader,
        trainable_model=model,
        loss_func=hybrid_loss,
        optimizer=opt,
        save_path=weight_base_path,
        trainings_plots_path=trainings_plots_path,
    )
