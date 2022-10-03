"""Train the GCN model.

Usage:
    python train.py [--config config_file_path]

@Time    : 10/3/2022 2:16 PM
@Author  : Haodong Zhao
"""
import copy
import os
import argparse

import torch

import webdataset as wds
from torch_geometric.loader import DataLoader
from typing import Optional

from spreadnet.pyg_gnn.loss.loss import cross_entropy_loss
from spreadnet.pyg_gnn.models import GCNet
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

default_yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

parser = argparse.ArgumentParser(description="Train the GCN model.")
parser.add_argument(
    "--config", default=default_yaml_path, help="Specify the path of the config file. "
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yaml_path = str(get_project_root()) + "/configs.yaml"
yaml_path = args.config
configs = yaml_parser(yaml_path)
train_configs = configs.train
model_configs = configs.model
data_configs = configs.data
dataset_path = os.path.join(
    os.path.dirname(__file__), data_configs["dataset_path"]
).replace("\\", "/")


def train(
    epoch_num,
    dataloader,
    trainable_model,
    loss_func,
    optimizer,
    save_path: Optional[str] = None,
):
    dataset_size = len(list(dataloader.dataset))  # for accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # record the best accuracies

    for epoch in range(epoch_num):
        nodes_loss, nodes_corrects = 0, 0
        dataset_nodes_size = 0  # for accuracy

        for batch, (data,) in enumerate(dataloader):
            data = data.to(device)
            node_true, _ = data.y
            edge_index = data.edge_index
            node_pred = trainable_model(data.x, edge_index, data.edge_attr)
            loss, correct = loss_func(node_pred, node_true)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            dataset_nodes_size += data.num_nodes
            nodes_loss += loss.item() * data.num_graphs
            nodes_corrects += correct

        # get epoch losses and accuracies
        nodes_loss /= dataset_size
        nodes_acc = nodes_corrects / dataset_nodes_size

        cur_acc = nodes_acc

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            f"[Epoch: {epoch + 1:4}/{epoch_num}] "
            f" Losses: {{'nodes': {nodes_loss} }} "
            f" Accuracies: {{'nodes': {nodes_acc} }}"
        )

        if save_path is not None:
            if epoch % train_configs["weight_save_freq"] == 0:
                weight_name = "model_weights_ep_{ep}.pth".format(ep=epoch)
                torch.save(model.state_dict(), os.path.join(save_path, weight_name))

    if save_path is not None:
        weight_name = train_configs["best_weight_name"]
        torch.save(best_model_wts, os.path.join(save_path, weight_name))


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

    model = GCNet(
        in_channels=model_configs["in_channels"],
        num_hidden_layers=model_configs["num_hidden_layers"],
        hidden_channels=model_configs["hidden_channels"],
        out_channels=model_configs["out_channels"],
        use_normalization=model_configs["use_normalization"],
        use_bias=model_configs["use_bias"],
    ).to(device)
    print(model)

    opt_lst = list()
    for i in range(len(model.gcn_stack)):
        if i == len(model.gcn_stack) - 1:
            opt_lst.append(
                dict(
                    params=model.gcn_stack[i].parameters(),
                    weight_decay=train_configs["adam_weight_decay"],
                )
            )
        else:
            opt_lst.append(
                dict(params=model.gcn_stack[i].parameters(), weight_decay=0),
            )  # don't perform weight-decay on the last convolution.

    opt = torch.optim.Adam(
        opt_lst,
        lr=train_configs["adam_lr"],
    )

    weight_base_path = os.path.join(
        os.path.dirname(__file__), train_configs["weight_base_path"]
    )

    if not os.path.exists(weight_base_path):
        os.makedirs(weight_base_path)

    train(
        epoch_num=epochs,
        dataloader=loader,
        trainable_model=model,
        loss_func=cross_entropy_loss,
        optimizer=opt,
        save_path=weight_base_path,
    )
