"""Train the model.

Usage:
    python train.py [--config config_file_path]

@Time    : 9/16/2022 1:31 PM
@Author  : Haodong Zhao
"""
import copy
import os
import argparse
import torch
import webdataset as wds
from torch_geometric.loader import DataLoader
from typing import Optional

from spreadnet.pyg_gnn.loss import hybrid_loss
from spreadnet.pyg_gnn.models import EncodeProcessDecode
from spreadnet.utils import data_to_input_label, yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder

default_yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

parser = argparse.ArgumentParser(description="Train the model.")
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
        nodes_loss, edges_loss = 0.0, 0.0
        nodes_corrects, edges_corrects = 0, 0
        dataset_nodes_size, dataset_edges_size = 0, 0  # for accuracy

        for batch, (data,) in enumerate(dataloader):
            data = data.to(device)

            (nodes_data, edges_data), (node_true, edge_true) = data_to_input_label(data)
            edge_index = data.edge_index
            node_pred, edge_pred = trainable_model(nodes_data, edge_index, edges_data)
            # losses, corrects = loss_func(data, trainable_model)
            losses, corrects = loss_func(node_pred, edge_pred, node_true, edge_true)
            optimizer.zero_grad()
            losses["nodes"].backward(retain_graph=True)
            losses["edges"].backward(retain_graph=True)
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
            f"[Epoch: {epoch + 1:4}/{epoch_num}] "
            f" Losses: {{'nodes': {nodes_loss}, 'edges': {edges_loss} }} "
            f"\n\t\t    Accuracies: {{'nodes': {nodes_acc}, 'edges': {edges_acc}}}"
        )

        if save_path is not None:
            if epoch % train_configs["weight_save_freq"] == 0:
                weight_name = "model_weights_ep_{ep}.pth".format(ep=epoch)
                torch.save(model.state_dict(), os.path.join(save_path, weight_name))

    if save_path is not None:
        weight_name = train_configs["best_weight_name"]
        model.load_state_dict(best_model_wts)
        torch.save(model, os.path.join(save_path, weight_name))


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

    model = EncodeProcessDecode(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        node_out=model_configs["node_out"],
        edge_out=model_configs["edge_out"],
        latent_size=model_configs["latent_size"],
        num_message_passing_steps=model_configs["num_message_passing_steps"],
        num_mlp_hidden_layers=model_configs["num_mlp_hidden_layers"],
        mlp_hidden_size=model_configs["mlp_hidden_size"],
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=train_configs["adam_lr"],
        weight_decay=train_configs["adam_weight_decay"],
    )
    # print(model)
    weight_base_path = os.path.join(
        os.path.dirname(__file__), train_configs["weight_base_path"]
    )

    if not os.path.exists(weight_base_path):
        os.makedirs(weight_base_path)

    train(
        epoch_num=epochs,
        dataloader=loader,
        trainable_model=model,
        loss_func=hybrid_loss,
        optimizer=opt,
        save_path=weight_base_path,
    )
