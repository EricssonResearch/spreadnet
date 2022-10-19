"""

    @Time    : 10/12/2022 10:52 AM
    @Author  : Haodong Zhao

"""

import copy
import os
import argparse

import torch

import webdataset as wds
from torch_geometric.loader import DataLoader
from typing import Optional

from torch_geometric.transforms import LineGraph
from tqdm import tqdm

from spreadnet.pyg_gnn.loss.loss import hybrid_loss

from spreadnet.pyg_gnn.models.graph_conv_network.sp_deep_gcn import SPCoDeepGCN
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
model_configs = configs.model
data_configs = dataset_configs.data
dataset_path = os.path.join(
    os.path.dirname(__file__), "..", data_configs["dataset_path"]
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
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # record the best accuracies
    line_graph = LineGraph(force_directed=True)

    for epoch in range(epoch_num):
        nodes_loss, edges_loss = 0.0, 0.0
        nodes_corrects, edges_corrects = 0, 0
        dataset_nodes_size, dataset_edges_size = 0, 0  # for accuracy

        print(f"[Epoch: {epoch + 1:4}/{epoch_num}] ")
        for batch, (data,) in tqdm(
            enumerate(dataloader), unit="batch", total=len(list(dataloader))
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
            optimizer.step()

            # assert data.num_nodes >= corrects["nodes"]
            # assert data.num_edges >= corrects["edges"]
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

    model = SPCoDeepGCN(
        node_in=3,
        edge_in=1,
        hidden_channels=64,
        num_layers=28,
        node_out=2,
        edge_out=2,
    ).to(device)

    print(model)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")

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
