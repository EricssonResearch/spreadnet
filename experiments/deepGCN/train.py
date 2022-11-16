import copy
import os
import argparse
from datetime import datetime

import torch

import webdataset as wds
from torch_geometric.loader import DataLoader
from typing import Optional

from spreadnet.datasets.data_utils.draw import plot_training_graph
from spreadnet.pyg_gnn.utils.loss import hybrid_loss
from spreadnet.pyg_gnn.models.deepGCN.sp_deepGCN import SPDeepGCN
from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.decoder import pt_decoder
from spreadnet.pyg_gnn.utils.metrics import get_correct_predictions

default_yaml_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "../dataset_configs.yaml"
)

default_loss_type = "d"

parser = argparse.ArgumentParser(description="Train the model.")
parser.add_argument(
    "--config", default=default_yaml_path, help="Specify the path of the config file. "
)
parser.add_argument(
    "--dataset-config",
    default=default_dataset_yaml_path,
    help="Specify the path of the dataset config file. ",
)
parser.add_argument(
    "--loss-type",
    default=default_loss_type,
    help="Specify if you want to use the original loss (d) or weighted loss (w)",
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

loss_type = args.loss_type

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

            (node_true, edge_true) = data.y
            edge_index = data.edge_index
            node_pred, edge_pred = trainable_model(data.x, edge_index, data.edge_attr)
            # losses, corrects = loss_func(data, trainable_model)
            losses = loss_func(node_pred, edge_pred, node_true, edge_true, loss_type)
            _, corrects = get_correct_predictions(
                node_pred, edge_pred, node_true, edge_true
            )
            optimizer.zero_grad()
            losses["nodes"].backward(retain_graph=True)
            losses["edges"].backward()

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
            os.path.dirname(__file__) + f"/trainings/{plot_name}",
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

    model = SPDeepGCN(
        node_in=model_configs["node_in"],
        edge_in=model_configs["edge_in"],
        encoder_hidden_channels=model_configs["encoder_hidden_channels"],
        encoder_layers=model_configs["encoder_layers"],
        gcn_hidden_channels=model_configs["gcn_hidden_channels"],
        gcn_layers=model_configs["gcn_layers"],
        decoder_hidden_channels=model_configs["decoder_hidden_channels"],
        decoder_layers=model_configs["decoder_layers"],
    ).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    weight_base_path = os.path.join(
        os.path.dirname(__file__), train_configs["weight_base_path"]
    )

    if not os.path.exists(weight_base_path):
        os.makedirs(weight_base_path)

    trainings_plots_path = os.path.join(os.path.dirname(__file__), "trainings")

    if not os.path.exists(trainings_plots_path):
        os.makedirs(trainings_plots_path)

    train(
        epoch_num=epochs,
        dataloader=loader,
        trainable_model=model,
        loss_func=hybrid_loss,
        optimizer=optimizer,
        save_path=weight_base_path,
    )
