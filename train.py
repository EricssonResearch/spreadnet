"""

    @Time    : 9/16/2022 1:31 PM
    @Author  : Haodong Zhao

"""
import copy
from typing import Optional

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from architecture.loss import loss_fn
from architecture.models import EncodeProcessDecode
from utils import get_project_root, SPGraphDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch_num, dataloader, trainable_model, loss_func, optimizer,
          save_path: Optional[str] = None):
    dataset_size = len(dataloader.dataset)  # for accuracy

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # record the best accuracies

    for epoch in range(epoch_num):
        nodes_loss, edges_loss = 0.0, 0.0
        nodes_corrects, edges_corrects = 0, 0
        dataset_nodes_size, dataset_edges_size = 0, 0  # for accuracy

        for batch, data in enumerate(dataloader):
            data = data.to(device)

            losses, corrects = loss_func(data, trainable_model)
            optimizer.zero_grad()
            losses['nodes'].backward(retain_graph=True)
            losses['edges'].backward(retain_graph=True)
            optimizer.step()

            assert (data.num_nodes >= corrects['nodes'])
            assert (data.num_edges >= corrects['edges'])
            dataset_nodes_size += data.num_nodes
            dataset_edges_size += data.num_edges
            nodes_loss += losses['nodes'].item() * data.num_graphs
            edges_loss += losses['edges'].item() * data.num_graphs
            nodes_corrects += corrects['nodes']
            edges_corrects += corrects['edges']

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
            f"\n\t\t    Accuracies: {{'nodes': {nodes_acc}, 'edges': {edges_acc}}}")

        if save_path is not None:
            if epoch % 50 == 0:
                weight_name = "model_weights_ep_{ep}.pth".format(ep=epoch)
                torch.save(model.state_dict(), save_path + weight_name)

    if save_path is not None:
        weight_name = "model_weights_best.pth"
        model.load_state_dict(best_model_wts)
        torch.save(model, save_path + weight_name)


if __name__ == '__main__':
    print(f"Using {device} device...")

    epochs = 2000
    dataset = SPGraphDataset(root=str(get_project_root()) + "/dataset/")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EncodeProcessDecode(
        node_in=3,
        edge_in=1,
        node_out=2,
        edge_out=2,
        latent_size=128,
        num_message_passing_steps=12,
        num_mlp_hidden_layers=2,
        mlp_hidden_size=128
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # print(model)

    weight_base_path = str(get_project_root()) + "/weights/"

    train(epoch_num=epochs, dataloader=loader,
          trainable_model=model, loss_func=loss_fn, optimizer=opt,
          save_path=weight_base_path)
