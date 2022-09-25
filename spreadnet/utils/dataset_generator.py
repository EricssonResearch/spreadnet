"""Dataset Generator.

@Time    : 9/16/2022 2:59 PM
@Author  : Haodong Zhao
"""

import os.path as osp
import pickle
from glob import glob

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx

from spreadnet.utils.convertor import graphnx_to_dict_spec


class SPGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SPGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.dataset_size = 100

    @property
    def raw_file_names(self):
        raw_filename = []
        dataset_size = len(glob("./dataset/raw/*.pickle"))
        for idx in range(dataset_size):
            filename = "raw_{idx}.pickle".format(idx=idx)
            raw_filename.append(filename)
        return raw_filename

    @property
    def processed_file_names(self):
        """return list of files should be in processed dir,

        if found - skip processing.
        """
        processed_filename = []
        for idx in range(len(self.raw_file_names)):
            filename = "data_{idx}.pt".format(idx=idx)
            processed_filename.append(filename)
        return processed_filename

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # print("-----", raw_path, "\n")
            graph_nx = pickle.load(open(raw_path, "rb"))
            graph_dict = graphnx_to_dict_spec(graph_nx)
            # Get ground truth labels.
            node_tensor = torch.tensor(graph_dict["nodes_feature"]["is_in_path"])
            node_labels = node_tensor.type(torch.int64)

            edge_tensor = torch.tensor(graph_dict["edges_feature"]["is_in_path"])
            edge_labels = edge_tensor.type(torch.int64)

            # remove node and edge features
            for (n, d) in graph_nx.nodes(data=True):
                del d["is_in_path"]

            for (s, e, d) in graph_nx.edges(data=True):
                del d["is_in_path"]

            data = from_networkx(graph_nx)
            data.label = (node_labels, edge_labels)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data
