from pathlib import Path

from .config_parser import yaml_parser
from spreadnet.datasets.data_utils.convertor import (
    graphnx_to_dict_spec,
    data_to_input_label,
)
from spreadnet.datasets.dataset_generator import SPGraphDataset
from spreadnet.datasets.graph_generator import GraphGenerator


def get_project_root() -> Path:
    return Path(__file__).parent.parent


__all__ = [
    "yaml_parser",
    "graphnx_to_dict_spec",
    "data_to_input_label",
    "SPGraphDataset",
    "GraphGenerator",
]