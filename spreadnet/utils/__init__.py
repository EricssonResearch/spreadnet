from pathlib import Path

from .config_parser import yaml_parser
from spreadnet.datasets.data_utils.convertor import (
    graphnx_to_dict_spec,
)
from spreadnet.datasets.graph_generator import GraphGenerator


def get_project_root() -> Path:
    return Path(__file__).parent.parent


__all__ = [
    "yaml_parser",
    "graphnx_to_dict_spec",
    "GraphGenerator",
]
