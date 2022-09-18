from .graph_generator import GraphGenerator
from .dataset_generator import SPGraphDataset
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent
