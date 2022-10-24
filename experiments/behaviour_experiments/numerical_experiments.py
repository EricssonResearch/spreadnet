import sys

# import json

# from spreadnet.utils.experiment_utils import ExperimentUtils

from spreadnet.tf_gnn.model import gnn


from spreadnet.pyg_gnn.models import EncodeProcessDecode

sys.modules["EncodeProcessDecode"] = EncodeProcessDecode
sys.modules["gnn"] = gnn


def increasing_graph_size():
    """Gradually increasing the graph size while keeping theta at the same
    ratio.

    This experiment first generates the data using the experiment utils.
    """


if __name__ == "__main__":
    increasing_graph_size()
