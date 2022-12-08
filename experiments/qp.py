"""Query Processor.

Args:
    --mode "AUTO|DIJKSTRA|GNN", defaults: "AUTO"
    --model "MPNN|DeepGCN|GAT|DeepCoGCN", defaults: "MPNN"

Example:
    python qp.py --mode "GNN"
"""
import argparse
import os
import logging

import spreadnet.utils.log_utils as log_utils
from spreadnet.query_processor import QueryProcessor

parser = argparse.ArgumentParser(description="Query Processor")

parser.add_argument(
    "--mode", help="Specify QP mode. Options: AUTO|DIJKSTRA|GNN", default="AUTO"
)
parser.add_argument(
    "--model",
    help="Specify GNN model AUTO|GNN mode is selected. "
    + " Options: MPNN|DeepGCN|GAT|DeepCoGCN",
    default="MPNN",
)

parser.add_argument(
    "--bidirectional",
    help="Toggle bidirectional GNN",
    action="store_true",
    default=False,
)

args = parser.parse_args()

if __name__ == "__main__":
    log_save_path = os.path.join(
        os.path.join(os.path.dirname(__file__), "qp_out"), "logs"
    ).replace("\\", "/")
    qpl = log_utils.init_file_console_logger("qpl", log_save_path, "qp")

    try:
        qp = QueryProcessor(
            args.mode, args.model, args.bidirectional, qpl, os.path.dirname(__file__)
        )
        while True:
            qp.read_input()

    except Exception as e:
        qpl.exception(e)

    logging.shutdown()
