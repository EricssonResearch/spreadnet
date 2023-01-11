"""Query Processor.

Args:
    --mode "AUTO|DIJKSTRA|GNN", defaults: "AUTO"
    --model "MPNN|DeepGCN|GAT|DeepCoGCN", defaults: "MPNN"
    --bidirectional "True|False", defaults: "False"
    --dijkstra-full "True|False", defaults: "False"

Example:
    python qp.py --mode "GNN"
"""
import argparse
import os
import logging

import spreadnet.utils.log_utils as log_utils
from spreadnet.query_processor import QueryProcessor
from codecarbon import OfflineEmissionsTracker

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
parser.add_argument(
    "--dijkstra-full",
    help="Toggle dijkstra full path search",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--hybrid",
    help="Toggle hybrid full path search",
    action="store_true",
    default=False,
)

args = parser.parse_args()

if __name__ == "__main__":
    log_save_path = os.path.join(
        os.path.join(os.path.dirname(__file__), "qp_out"), "logs"
    ).replace("\\", "/")
    qpl = log_utils.init_file_console_logger("qpl", log_save_path, "qp")

    co2_emissions = OfflineEmissionsTracker(country_iso_code="SWE")
    co2_emissions.start()
    try:
        qp = QueryProcessor(
            args.mode,
            args.model,
            args.bidirectional,
            args.dijkstra_full,
            args.hybrid,
            qpl,
            os.path.dirname(__file__),
        )

        while True:
            qp.read_input()
            co2_emissions_final = co2_emissions.stop()
            qpl.info(
                f"Co2 Emissions: {co2_emissions_final} kg co2.eq/KWh."
                f"For more data see emissions.csv"
            )

    except Exception as e:
        qpl.exception(e)

    logging.shutdown()
