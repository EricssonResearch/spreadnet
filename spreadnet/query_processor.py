import networkx as nx
import json
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import readline
import torch
from time import time
import enum

from spreadnet.utils import yaml_parser
from spreadnet.datasets.data_utils.processor import process_nx
from spreadnet.datasets.data_utils.draw import draw_networkx
from spreadnet.datasets.data_utils.encoder import NpEncoder
from spreadnet.utils.post_processor import (
    process_prediction,
    swap_start_end,
    aggregate_results,
    probability_first_search,
    apply_path_on_graph,
    get_start_end_nodes,
)
from spreadnet.utils.model_loader import load_model
from spreadnet.dijkstra_memoization import dijkstra_runner

readline.set_auto_history(True)
torch.multiprocessing.set_start_method("spawn", force=True)


class bcolors:
    HEADER = "\033[95m"
    HIGHLIGHT = "\033[100m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    GRAY = "\033[90m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ModelFolders(enum.Enum):
    MPNN = "encode_process_decode"
    DeepCoGCN = "co_graph_conv_network"
    DeepGCN = "deep_graph_conv_network"
    GAT = "graph_attention_network"


class QueryProcessor:
    modes = ["AUTO", "DIJKSTRA", "GNN"]
    models = ["MPNN", "DeepGCN", "GAT", "DeepCoGCN"]
    use_gnn_if_nodes_above = 40
    which_weight = "model_weights_best.pth"
    plot_size = 20

    runtime_start = -1
    runtime_end = -1

    def __init__(
        self,
        mode: str,
        model: str,
        bidirectional: bool,
        dijkstra_full: bool,
        logger,
        exp_path,
    ):
        print(f"{bcolors.HEADER}{bcolors.BOLD}Query Processor!{bcolors.ENDC}")
        self.which_mode = mode
        self.which_model = model
        self.bidirectional = bidirectional
        self.dijkstra_full = dijkstra_full
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qpl = logger
        self.exp_path = exp_path
        self.out_path = os.path.join(exp_path, "qp_out")
        self.check_trained()
        self.load_config()

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def check_trained(self):
        self.models = []
        self.qpl.info("Trained models:")

        if self.get_weight(ModelFolders.MPNN.value):
            self.models.append("MPNN")
            self.qpl.info(f"MPNN {bcolors.OKGREEN}✓{bcolors.ENDC}")
        if self.get_weight(ModelFolders.DeepCoGCN.value):
            self.models.append("DeepCoGCN")
            self.qpl.info(f"DeepCoGCN {bcolors.OKGREEN}✓{bcolors.ENDC}")
        if self.get_weight(ModelFolders.DeepGCN.value):
            self.models.append("DeepGCN")
            self.qpl.info(f"DeepGCN {bcolors.OKGREEN}✓{bcolors.ENDC}")
        if self.get_weight(ModelFolders.GAT.value):
            self.models.append("GAT")
            self.qpl.info(f"GAT {bcolors.OKGREEN}✓{bcolors.ENDC}")

        if len(self.models) == 0:
            raise Exception("Please train one of the models")

        if self.which_model not in self.models:
            self.which_model = self.models[0]

    def get_weight(self, m_folder):
        p = os.path.join(self.exp_path, m_folder, "weights", self.which_weight)

        if os.path.exists(p):
            return p
        else:
            return False

    def load_config(self):
        if self.which_model not in self.models:
            print(f"{bcolors.FAIL}Please check the model name. -h for more details.")
            exit()

        m_folder = ModelFolders[self.which_model].value
        self.model_configs = yaml_parser(
            os.path.join(self.exp_path, m_folder, "configs.yaml")
        ).model
        self.model = load_model(self.which_model, self.model_configs, self.device)

        weight_path = self.get_weight(m_folder)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=torch.device(self.device))
        )
        self.model.eval()

    def use_dijkstra(self, graph_json):
        self.qpl.info("Using DIJKSTRA")
        graph_nx = nx.node_link_graph(graph_json)
        (start, end) = get_start_end_nodes(graph_nx.nodes(data=True))
        graph_hash = dijkstra_runner.hash_graph_weisfeiler(graph_nx)
        self.runtime_start = time()
        best_path = (
            dijkstra_runner.shortest_path
            if self.dijkstra_full
            else dijkstra_runner.shortest_path_single
        )(graph_nx, graph_hash, start, end)
        self.runtime_end = time()
        (best_path_graph, best_path_weight) = apply_path_on_graph(
            graph_nx, best_path, True
        )
        return best_path_weight, best_path, best_path_graph

    def predict(self, data):
        if self.which_model == "GAT":
            (node_pred, edge_pred) = self.model(
                data.x,
                data.edge_index,
                data.edge_attr,
                return_attention_weights=self.model_configs["return_attention_weights"],
            )
        else:
            (node_pred, edge_pred) = self.model(data.x, data.edge_index, data.edge_attr)

        return {"nodes": node_pred, "edges": edge_pred}

    def use_gnn_bi(self, graph_json):
        self.qpl.info(f"Using GNN {self.which_model} on {self.device} device...")

        best_path = None
        best_path_graph = None
        best_path_weight = -1

        with torch.no_grad():
            graph_nx = nx.node_link_graph(graph_json)
            graph_nx_r = deepcopy(graph_nx)

            (start_node, end_node) = get_start_end_nodes(graph_nx.nodes(data=True))
            swap_start_end(graph_nx_r, start_node, end_node)

            graph_data = process_nx(graph_nx)
            graph_data_r = process_nx(graph_nx_r)

            graph_data.to(self.device)
            graph_data_r.to(self.device)

            self.runtime_start = time()
            preds = self.predict(graph_data)
            preds_r = self.predict(graph_data_r)
            self.runtime_end = time()

            (pred_graph_nx, _) = process_prediction(graph_nx, preds)
            (pred_graph_nx_r, _) = process_prediction(graph_nx_r, preds_r)

            aggregated_nx = aggregate_results(deepcopy(pred_graph_nx), pred_graph_nx_r)

            (is_path_complete, prob_path) = probability_first_search(
                deepcopy(pred_graph_nx), start_node, end_node
            )

            applied_nx, pred_edge_weights = apply_path_on_graph(
                deepcopy(pred_graph_nx), prob_path, True
            )

            (is_path_complete_a, prob_path_a) = probability_first_search(
                deepcopy(aggregated_nx), start_node, end_node
            )

            applied_nx_a, pred_edge_weights_a = apply_path_on_graph(
                deepcopy(aggregated_nx), prob_path_a, True
            )

            if is_path_complete and pred_edge_weights < pred_edge_weights_a:
                best_path = prob_path
                best_path_graph = applied_nx
                best_path_weight = pred_edge_weights
            else:
                best_path = prob_path_a
                best_path_graph = applied_nx_a
                best_path_weight = pred_edge_weights_a

        return best_path_weight, best_path, best_path_graph

    def use_gnn(self, graph_json):
        self.qpl.info(f"Using GNN {self.which_model} on {self.device} device...")

        with torch.no_grad():
            graph_nx = nx.node_link_graph(graph_json)
            (start_node, end_node) = get_start_end_nodes(graph_nx.nodes(data=True))

            graph_data = process_nx(graph_nx)
            graph_data.to(self.device)

            self.runtime_start = time()
            preds = self.predict(graph_data)
            self.runtime_end = time()

            (pred_graph_nx, _) = process_prediction(graph_nx, preds)

            (is_path_complete, prob_path) = probability_first_search(
                deepcopy(pred_graph_nx), start_node, end_node
            )

            applied_nx, pred_edge_weights = apply_path_on_graph(
                deepcopy(pred_graph_nx), prob_path, True
            )

        return pred_edge_weights, prob_path, applied_nx

    def read_input(self):
        print(f"{bcolors.OKBLUE}Mode: {self.which_mode}")
        if self.which_mode != "DIJKSTRA":
            print(
                f"GNN Model: {self.which_model}, "
                + f" Bidirectional: {self.bidirectional}"
            )
        if self.which_mode != "GNN":
            print(f"Dijkstra Full: {self.dijkstra_full}")

        print(
            f"{bcolors.GRAY}To change settings, enter: "
            + "mode="
            + "|".join(self.modes)
            + ", model="
            + "|".join(self.models)
            + ", bidirectional=True|False"
            + ", dijkstra_full=True|False"
        )
        print("To exit, enter: exit")
        print(f"{bcolors.OKCYAN}Enter a json graph path{bcolors.ENDC}")
        user_input = input("> ")

        if user_input.lower() == "exit":
            exit()
        elif user_input.startswith("mode="):
            new_mode = user_input.split("=")[-1]
            if new_mode in self.modes:
                self.which_mode = new_mode
                self.qpl.info(f"Mode set to: {self.which_mode}")
            else:
                self.qpl.error("Invalid mode")
        elif user_input.startswith("model="):
            new_model = user_input.split("=")[-1]
            self.check_trained()

            if new_model in self.models:
                self.which_model = new_model
                self.load_config()
                self.qpl.info(f"Model set to: {self.which_model}")
            else:
                self.qpl.error("Invalid model or model not trained")
        elif user_input.startswith("bidirectional="):
            self.bidirectional = bool(json.loads(user_input.split("=")[-1].lower()))
            self.qpl.info(f"GNN bidirectional set to: {self.bidirectional}")
        elif user_input.startswith("dijkstra_full="):
            self.dijkstra_full = bool(json.loads(user_input.split("=")[-1].lower()))
            self.qpl.info(f"dijkstra_full search set to: {self.dijkstra_full}")
        else:
            if not os.path.exists(user_input):
                self.qpl.error(f"Input file does not exist, {user_input}\n")
                return

            self.qpl.info(f"Selected mode: {self.which_mode}")
            self.qpl.info(f"Graph path: {user_input}")

            graph_json = json.load(open(user_input))[0]
            number_of_nodes = len(graph_json["nodes"])
            self.qpl.info(f"Num Nodes: {number_of_nodes}")

            mode = self.which_mode
            if mode == "AUTO":
                if number_of_nodes > self.use_gnn_if_nodes_above:
                    mode = "GNN"
                else:
                    mode = "DIJKSTRA"

            if mode == "DIJKSTRA":
                (best_path_weight, best_path, best_path_graph) = self.use_dijkstra(
                    graph_json
                )
            elif mode == "GNN":
                (best_path_weight, best_path, best_path_graph) = (
                    self.use_gnn_bi if self.bidirectional else self.use_gnn
                )(graph_json)
            else:
                raise Exception("Invalid mode")

            best_path_weight = round(best_path_weight, 3)
            self.qpl.info(f"Best path: {best_path_weight} {best_path}")
            self.qpl.info(
                f"Time Taken: {(self.runtime_end - self.runtime_start):.15f}s"
            )

            file_name = str(user_input).split("/")[-1]
            plot_name = self.out_path + f"/{file_name}"

            with open(f"{plot_name}.json", "w") as outfile:
                json.dump([nx.node_link_data(best_path_graph)], outfile, cls=NpEncoder)

            self.qpl.info("Drawing result...")
            fig = plt.figure(figsize=(self.plot_size, self.plot_size))

            full_mode = mode if mode == "DIJKSTRA" else f"{mode} {self.which_model}"
            draw_networkx(
                f"Path found using {full_mode}, Edge Weights: {best_path_weight}",
                fig,
                best_path_graph,
                1,
                1,
                per_row=1,
            )

            fig.tight_layout()
            plt.savefig(f"{plot_name}.jpg", pad_inches=0, bbox_inches="tight")
            plt.clf()
            self.qpl.info(f"Result saved at {plot_name}.jpg|json")

        print("\n")
