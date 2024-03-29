import networkx as nx
import json
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from time import time
import enum
import platform

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
    hybrid_complete_path,
)
from spreadnet.utils.model_loader import load_model
from spreadnet.dijkstra_memoization import dijkstra_runner

torch.multiprocessing.set_start_method("spawn", force=True)

if platform.system() == "Windows":
    import pyreadline3

    pyreadline3.modes
else:
    import readline

    readline.set_auto_history(True)


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
    MPNN = "message_passing_network"
    CGCN = "co_graph_conv_network"
    GCN = "deep_graph_conv_network"
    GAT = "graph_attention_network"


class QueryProcessor:
    modes = ["AUTO", "DIJKSTRA", "GNN"]
    models = ["MPNN", "GCN", "GAT", "CGCN"]
    use_gnn_if_nodes_above = 4000
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
        hybrid: bool,
        logger,
        exp_path,
    ):
        print(f"{bcolors.HEADER}{bcolors.BOLD}Query Processor!{bcolors.ENDC}")
        self.which_mode = mode
        self.which_model = model
        self.bidirectional = bidirectional
        self.dijkstra_full = dijkstra_full
        self.hybrid = hybrid
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
            self.qpl.info(f"{bcolors.OKGREEN}MPNN{bcolors.ENDC}")
        else:
            self.qpl.info(f"{bcolors.FAIL}MPNN{bcolors.ENDC}")
        if self.get_weight(ModelFolders.CGCN.value):
            self.models.append("CGCN")
            self.qpl.info(f"{bcolors.OKGREEN}CGCN{bcolors.ENDC}")
        else:
            self.qpl.info(f"{bcolors.FAIL}CGCN{bcolors.ENDC}")
        if self.get_weight(ModelFolders.GCN.value):
            self.models.append("GCN")
            self.qpl.info(f"{bcolors.OKGREEN}GCN{bcolors.ENDC}")
        else:
            self.qpl.info(f"{bcolors.FAIL}GCN{bcolors.ENDC}")
        if self.get_weight(ModelFolders.GAT.value):
            self.models.append("GAT")
            self.qpl.info(f"{bcolors.OKGREEN}GAT{bcolors.ENDC}")
        else:
            self.qpl.info(f"{bcolors.FAIL}GAT{bcolors.ENDC}")

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

    def get_fig_title(self, mode, best_path_weight):
        full_mode = mode if mode == "DIJKSTRA" else f"{mode} {self.which_model}"
        return f"Path found using {full_mode}, Edge Weights: {best_path_weight}"

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

            (pred_graph_nx, truth_total_weight) = process_prediction(graph_nx, preds)
            (pred_graph_nx_r, _) = process_prediction(graph_nx_r, preds_r)

            aggregated_nx = aggregate_results(deepcopy(pred_graph_nx), pred_graph_nx_r)

            if self.hybrid:
                is_path_complete = True
                (prob_path) = hybrid_complete_path(deepcopy(pred_graph_nx))
            else:
                (is_path_complete, prob_path) = probability_first_search(
                    deepcopy(pred_graph_nx), start_node, end_node
                )

            applied_nx, pred_edge_weights = apply_path_on_graph(
                deepcopy(pred_graph_nx), prob_path, True
            )

            if self.hybrid:
                is_path_complete_a = True
                (prob_path_a) = hybrid_complete_path(deepcopy(aggregated_nx))
            else:
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

            fig = plt.figure(figsize=(self.plot_size * 2, self.plot_size * 3))
            fig.suptitle(self.get_fig_title("GNN", best_path_weight))
            draw_networkx(
                f"Truth, Edge Weights: {truth_total_weight}",
                fig,
                graph_nx,
                1,
                6,
                per_row=2,
            )
            draw_networkx(
                "Pred",
                fig,
                pred_graph_nx,
                2,
                6,
                "probability",
                "probability",
                per_row=2,
            )
            draw_networkx(
                "Pred Rev",
                fig,
                pred_graph_nx_r,
                3,
                6,
                "probability",
                "probability",
                per_row=2,
            )

            draw_networkx(
                "Aggregated",
                fig,
                aggregated_nx,
                4,
                6,
                "probability",
                "probability",
                per_row=2,
            )

            draw_networkx(
                f"Prob Walk on Pred, Edge Weights: {pred_edge_weights}",
                fig,
                applied_nx,
                5,
                6,
                "default",
                "probability",
                per_row=2,
            )

            draw_networkx(
                f"Prob Walk on Aggregated, Edge Weights: {pred_edge_weights_a}",
                fig,
                applied_nx_a,
                6,
                6,
                "default",
                "probability",
                per_row=2,
            )
            fig.tight_layout(pad=5)

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

            if self.hybrid:
                (prob_path) = hybrid_complete_path(deepcopy(pred_graph_nx))
            else:
                (_, prob_path) = probability_first_search(
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
                + f" Bidirectional: {self.bidirectional}, "
                + f" Hybrid: {self.hybrid}"
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
            + ", hybrid=True|False"
        )
        print("To exit, enter: exit")
        print(f"{bcolors.OKCYAN}Enter a json graph path{bcolors.ENDC}")
        user_input = input("> ").replace("\\", "/")

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
        elif user_input.startswith("hybrid="):
            self.hybrid = bool(json.loads(user_input.split("=")[-1].lower()))
            self.qpl.info(f"Hybrid path finding set to: {self.hybrid}")
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

            if not (self.bidirectional and mode == "GNN"):
                fig = plt.figure(figsize=(self.plot_size, self.plot_size))
                draw_networkx(
                    self.get_fig_title(mode, best_path_weight),
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
