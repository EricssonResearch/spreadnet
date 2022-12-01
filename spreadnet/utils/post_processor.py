import networkx as nx
from copy import deepcopy
from torch.nn.functional import softmax


def swap_start_end(graph_nx: nx.DiGraph):
    """Swap start and end node for bidirectional inference.

    :param graph_nx: networkx graph to predict

    :return: swapped graph
    """

    swapped = 0
    for (_, data) in graph_nx.nodes(data=True):
        if data["is_start"]:
            data["is_end"] = True
            data["is_start"] = False
            swapped += 1
        elif data["is_end"]:
            data["is_start"] = True
            data["is_end"] = False
            swapped += 1

        if swapped == 2:
            break

    rev_edges = []

    for (u, v, d) in graph_nx.edges(data=True):
        if not graph_nx.has_edge(v, u):
            rev_edges.append((v, u, {"weight": d["weight"], "is_in_path": False}))

    graph_nx.add_edges_from(rev_edges)

    return graph_nx


def process_prediction(input_graph_nx: nx.DiGraph, preds, infers):
    """Construct networkx graph from prediction results.

    Args:
        input_graph_nx: original graph
        preds: nodes and edges prediction
        infers: nodes and edges infers

    Returns:
        (predicted, truth_total_edge_weight)
    """

    pred_graph_nx = deepcopy(input_graph_nx)
    node_pred = preds["nodes"].cpu().detach()
    edge_pred = preds["edges"].cpu().detach()

    truth_total_weight = 0.0

    for i, (n, data) in enumerate(pred_graph_nx.nodes(data=True)):
        data["is_in_path"] = bool(infers["nodes"][i])

        probability = softmax(node_pred[i], dim=-1).numpy()[1]
        data["probability"] = probability

    for i, (u, v, data) in enumerate(pred_graph_nx.edges(data=True)):
        if data["is_in_path"]:
            truth_total_weight += data["weight"]

        data["is_in_path"] = bool(infers["edges"][i])

        probability = softmax(edge_pred[i], dim=-1).numpy()[1]
        data["probability"] = probability

    return pred_graph_nx, truth_total_weight


def aggregate_results(g1: nx.DiGraph, g2: nx.DiGraph):
    """Merge two results together favoring nodes/edges with higher
    probabilities into g1.

    :pre: two graphs must have the same structure

    :param g1: inferred graph
    :param g2: inferred reversed graph

    :return: aggregated graph
    """

    g1_nodes = g1.nodes(data=True)
    g2_nodes = g2.nodes(data=True)
    g2_edges = g2.edges(data=True)

    for idx, (_, d2) in enumerate(g2_nodes):
        if d2["is_in_path"]:
            g1_nodes[idx]["is_in_path"] = True
        if d2["probability"] > g1_nodes[idx]["probability"]:
            g1_nodes[idx]["probability"] = d2["probability"]

    for idx, (u, v, d2) in enumerate(g2_edges):
        d1 = g1.get_edge_data(v, u)

        if d2["is_in_path"]:
            d1["is_in_path"] = True
        if d2["probability"] > d1["probability"]:
            d1["probability"] = d2["probability"]

    return g1


def _exhaustive_probability_walk(
    G: nx.DiGraph,
    nodes: nx.DiGraph.nodes,
    current_node: int,
    end_node: int,
    path: list,
    is_strongest: bool,
    strongest_path: list,
    visited: list,
    prob_treshold: float,
    edge_probability_ratio=2,
):
    """Recursive child for max_probability_walk."""
    visited.append(current_node)
    out_edges = list()
    for (u, v, d) in G.out_edges(current_node, data=True):
        d["probability"] = (
            nodes[v]["probability"] + (d["probability"] * edge_probability_ratio)
        ) / 2

        if d["probability"] > prob_treshold and v not in visited:
            out_edges.append((u, v, d))

    out_edges.sort(key=lambda x: x[2]["probability"], reverse=True)

    for idx, (u, v, d) in enumerate(out_edges):
        new_path = deepcopy(path)
        new_path.append(v)

        if is_strongest:
            strongest_path.append(v)

        if v == end_node:
            return new_path

        result = _exhaustive_probability_walk(
            G,
            nodes,
            v,
            end_node,
            new_path,
            not idx,
            strongest_path,
            visited,
            prob_treshold,
        )

        if result:
            return result

    return False


def exhaustive_probability_walk(G: nx.DiGraph, prob_treshold: float):
    """Takes an output graph with a start and end node, outputs the nodes path
    prioritizing highest probability.

    Args:
        G: Output Graph
        start_node int: Start node.
        end_node int: End node.
        prob_treshold: float (0,1]
    Returns:
        is_complete, node_path: list of nodes or False if there is no path.
    """

    nodes = G.nodes(data=True)
    start_node = -1
    end_node = -1

    for (n, d) in nodes:
        if d["is_start"]:
            start_node = n
        elif d["is_end"]:
            end_node = n

        if start_node != -1 and end_node != -1:
            break

    strongest_path = [start_node]  # in case incomplete
    visited = []

    path = _exhaustive_probability_walk(
        G,
        nodes,
        start_node,
        end_node,
        [start_node],
        True,
        strongest_path,
        visited,
        prob_treshold,
    )

    is_complete = path and path[-1] == end_node

    return is_complete, path if is_complete else strongest_path


def apply_path_on_graph(G: nx.DiGraph, path: list, require_clean: bool):
    """Apply is_in_path on graph using the path list.

    Args:
        G: Graph
        path: list of nodes
        require_clean: set other nodes and edges as False (slower)
    Returns:
        applied_graph, total_edge_weights
    """

    nodes = G.nodes(data=True)

    if require_clean:
        for (n, d) in nodes:
            d["is_in_path"] = False
        for (u, v, d) in G.edges(data=True):
            d["is_in_path"] = False

    edge_weights = 0.0

    for idx, node in enumerate(path):
        nodes[node]["is_in_path"] = True

        if idx + 1 < len(path):
            edge = G.get_edge_data(node, path[idx + 1])
            edge["is_in_path"] = True
            edge_weights += edge["weight"]

    return G, edge_weights
