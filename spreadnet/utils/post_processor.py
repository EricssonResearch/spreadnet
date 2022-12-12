import networkx as nx
from copy import deepcopy
from torch.nn.functional import softmax
from heapq import heappop, heappush


def swap_start_end(graph_nx: nx.DiGraph, start: int, end: int):
    """Swap start and end node for bidirectional inference.

    :param graph_nx: networkx graph to predict

    :return: swapped graph
    """

    nodes = graph_nx.nodes(data=True)
    nodes[start]["is_end"] = True
    nodes[start]["is_start"] = False
    nodes[end]["is_start"] = True
    nodes[end]["is_end"] = False

    return graph_nx.reverse()


def process_prediction(input_graph_nx: nx.DiGraph, preds, in_path_threshold=0.5):
    """Construct networkx graph from prediction results.

    Args:
        input_graph_nx: original graph
        preds: nodes and edges prediction

    Returns:
        (predicted, truth_total_edge_weight)
    """

    pred_graph_nx = deepcopy(input_graph_nx)
    node_pred = preds["nodes"].cpu().detach()
    edge_pred = preds["edges"].cpu().detach()

    truth_total_weight = 0.0

    for i, (n, data) in enumerate(pred_graph_nx.nodes(data=True)):
        probability = softmax(node_pred[i], dim=-1).numpy()[1]
        data["probability"] = probability
        data["is_in_path"] = bool(probability > in_path_threshold)

    for i, (u, v, data) in enumerate(pred_graph_nx.edges(data=True)):
        if data["is_in_path"]:
            truth_total_weight += data["weight"]

        probability = softmax(edge_pred[i], dim=-1).numpy()[1]
        data["probability"] = probability
        data["is_in_path"] = bool(probability > in_path_threshold)

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


def get_start_end_nodes(nodes: nx.DiGraph.nodes):
    start_node = -1
    end_node = -1

    for (n, d) in nodes:
        if d["is_start"]:
            start_node = n
        elif d["is_end"]:
            end_node = n

        if start_node != -1 and end_node != -1:
            return (start_node, end_node)


def _exhaustive_probability_walk(
    G: nx.DiGraph,
    nodes: nx.DiGraph.nodes,
    current_node: int,
    end_node: int,
    path: list,
    is_strongest: bool,
    strongest_path: list,
    visited: list,
    prob_threshold: float,
    edge_probability_ratio=2,
):
    """Recursive child for max_probability_walk."""
    visited.append(current_node)
    out_edges = list()
    for (u, v, d) in G.out_edges(current_node, data=True):
        d["probability"] = (
            nodes[v]["probability"] + (d["probability"] * edge_probability_ratio)
        ) / 2

        if d["probability"] > prob_threshold and v not in visited:
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
            prob_threshold,
        )

        if result:
            return result

    return False


def exhaustive_probability_walk(
    G: nx.DiGraph, start_node: float, end_node: float, prob_threshold=0.1
):
    """Takes an output graph with a start and end node, outputs the nodes path
    prioritizing highest probability.

    Args:
        G: Output Graph
        start_node int: Start node.
        end_node int: End node.
        prob_threshold: float (0,1]
    Returns:
        is_complete, node_path: list of nodes or False if there is no path.
    """
    nodes = G.nodes(data=True)
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
        prob_threshold,
    )

    is_complete = path and path[-1] == end_node
    final_path = path if is_complete else strongest_path

    return is_complete, final_path


def probability_first_search(
    G: nx.DiGraph,
    start_node: float,
    end_node: float,
    visited=False,
    prob_threshold=0.1,
    edge_probability_ratio=1,
    force_my_way_if_no_path_found=True,
):
    """Takes an output graph with a start and end node, outputs the nodes path
    prioritizing highest probability. Hybrid of BFS and DFS.

    Args:
        G: Output Graph
        start_node int: Start node.
        end_node int: End node.
        prob_threshold: float (0,1]
        edge_probability_ratio: ratio compared to node's probability
    Returns:
        is_complete, node_path: list of nodes or False if there is no path.
    """
    nodes = G.nodes(data=True)
    if not visited:
        visited = []
    strongest_path = [start_node]  # in case incomplete
    queue = []
    heappush(queue, (0, [start_node], True))  # (priority, path, is_strong)

    while queue:
        heap_item = heappop(queue)
        path = heap_item[1]
        is_strong = heap_item[2]
        node = path[-1]

        if node == end_node:
            return True, path

        if is_strong and len(path) > len(strongest_path):
            strongest_path = list(path)

        for (_, v, d) in G.out_edges(node, data=True):
            prob = (
                nodes[v]["probability"] + (d["probability"] * edge_probability_ratio)
            ) / 2
            if v not in visited and prob > prob_threshold:
                new_path = list(path)
                visited.append(v)
                new_path.append(v)
                heappush(queue, (1 - prob, new_path, is_strong and prob > 0.5))

    if len(visited) == G.number_of_nodes or not force_my_way_if_no_path_found:
        return False, strongest_path

    (c_is_complete, c_node_path) = probability_first_search(
        G, strongest_path.pop(), end_node, visited, -1
    )
    strongest_path.extend(c_node_path)
    return c_is_complete, strongest_path


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
