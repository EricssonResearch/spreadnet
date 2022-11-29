import networkx as nx
from copy import deepcopy


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


def _max_probability_walk(
    G: nx.DiGraph,
    nodes,
    current_node,
    end_node,
    path: list,
    visited: list,
    prob_treshold,
):
    """Recursive child for max_probability_walk."""
    visited.append(current_node)
    out_edges = list()
    for (u, v, d) in G.out_edges(current_node, data=True):
        d["probability"] += nodes[v]["probability"]
        d["probability"] /= 2

        if d["probability"] > prob_treshold and v not in visited:
            out_edges.append((u, v, d))

    out_edges.sort(key=lambda x: x[2]["probability"], reverse=True)

    for (u, v, d) in out_edges:
        new_path = deepcopy(path)
        new_path.append(v)

        if v == end_node:
            return new_path

        result = _max_probability_walk(
            G, nodes, v, end_node, new_path, visited, prob_treshold
        )

        if result:
            return result

    return False


def max_probability_walk(G: nx.DiGraph, prob_treshold: float):
    """Takes an output graph with a start and end node, outputs the nodes path
    prioritizing highest probability.

    Args:
        G: Output Graph
        start_node int: Start node.
        end_node int: End node.
        prob_treshold: float (0,1]
    Returns:
        node_path: list of nodes.

    Notes: The path can be incomplete.
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

    return _max_probability_walk(
        G, nodes, start_node, end_node, [start_node], [], prob_treshold
    )
