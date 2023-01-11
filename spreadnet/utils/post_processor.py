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
    force_my_way_if_no_path_found=True,
    edge_probability_ratio=1,
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
        visited = [start_node]
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

            if v not in visited and (
                prob > prob_threshold or force_my_way_if_no_path_found
            ):
                new_path = list(path)
                visited.append(v)
                new_path.append(v)
                heappush(queue, (1 - prob, new_path, is_strong and prob > 0.5))

    return False, strongest_path


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
    # clear all prev paths and weights from graph
    if require_clean:
        for (n, d) in nodes:
            d["is_in_path"] = False
        for (u, v, d) in G.edges(data=True):
            d["is_in_path"] = False

    edge_weights = 0.0

    # assign only the given nodes and edges in path to a clean graph
    for idx, node in enumerate(path):
        nodes[node]["is_in_path"] = True

        if idx + 1 < len(path):
            edge = G.get_edge_data(node, path[idx + 1])
            edge["is_in_path"] = True
            edge_weights += edge["weight"]

    return G, edge_weights


def extract_path(G):
    """Extracts nodes and edges in the graph as per probabilities predicted by
    GNN.

    Args:
        G (nx.DiGraph): graph to operate on
    Returns:
        sp_path: nodes that are in path
        sp_edges: edges that are in path
    """
    sp_path = []
    sp_path_edges = []

    for i in range(0, len(G.nodes)):
        if G.nodes[i]["is_in_path"]:
            sp_path.append(i)

    edges_list = list(G.edges(data=True))

    for e in edges_list:
        if e[2]["is_in_path"]:
            sp_path_edges.append([e[0], e[1]])

    return sp_path, sp_path_edges


def look_ahead_without_potentials(G, node_idx):
    """For a given node in graph, look ahead at all out-going edges and returns
    a list based on their connection status.

    Potential nodes and edges are not
    taken into account, instead counted as point of disconnect.
    Args:
        G (nx.DiGraph): graph to operate on
        node_idx (int): index for the node in G
    Returns:
        in_cont_path (list(int,int,str)): list of edges from current node
    """

    in_cont_path = []
    node_out_edges = list(G.out_edges(node_idx, data=True))

    for u, v, data in node_out_edges:

        e_frm_u = data["is_in_path"]
        next_node_v = G.nodes[v]["is_in_path"]

        if G.nodes[v]["is_end"]:
            in_cont_path.append(tuple((u, v, "end_is_connected")))
            break
        else:

            if e_frm_u and next_node_v:
                # confirmed link between u,v where uv are also in path
                in_cont_path.append(tuple((u, v, "is_connected")))

            elif not e_frm_u and not next_node_v:
                # neither edge or node is in path; not required
                in_cont_path.append(tuple((u, v, "is_disconnected")))

    return in_cont_path


def path_continuos_traversal(G, start_node_idx):
    """Traverses a simple path based on probabilities in a given graph G;
    starting form given node start_node_idx.

    This node can be any node in graph, but the traversal will only
    look ahead for nodes and edges that are "is_in_path"
    Args:
            G (nx.DiGraph): graph to operate on
            start_node_idx (int): index for the node in G
    Returns:
            visited (list[int]): ordered nodes from start node index
            disconnected (list(int)): nodes which have no in_path
            nodes or edges; broken path
    """
    visited = [start_node_idx]
    next_visit = visited[-1]

    disconnected = []
    flag = True

    while flag and G.nodes[next_visit]["is_in_path"]:
        in_cont_path = look_ahead_without_potentials(G, next_visit)
        for u, v, status in in_cont_path:
            # check_v_in_path = G.nodes[v]["is_in_path"]
            if status == "is_connected" or status == "end_is_connected":
                visited.append(v)
                next_visit = visited[-1]
                flag = True
                break

        else:
            if status == "is_disconnected":
                if G.nodes[u]["is_end"]:
                    print(f"{u} end node")
                else:
                    disconnected.append(u)

            break

    return visited, disconnected


def eval_path(G):
    """Evaluate/walk the path from starting node until a disconnected node is
    encountered.

    Args:
       G (nx.DiGraph): graph to operate on
    Returns:
        start_node: given start node in G
        end_node: given end node in G
        visited (list[int]): list of node indexes that
        are visited continuously until disconnect
    """
    G_cpy = deepcopy(G)
    # Separate start and end node indexes for further computation
    G_nodes, G_edges = extract_path(G_cpy)

    for (node, data) in G_cpy.nodes(data=True):
        if data["is_start"]:
            start_node_idx = node

        if data["is_end"]:
            end_node_idx = node

    if start_node_idx == end_node_idx:
        print("invalid path")
    else:

        visited, disconnected = path_continuos_traversal(G_cpy, start_node_idx)

    return start_node_idx, end_node_idx, visited


def hybrid_complete_path(G):
    """Complete path from last known node from visited chain of nodes in
    evaluation step.

    Args:
        G (nx.DiGraph): graph to operate on
        end_node_idx (int): given end node index
        visited (list[int]): list of node indexes
        that are visited continuously until disconnect
    Return:
        completed_pth (list[int]): completed path with dijkstra
    """
    G_cpy = deepcopy(G)

    start_node_idx, end_node_idx, visited = eval_path(G_cpy)

    remaining_path = nx.shortest_path(G, source=visited[-1], target=end_node_idx)
    completed_pth = [*visited, *remaining_path[1:]]

    # print(nx.is_path(G,completed_pth))
    return completed_pth
