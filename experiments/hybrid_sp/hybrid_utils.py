# import matplotlib.pyplot as plt
import networkx as nx
import json

# from spreadnet.utils import post_processor

# from copy import deepcopy


def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)

    return nx.node_link_graph(js_graph[0])


def extract_path(G):
    """_summary_

    _extended_summary_

    Args:
        G (_type_): _description_

    Returns:
        _type_: _description_
    """
    sp_path = []
    sp_path_edges = []
    sp_pos = {}

    for i in range(0, G.number_of_nodes()):
        sp_pos[i] = G.nodes[i]["pos"]

    for i in range(0, len(G.nodes)):
        if G.nodes[i]["is_in_path"]:
            sp_path.append(i)

    edges_list = list(G.edges(data=True))

    for e in edges_list:
        if e[2]["is_in_path"]:
            sp_path_edges.append([e[0], e[1]])

    return sp_path, sp_path_edges, sp_pos


def look_ahead(G, node_idx):
    """For a given node in graph, look ahead at all out-going edges and returns
    a list based on their connection status.

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

            elif not e_frm_u and next_node_v:
                # missing link between u,v but v is in path;
                # hence this a potential edge
                in_cont_path.append(tuple((u, v, "is_potential_edge")))

            elif e_frm_u and not next_node_v:
                # confirmed link between u,v but v is not in path;
                # hence this is potential next node
                in_cont_path.append(tuple((u, v, "is_potential_node")))

            elif not e_frm_u and not next_node_v:
                # neither edge or node is in path; not required
                in_cont_path.append(tuple((u, v, "is_disconnected")))

    return in_cont_path


def path_continuos_traversal(G, start_node_idx):
    """Traverses a simple path based on probabilities in a given graph G;
    starting form given node start_node_idx. This node can be any node in
    graph, but the traversal will only look ahead for nodes and edges that are
    "is_in_path".

    Args:
        G (nx.DiGraph): graph to operate on

        start_node_idx (int): index for the node in G

    Returns:
        visited (list[int]): ordered nodes from start node index

        potential_node (list(int)):nodes which have an in_path
            incoming edge but node itself is not in_path

        potential_edge (list(int)): nodes which have no in_path
            incoming edge but are in_path

        disconnected (list(int)): nodes which have no
            in_path nodes or edges; broken path
    """
    visited = [start_node_idx]
    next_visit = visited[-1]

    potential_node = []
    potential_edge = []
    disconnected = []
    flag = True

    while flag and G.nodes[next_visit]["is_in_path"]:
        in_cont_path = look_ahead(G, next_visit)
        for u, v, status in in_cont_path:
            check_v_in_path = G.nodes[v]["is_in_path"]
            if status == "is_connected" or status == "end_is_connected":
                visited.append(v)
                next_visit = visited[-1]
                flag = True
                break
            else:
                if status == "is_potential_node":
                    potential_node.append(v)
                    next_visit = visited[-1]
                    flag = True
                    continue

                elif status == "is_potential_edge" and not check_v_in_path:
                    potential_edge.append(v)
                    next_visit = visited[-1]
                    flag = True
                    continue

        else:
            if status == "is_disconnected":
                disconnected.append(u)

            break

    return visited, potential_node, potential_edge, disconnected


def eval_path(G, node_idx):
    """_summary_

    _extended_summary_

    Args:
        G (_type_): _description_
        node_idx (_type_): _description_
    """
    visited, potential_node, potential_edge, disconnected = path_continuos_traversal(
        G, node_idx
    )
    G_nodes, G_edges, G_pos = extract_path(G)

    # evaluate visited
    if nx.is_simple_path(G, visited):
        print("is simple path")
        # TODO: 1. check if first node is start and last is end
        # 2. check if there are potentials to add
        # 3. do some checks to them and add in right places

    else:
        print("not a valid path")
