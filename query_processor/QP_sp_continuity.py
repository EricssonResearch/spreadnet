import networkx as nx
import json

# import matplotlib
import matplotlib.pyplot as plt
from spreadnet.datasets.data_utils.draw import draw_networkx


def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)

    return nx.node_link_graph(js_graph[0])


def extract_path(G):
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


def is_continuos(G):
    # find start and end nodes
    g_path_idx, g_edges_idx, g_pos = extract_path(G)

    is_start_index = 0
    is_end_index = 0

    detected = []
    missing_out_edge = []
    for i in g_path_idx:  # list of node indexes that are in path
        if G.nodes[i]["is_start"]:
            is_start_index = i

        if G.nodes[i]["is_end"]:
            is_end_index = i

        succ_list = list(G.successors(i))
        print(f"curr_node:{i}, successors:{succ_list}")
        out_from_curr = list(
            G.out_edges(i, data="is_in_path")
        )  # outgoing edges from curr node

        missing = 0
        for e in out_from_curr:
            if e[2]:  # for each outgoing edge e, 'is_in_path' is true
                print(f"{e} --> edge exits")
                detected.append(e)
                break

            else:
                missing = missing + 1
                if missing == len(succ_list):  # iterate over all successors until
                    if G.nodes[i]["is_end"] is False:
                        print(f"no outgoing edges from node {i} to successors")
                        missing_out_edge.append(i)

                    else:
                        print(f"end node reached:{i}")

                        break
                    print("\n\n")

    print("\n\n")
    print(f"nodes_in_path:{g_path_idx}")
    print(f"Start_node:{is_start_index}")
    print(f"End_node:{is_end_index}")
    print(f"detected out going edges from:{detected}")
    print(f"missing out going edge :{missing_out_edge}")


def draw_networkx_utils(graph, title="optional"):
    plot_size = 20

    fig = plt.figure(
        figsize=(
            plot_size,
            plot_size,
        )
    )
    draw_networkx(
        title,
        fig,
        graph,
        1,
        1,
    )

    # fig.tight_layout()
    # return fig


def draw_just_path(G):
    g_path, g_edges, g_pos = extract_path(G)
    fig_sp = plt.figure(figsize=(15, 15))

    nx.draw_networkx_nodes(G, pos=g_pos, nodelist=g_path, node_color="r", node_size=350)
    nx.draw_networkx_edges(
        G,
        pos=g_pos,
        edgelist=g_edges,
        width=2,
        edge_color="r",
    )

    return fig_sp
