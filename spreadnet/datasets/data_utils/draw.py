import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import math

matplotlib.use("Agg")


def get_node_color(node):
    return (
        "#FFF3B8"
        if node[1]["is_start"]
        else "#90EE90"
        if node[1]["is_end"]
        else "#C9FFD8"
        if node[1]["is_in_path"]
        else "#D3D3D3"
    )


def draw_networkx(figure, graph, plot_index, num_graphs_to_draw):
    """Draw networkx graph to figure.

    Args:
        figure: matplotlib figure
        graph: networkx graph
        plot_index: subplot index to be drawn on the figure
        num_graphs_to_draw: total number of plots

    Returns:
        None
    """

    highlight_edges = list()
    normal_edges = list()

    for (s, e, d) in graph.edges(data=True):
        if d["is_in_path"]:
            highlight_edges.append((s, e))
        elif (e, s) not in highlight_edges:
            normal_edges.append((s, e))

    pos = nx.spring_layout(graph)

    figure.add_subplot(
        math.ceil(num_graphs_to_draw / 5),
        num_graphs_to_draw if num_graphs_to_draw <= 5 else 5,
        plot_index,
    )

    values = list(map(lambda node: get_node_color(node), graph.nodes(data=True)))
    nx.draw_networkx_nodes(
        graph, pos, cmap=plt.get_cmap("jet"), node_color=values, node_size=500
    )
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, arrows=True)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=highlight_edges,
        edge_color="r",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )
    edge_labels = dict(
        [
            (
                (
                    u,
                    v,
                ),
                round(d["weight"], 2),
            )
            for u, v, d in graph.edges(data=True)
        ]
    )
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
