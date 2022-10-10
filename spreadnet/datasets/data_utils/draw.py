import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy import signal

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


def plot_training_graph(
    steps_curve, losses_curve, accuracies_curve, save_path, smoooth_window_half_width=3
):
    """Plot training losses and accuracies.

    Args:
        steps_curve: epoch iteration
        losses_curve: nodes and edges losses
        accuracies_curve: nodes and edges accuracies
        save_path: folder and file name
        smoooth_window_half_width: smoothen plot lines, the higher the smoother

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
    for ax, metric, data_list in zip(
        axes, ["Loss", "Accuracy"], [losses_curve, accuracies_curve]
    ):
        for k in ["edges", "nodes"]:
            x = steps_curve
            y = [d[k] for d in data_list]

            window = signal.triang(1 + 2 * smoooth_window_half_width)
            window /= window.sum()
            y = list(map(lambda p: p.detach().cpu().numpy(), y))

            y = signal.convolve(y, window, mode="valid")
            x = signal.convolve(x, window, mode="valid")

            ax.plot(x, y, label=k)

        ax.set_ylabel(metric)
        ax.set_xlabel("Training Iteration")

        axes[0].set_yscale("log")
        # axes[0].set_ylim(0.00801, 4.5)
        # axes[1].set_ylim(0.79, 1.01)
        axes[0].legend()
        plt.savefig(save_path)