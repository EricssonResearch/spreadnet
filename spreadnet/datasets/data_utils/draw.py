import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import math
from scipy import signal

matplotlib.use("Agg")


def get_node_color(node):
    if node[1]["is_start"]:
        return "#FFF3B8"
    if node[1]["is_end"]:
        return "#90EE90"
    if node[1]["is_in_path"]:
        return "#C9FFD8"
    return "#D3D3D3"


def draw_networkx(
    title,
    figure,
    graph,
    plot_index,
    num_graphs_to_draw,
    node_label_key="default",
    edge_label_key="weight",
):
    """Draw networkx graph to figure.

    Args:
        figure: matplotlib figure
        graph: networkx graph
        plot_index: subplot index to be drawn on the figure
        num_graphs_to_draw: total number of plots

    Returns:
        None
    """

    ax = figure.add_subplot(
        math.ceil(num_graphs_to_draw / 5),
        num_graphs_to_draw if num_graphs_to_draw <= 5 else 5,
        plot_index,
    )
    ax.set_title(title)
    info = [
        Line2D([0], [0], color="#FFF3B8", lw=4),
        Line2D([0], [0], color="#90EE90", lw=4),
        Line2D([0], [0], color="#C9FFD8", lw=4),
        Line2D([0], [0], color="r", lw=4),
        Line2D([0], [0], color="b", lw=4),
    ]
    ax.legend(
        info,
        [
            "Start Node",
            "Destination Node",
            "Path Node",
            "Path Edge",
            "Discarded prediction probability",
        ],
    )

    path_edges = list()
    normal_edges = list()

    highlight_edge_labels = dict()
    path_edge_labels = dict()
    normal_edge_labels = dict()

    for (s, e, d) in graph.edges(data=True):
        label = round(d[edge_label_key], 2)

        if d["is_in_path"]:
            path_edges.append((s, e))
            path_edge_labels[(s, e)] = label
        elif (e, s) not in path_edges:
            normal_edges.append((s, e))

            if edge_label_key != "weight" and label > 0.00:
                highlight_edge_labels[(s, e)] = label
            else:
                normal_edge_labels[(s, e)] = label

    node_colors = list(map(lambda node: get_node_color(node), graph.nodes(data=True)))

    normal_node_labels = None
    highlight_node_labels = None

    if node_label_key != "default":
        normal_node_labels = dict()
        highlight_node_labels = dict()
        for n, d in graph.nodes(data=True):
            label = round(d[node_label_key], 2)

            if d["is_in_path"] is False and label > 0.00:
                highlight_node_labels[n] = label
            else:
                normal_node_labels[n] = label

    pos = nx.get_node_attributes(graph, "pos")

    nx.draw_networkx_nodes(
        graph, pos, cmap=plt.get_cmap("jet"), node_color=node_colors, node_size=500
    )
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, arrows=True)
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=path_edges,
        edge_color="r",
        arrows=True,
        arrowsize=25,
        width=3,
    )

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=normal_edge_labels)
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=highlight_edge_labels, font_color="blue"
    )
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=path_edge_labels, font_color="r"
    )
    nx.draw_networkx_labels(graph, pos, labels=normal_node_labels, font_color="black")

    if node_label_key != "default":
        nx.draw_networkx_labels(
            graph, pos, labels=highlight_node_labels, font_color="blue"
        )


def get_line_color(mode, type):
    if type == "nodes" and mode == "training":  # Dark blue
        return "#4682B4"
    if type == "nodes" and mode == "validation":  # Dark red
        return "#C60C30"
    if type == "edges" and mode == "training":  # Light blue
        return "#99D8F2"
    if type == "edges" and mode == "validation":  # Light red
        return "#FF8A8F"


def plot_training_graph(
    steps_curve,
    losses_curve,
    validation_losses_curve,
    accuracies_curve,
    validation_accuracies_curve,
    save_path,
    separate_training_testing=False,
    smoooth_window_half_width=3,
):
    """Plot training losses and accuracies.

    Args:
        steps_curve: epoch iteration
        losses_curve: nodes and edges losses on training set
        validation_losses_curve: nodes and edges losses on validation set
        accuracies_curve: nodes and edges accuracies on training set
        validation_accuracies_curve: nodes and edges accuracies on validation set
        save_path: folder and file name
        separate_training_testing: separate training and testing graph or merge
        smoooth_window_half_width: smoothen plot lines, the higher the smoother

    Returns:
        None
    """

    if separate_training_testing:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, metric, data_list in zip(
            [axes[0][0], axes[0][1], axes[1][0], axes[1][1]],
            [
                "Training Loss",
                "Training Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ],
            [
                losses_curve,
                accuracies_curve,
                validation_losses_curve,
                validation_accuracies_curve,
            ],
        ):
            for k in ["edges", "nodes"]:
                x = steps_curve
                y = [d[k] for d in data_list]

                window = signal.triang(1 + 2 * smoooth_window_half_width)
                window /= window.sum()

                y = signal.convolve(y, window, mode="valid")
                x = signal.convolve(x, window, mode="valid")

                ax.plot(
                    x, y, label=k, color=get_line_color(metric.split(" ")[0].lower(), k)
                )

            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.set_xlabel("Training Iteration")
            ax.legend()

            axes[0][0].set_yscale("log")
            axes[1][0].set_yscale("log")
            axes[0][1].margins(0.02, 0.02)
            axes[1][1].margins(0.02, 0.02)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, metric, data_list in zip(
            axes,
            [
                "Loss",
                "Accuracy",
            ],
            [
                [losses_curve, validation_losses_curve],
                [accuracies_curve, validation_accuracies_curve],
            ],
        ):
            for k in ["edges", "nodes"]:
                x = steps_curve
                y0 = [d[k] for d in data_list[0]]
                y1 = [d[k] for d in data_list[1]]

                window = signal.triang(1 + 2 * smoooth_window_half_width)
                window /= window.sum()

                y0 = signal.convolve(y0, window, mode="valid")
                y1 = signal.convolve(y1, window, mode="valid")
                x = signal.convolve(x, window, mode="valid")

                ax.plot(x, y0, label="Train " + k, color=get_line_color("training", k))
                ax.plot(
                    x,
                    y1,
                    label="Validation " + k,
                    color=get_line_color("validation", k),
                )

            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.set_xlabel("Training Iteration")
            ax.legend()

            axes[0].set_yscale("log")
            axes[1].margins(0.02, 0.02)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(save_path)
