import json
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from spreadnet.utils.max_prob_path_utils import MaxProbWalk

# IDEa: we could save the path length also when we do the accuracy.
# Currently for each graph we can deduce the minimum path length.


def plot_graph(df, pb_treshold, model_name, title):
    df.drop(
        [
            "Model Type",
        ],
        axis=1,
        inplace=True,
    )

    for pb in pb_treshold:
        df_tr = df[df["Probabiltiy Threshold"] == pb]
        plt.scatter(df_tr["Graph Size"], df_tr["Accuracy"])
    plt.yticks(np.arange(0, df_tr["Accuracy"].max() + 0.1, 0.05))
    plt.legend(df["Probabiltiy Threshold"], title="Probability Threshold")
    plt.xlabel("Graph Size")
    plt.ylabel("Accuracy Nodes")
    plt.title(title + model_name)
    plt.grid(visible=True)
    plt.show()


def prob_plot(file_name, title):
    """Plot the accuracies for each prob threshold."""
    df = pd.read_csv(file_name, index_col=False)

    df = df.sort_values(["Graph Size", "Model Type"])
    pd.set_option("display.max_rows", None)
    # print(df)
    # pb_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
    pb_treshold = [0.5, 0.25, 0.01]
    df_pyg = df[df["Model Type"] == "pyg"]
    df_tf = df[df["Model Type"] == "tf"]
    plot_graph(df_pyg, pb_treshold=pb_treshold, model_name=" Pyg Original", title=title)
    plot_graph(
        df_tf, pb_treshold=pb_treshold, model_name=" Tensorflow Original", title=title
    )


def prob_accuracy(
    file_name,
    only_path=False,
):
    pred_dir = "increasing_size_predictions"
    datasets = list()

    # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
    prob_treshold = [0.5, 0.25, 0.01]
    for path in os.listdir(pred_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(pred_dir, path)):
            datasets.append(path)
    node_accuracy = []
    for ds in tqdm(datasets):
        raw_data_path = pred_dir + "/" + ds
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        for pt in prob_treshold:

            probs = []
            pred = []
            for g in graphs:
                g = nx.node_link_graph(g)  # TODO add probability for the edges also

                if only_path:
                    no_ground_truth_nodes = 0
                    for i in range(0, g.number_of_node):
                        if g.nodes[i]["is_in_path"]:
                            no_ground_truth_nodes += 1
                for i in range(0, g.number_of_nodes()):
                    prob = np.round(tf.nn.softmax(g.nodes[i]["logits"])[1].numpy(), 2)
                    if prob >= pt and g.nodes[i]["is_in_path"]:
                        # print(prob, pt, g.nodes[i]["is_in_path"])
                        probs.append(prob)
                        pred.append(1)
                    elif prob < pt and not g.nodes[i]["is_in_path"] and not only_path:
                        pred.append(1)
                    else:
                        pred.append(0)
                # print(ds, pt, sum(pred) / len(pred))
            # print(ds, pt, sum(pred) / len(pred))
            if only_path:
                node_accuracy.append(sum(pred) / no_ground_truth_nodes)
            else:
                node_accuracy.append(sum(pred) / len(pred))
    to_df_format(
        datasets,
        prob_threshold=prob_treshold,
        accuracy=node_accuracy,
        name=file_name,
    )


def to_df_format(datasets, prob_threshold, accuracy, name):
    df = pd.DataFrame()

    prob_treshold_df = []
    model_type = []
    g_size = []

    for ds in datasets:
        for pt in prob_threshold:

            prob_treshold_df.append(pt)
            # The following line is an abomination
            ds_clean = (
                ds.replace("increasing_size_", "")
                .replace(".json", "")
                .replace("_out", "")
            )
            split = ds_clean.split("_")
            model_type.append(split[(len(split) - 2)])
            g_size.append(split[1])
            # g_size.append((split[0], split[1]))

    df["Graph Size"] = g_size
    df["Model Type"] = model_type
    df["Probabiltiy Threshold"] = prob_treshold_df

    df["Accuracy"] = accuracy

    df.to_csv(name, index=False)


def max_prob_path_lengths():
    """Comapres the length of the path found to the ground truth length."""

    pred_dir = "increasing_size_predictions"
    datasets = list()

    # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
    prob_threshold = [0.5, 0.25, 0.01]
    for path in os.listdir(pred_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(pred_dir, path)):
            datasets.append(path)
    mpw = MaxProbWalk()
    accuracy_path_length = []
    for ds in tqdm(datasets):
        raw_data_path = pred_dir + "/" + ds
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        for pt in prob_threshold:
            paths_correct = []
            for g in graphs:
                g = nx.node_link_graph(g)  # TODO add probability for the edges also
                max_prob_walk_nodes, max_prob_walk_edges = mpw.max_probability_walk(
                    g, pt
                )

                start_node, end_node = mpw.get_start_end_nodes(g)

                nodes_in_path = 0
                for i in range(0, g.number_of_nodes()):
                    if g.nodes[i]["is_in_path"]:
                        nodes_in_path += 1

                len_walk = len(max_prob_walk_nodes)

                if (
                    len_walk == nodes_in_path
                    and max_prob_walk_nodes[len(max_prob_walk_nodes) - 1] == end_node
                ):
                    points = 1
                elif len_walk < nodes_in_path:
                    points = 0

                elif max_prob_walk_nodes[len(max_prob_walk_nodes) - 1] == end_node:
                    points = nodes_in_path / len_walk

                else:
                    points = 0

                paths_correct.append(points)
            accuracy_path_length.append(sum(paths_correct) / len(graphs))

    to_df_format(
        datasets=datasets,
        prob_threshold=prob_threshold,
        accuracy=accuracy_path_length,
        name="acc_prob_walk.csv",
    )


if __name__ == "__main__":
    prob_accuracy(only_path=False, file_name="all_nodes_acc.csv")
    prob_accuracy(only_path=True, file_name="only_path_nodes_acc.csv")
    max_prob_path_lengths()

    prob_plot("acc_prob_walk.csv", "Max Prob Walk")
    prob_plot("all_pred_accuracy.csv", "All Nodes Walk")
    prob_plot("only_path_nodes_acc.csv", "Only Path Nodes Walk")
