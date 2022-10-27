import json
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from spreadnet.utils.max_prob_path_utils import MaxProbWalk

# IDEa: we could save the path length also when we do the accuracy.
# Currently for each graph we can deduce the minimum path length.


def plot_graph(df, pb_treshold, model_name):
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
    plt.yticks(np.arange(0, df_tr["Accuracy"].max() + 0.1, 0.01))
    plt.legend(df["Probabiltiy Threshold"], title="Probability Threshold")
    plt.xlabel("Grapph Size")
    plt.ylabel("Accuracy")
    plt.title("In path prediction over all nodes" + model_name)
    plt.grid(visible=True)
    plt.show()


def prob_plot():
    """Plot the accuracies for each prob threshold."""
    df = pd.read_csv("results.csv", index_col=False)
    df = df.sort_values(["Graph Size", "Model Type"])
    pd.set_option("display.max_rows", None)
    # print(df)
    # pb_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
    pb_treshold = [0.5, 0.25, 0.01]
    df_pyg = df[df["Model Type"] == "pyg"]
    df_tf = df[df["Model Type"] == "tf"]
    plot_graph(df_pyg, pb_treshold=pb_treshold, model_name="Pyg Original")
    plot_graph(df_tf, pb_treshold=pb_treshold, model_name="Tensorflow Original")


def prob_accuracy():
    pred_dir = "increasing_size_predictions"
    datasets = list()

    # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
    prob_treshold = [0.5, 0.25, 0.01]
    for path in os.listdir(pred_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(pred_dir, path)):
            datasets.append(path)
    node_accuracy = []
    for ds in datasets:
        raw_data_path = pred_dir + "/" + ds
        file_raw = open(raw_data_path)

        graphs = json.load(file_raw)

        for pt in prob_treshold:

            probs = []
            pred = []
            for g in graphs:
                g = nx.node_link_graph(g)  # TODO add probability for the edges also
                for i in range(0, g.number_of_nodes()):
                    prob = np.round(tf.nn.softmax(g.nodes[i]["logits"])[1].numpy(), 2)
                    if prob >= pt and g.nodes[i]["is_in_path"]:
                        # print(prob, pt, g.nodes[i]["is_in_path"])
                        probs.append(prob)
                        pred.append(1)
                    else:
                        pred.append(0)
                # print(ds, pt, sum(pred) / len(pred))
            # print(ds, pt, sum(pred) / len(pred))

            node_accuracy.append(sum(pred) / len(pred))
    df = pd.DataFrame()

    prob_treshold_df = []
    model_type = []
    g_size = []

    for ds in datasets:
        for pt in prob_treshold:

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

    df["Accuracy"] = node_accuracy

    df.to_csv("results.csv", index=False)


# def max_prob_path_accuracy():
#     """Needs to have the max prob path walk updated so that it can be used
#     outside of the graph visualization.
#     """
#     pred_dir = "increasing_size_predictions"
#     datasets = list()

#     # prob_treshold = [0.5, 0.45, 0.4, 0.35, 0.30, 0.25, 0.2, 0.15, 0.1, 0.05]
#     prob_treshold = [0.5, 0.25, 0.01]
#     for path in os.listdir(pred_dir):
#         # check if current path is a file
#         if os.path.isfile(os.path.join(pred_dir, path)):
#             datasets.append(path)
#     mpw = MaxProbWalk()
#     for ds in datasets:
#         raw_data_path = pred_dir + "/" + ds
#         file_raw = open(raw_data_path)

#         graphs = json.load(file_raw)

#         for pt in prob_treshold:

#             for g in graphs:
#                 g = nx.node_link_graph(g)  # TODO add probability for the edges also
#                 mpw.max_probability_walk(
#                     G,
#                 )
#             node_accuracy.append(sum(pred) / len(pred))
#     mpw.max_probability_walk()


if __name__ == "__main__":
    # prob_accuracy()
    prob_plot()
