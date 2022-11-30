import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# pd.options.plotting.backend = "plotly"


"""
The graphs will print from the data in the runtime_test_results.csv which is
created after completing runtime_experiments.py
Available data in dataframe:
    "file_name": file_name_list,
    "min_nodes": min_node_list,
    "max_nodes": max_node_list,
    "Hashing Graphs RunTime": hashing_runtime_test_list,
    "Dijkstra Full Path and Memoization Runtime": dijksta_runtime_msp_memo_list,
    "Dijkstra With Memoization Runtime": dijkstra_runtime_memo_list,
    "Message Passing GNN Runtime": message_passing_runtime_list,
    "Networkx Dijkstra Runtime": networkx_dijkstra_runtime_list,
    "Hashing Graphs Walltime": hashing_runtime_test_list,
    "Dijkstra Full Path and Memoization Walltime": dijksta_runtime_msp_memo_list,
    "Dijkstra With Memoization Walltime": dijkstra_runtime_memo_list,
    "Message Passing GNN Walltime": message_passing_runtime_list,
    "Networkx Dijkstra Walltime": networkx_dijkstra_walltime_list,
    "Memotable Walltime": memotable_walltime,
    "Memotable Runtime": memotable_runtime,
"""


def bar_plot_gnn_dijkstra(df):
    # Convert the min nodes to the datatype category since the plot tool needs this.
    df["mn"] = df["min_nodes"].astype("category")

    # flatten out the data for the GNN runs
    d1 = df.explode("Message Passing GNN Runtime")
    # convert the GNN data to a float
    d1["GNN"] = d1["Message Passing GNN Runtime"].astype("float")

    # flatten out the data for the dijkstra runs
    d2 = df.explode("Dijkstra Full Path and Memoization Runtime")
    # Need to convert this to a float. csv probably imported it as a string.
    d2["Dijkstra_All_Paths_Memo"] = d2[
        "Dijkstra Full Path and Memoization Runtime"
    ].astype("float")

    # flatten out the data for the dijkstra runs
    d3 = df.explode("Dijkstra With Memoization Runtime")
    # Need to convert this to a float. csv probably imported it as a string.
    d3["Dijkstra_Single_Path_Memo"] = d3["Dijkstra With Memoization Runtime"].astype(
        "float"
    )

    cdf = pd.concat([d1, d2, d3])  # CONCATENATE
    mdf = pd.melt(
        cdf,
        id_vars=["mn"],
        value_vars=["GNN", "Dijkstra_All_Paths_Memo", "Dijkstra_Single_Path_Memo"],
        var_name=["Test Run"],
    )  # MELT

    ax = sns.boxplot(x="mn", y="value", hue="Test Run", data=mdf)  # RUN PLOT
    # Plotting box plots in log scale probably isn't so great but with
    # this you can at least see
    # the dijkstra run data.
    ax.set(yscale="log")

    # sns.boxplot(data=df.explode('Message Passing GNN Runtime'), x='min_nodes',
    #  y='Message Passing GNN Runtime', palette='magma')

    plt.title("Comparison of GNN Runtimes to Dijkstra Algorithms")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


def bar_plot_dijkstra_variants(df):
    # Convert the min nodes to the datatype category since the plot tool needs this.
    df["mn"] = df["min_nodes"].astype("category")

    # flatten out the data
    d1 = df.explode("Networkx Dijkstra Runtime")
    # convert data to a float
    d1["Network_x_Dijkstra"] = d1["Networkx Dijkstra Runtime"].astype("float")

    # flatten out the data
    d2 = df.explode("Dijkstra Full Path and Memoization Runtime")
    # Need to convert this to a float. csv imported it as a string.
    d2["Dijkstra_All_Paths_Memo"] = d2[
        "Dijkstra Full Path and Memoization Runtime"
    ].astype("float")

    # flatten out the data
    d3 = df.explode("Dijkstra With Memoization Runtime")
    # Need to convert this to a float. csv imported it as a string.
    d3["Dijkstra_Single_Path_Memo"] = d3["Dijkstra With Memoization Runtime"].astype(
        "float"
    )

    cdf = pd.concat([d1, d2, d3])  # CONCATENATE
    mdf = pd.melt(
        cdf,
        id_vars=["mn"],
        value_vars=[
            "Network_x_Dijkstra",
            "Dijkstra_All_Paths_Memo",
            "Dijkstra_Single_Path_Memo",
        ],
        var_name=["Test Run"],
    )  # MELT

    ax = sns.boxplot(x="mn", y="value", hue="Test Run", data=mdf)
    # Plotting box plots in log scale to see
    # the dijkstra run data.
    ax.set(yscale="log")

    plt.title("Comparison of Dijkstra Algorithms")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


# This is not working properly.
# Taking the time with the order run instead of the set. TODO: fix
def bar_plot_gnn_dijkstra_pandas(df):
    df["Dijkstra Full Path and Memoization Runtime"].apply(
        lambda x: pd.Series(x)
    ).boxplot()
    plt.title("Run Time")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


def main():
    data = pd.read_csv(
        "runtime_test_results.csv",
        converters={
            "Networkx Dijkstra Runtime": pd.eval,
            "Message Passing GNN Runtime": pd.eval,
            "Hashing Graphs RunTime": pd.eval,
            "Dijkstra Full Path and Memoization Runtime": pd.eval,
            "Dijkstra With Memoization Runtime": pd.eval,
        },
    )
    df = pd.DataFrame(data)
    df = df.sort_values(by=["min_nodes"])
    bar_plot_gnn_dijkstra(df)
    bar_plot_dijkstra_variants(df)

    return


if __name__ == "__main__":
    main()
