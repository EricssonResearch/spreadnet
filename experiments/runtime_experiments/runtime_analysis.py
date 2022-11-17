import pandas as pd
import matplotlib.pyplot as plt

# pd.options.plotting.backend = "plotly"


"""
These tests will only show correct data with 20 graph sets
(it has been hard coded, so this value will need to be changed
for variations on this or an update to set the value
based off counted runs.)

The graphs will print from the data in the runtime_test_results.csv which is
created after completing runtime_experiments.py
"""


def line_graph_pandas_dijkstra_gnn_runtime(df):
    df.plot(
        x="min_nodes",
        y=[
            "Dijkstra Full Path and Memoization Runtime",
            "Dijkstra With Memoization Runtime",
            "Message Passing GNN Runtime",
        ],
        figsize=(10, 5),
        grid=True,
        style=".-",
    )
    plt.title("Run Time for 20 Graph Sets")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


def line_graph_pandas_dijkstra_gnn_walltime(df):
    df.plot(
        x="min_nodes",
        y=[
            "Dijkstra Full Path and Memoization Walltime",
            "Dijkstra With Memoization Walltime",
            "Message Passing GNN Walltime",
        ],
        figsize=(10, 5),
        grid=True,
        style=".-",
    )
    plt.title("Run Time for 20 Graph Sets")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


def line_graph_pandas_dijkstra_versions_walltime(df):
    df.plot(
        x="min_nodes",
        y=[
            "Dijkstra Full Path and Memoization Walltime",
            "Dijkstra With Memoization Walltime",
            "Networkx Dijkstra Walltime",
        ],
        figsize=(10, 5),
        grid=True,
        style=".-",
    )
    plt.title("Run Time for 20 Graph Sets")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


def line_graph_pandas_dijkstra_versions_runtime(df):
    df.plot(
        x="min_nodes",
        y=[
            "Dijkstra Full Path and Memoization Runtime",
            "Dijkstra With Memoization Runtime",
            "Networkx Dijkstra Runtime",
        ],
        figsize=(10, 5),
        grid=True,
        style=".-",
    )
    plt.title("Run Time for 20 Graph Sets")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    plt.show()


"""
#TODO create plotly graphs
def linear_regression_plotly(df):
    df = px.data.tips()
    fig = px.scatter(df, x="total_bill", y="tip", trendline="ols")
    fig.show()
"""


def main():
    data = pd.read_csv("runtime_test_results.csv")
    df = pd.DataFrame(data)
    df = df.sort_values(by=["min_nodes"])
    line_graph_pandas_dijkstra_gnn_runtime(df)
    line_graph_pandas_dijkstra_gnn_walltime(df)
    line_graph_pandas_dijkstra_versions_runtime(df)
    line_graph_pandas_dijkstra_versions_walltime(df)

    print(df)

    # barplot = df.plot(x='min_nodes', columns=['dijkstra_msp_memo',
    # 'dijkstra_memo', 'Message Passing GNN'])

    # plt.show()
    # index_col='min_nodes'
    # df.plot()
    # data = data.head()
    # df = pd.DataFrame(data, columns=)
    # plt.figure(figsize=(16, 8), dpi=150)
    # dijkstra_msp_memo_runtime = df['dijkstra_msp_memo']
    # .plot(label='Dijkstra_MSP_memo', color='orange')
    # df['dijkstra_memo'].plot(label='Dijkstra_Shortest_Path',color = 'purple')

    # df.plot(
    #    x="min_nodes", y=["dijkstra_msp_memo", "dijkstra_memo", "Message Passing GNN"]
    # )
    # plt.legend()
    return


if __name__ == "__main__":
    main()
