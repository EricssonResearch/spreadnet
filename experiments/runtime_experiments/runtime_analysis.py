import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("runtime_test_results.csv")
    df = pd.DataFrame(data)
    df = df.sort_values(by=["min_nodes"])

    print(df)
    df.plot(
        x="min_nodes",
        y=["dijkstra_msp_memo", "dijkstra_memo", "Message Passing GNN"],
        figsize=(10, 5),
        grid=True,
        style=".-",
    )
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
    plt.title("Run Time for 20 Graphs")
    plt.xlabel("Minimum Number of Nodes in Set")
    plt.ylabel("Seconds")
    # df.plot(
    #    x="min_nodes", y=["dijkstra_msp_memo", "dijkstra_memo", "Message Passing GNN"]
    # )
    # plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
