import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("runtime_test_results.csv")
    # df.plot()
    # df['dijkstra_msp_memo'].plot(label='Dijkstra_MSP_memo', color='orange',
    # index_col='min_nodes')
    # df['dijkstra_memo'].plot(label='Dijkstra_Shortest_Path',
    # color = 'purple', index_col='min_nodes')
    plt.xlabel("Minimum Number of Nodes")
    plt.ylabel("seconds")
    df.plot(
        x="min_nodes", y=["dijkstra_msp_memo", "dijkstra_memo", "Message Passing GNN"]
    )
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
