from dijkstra_algorithm import *
import networkx as nx

def main():
    # build your graph
    # call your dijkstra function
    print("a coffee")
    G = nx.path_graph(5)
    print(dijkstra_path(G, 0, 4))


if __name__ == "__main__":
    main()