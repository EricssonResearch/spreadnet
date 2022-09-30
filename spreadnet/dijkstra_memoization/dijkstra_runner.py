from dijkstra_algorithm import *
import networkx as nx

memoTable = {}

def addItemToMemoTable( startNode, item ):
    keyList = item.keys()
    for itemKey in keyList:
        if memoTable.get((startNode, itemKey)):
            print("already had one..")

        memoTable[(startNode, itemKey)] = item[itemKey]
        print("adding something.. ")


def main():
    # build graph
    #TODO: if not in memoization table call single_source_djikstra
    #check memoization table
    # call dijkstra function
    print("a cof")
    G = nx.path_graph(10)
    print(G.edges)
    print("dijkstra_path:"),
    print(dijkstra_path(G, 0, 4))
    print("dijkstra_path_length:"),
    print(dijkstra_path_length(G, 0, 4))
    print("bidirectional_dijkstra:"),
    print(bidirectional_dijkstra(G, 0, 4))
    print("single_source_dijkstra:"),
    print(single_source_dijkstra(G, 0, 4))
    print("single_source_dijkstra_path:"),
    #print(single_source_dijkstra_path(G, 0, 4)
    mydict = single_source_dijkstra_path(G, 0, 4)
    addItemToMemoTable(0, mydict)
    print(mydict)
    print(memoTable)
    print("single_source_dijkstra_path_length:"),
    print(single_source_dijkstra_path_length(G, 0, 4))
    print("multi_source_dijkstra:"),
   # print(multi_source_dijkstra(G, 0))
    print("multi_source_dijkstra_path:"),
   # print(multi_source_dijkstra_path(G, 0))
    print("multi_source_dijkstra_path_length:"),
   # print(multi_source_dijkstra_path_length(G, 0))
    #print("all_pairs_dijkstra:"),
    #(x,y) = all_pairs_dijkstra(G)
    #print(x)
    #print(y)
    print("all_pairs_dijkstra_path:"),
    print(all_pairs_dijkstra_path(G))
    print("all_pairs_dijkstra_path_length:"),
    length = dict(nx.all_pairs_dijkstra_path_length(G))
        #for node in [0, 1, 2, 3, 4]:
         #   print(f"1 - {node}: {length[1][node]}")
    print("dijkstra_predecessor_and_distance:")
  #  print(dijkstra_predecessor_and_distance(G))



if __name__ == "__main__":
    main()