from dijkstra_algorithm import *
import networkx as nx

#holds all SP from all source node runs.
memoTable = {}

#i took out the check before adding as the table itself only holds unique values so it wasn't needed
def addItemToMemoTable( startNode, item ):
    keyList = item.keys()
    for itemKey in keyList:
        memoTable[(startNode, itemKey)] = item[itemKey]
       # print("adding something.. ")

#memoization table search function
def searchMemoTable(startNode, endNode):
    if memoTable.get((startNode, endNode)):
        print("shortest path:")
        print(memoTable[(startNode, endNode)])

#runs the shortest path djikstra algorithm, brings back lengths for all (unused at this point)
# and all paths which is printed and then added to the memoization table.
def shortestPath(G, startNode, endNode, weight="weight"):
    if searchMemoTable(startNode, endNode):
        print(memoTable[(startNode,endNode)])
    else:
        allLengthsFromSource, allPathsFromSource = single_source_dijkstra(G, startNode, None, None, weight)
        print(allPathsFromSource[endNode])
        addItemToMemoTable(startNode, allPathsFromSource)
        #searchMemoTable(startNode, endNode)

#TODO: need to add in a comparison to the unchanged SP run as well as some analysis.
def main():
    # build graph
    G = nx.path_graph(10)
    #tests here
    shortestPath(G, 1, 4)
    shortestPath(G, 1, 8)
    shortestPath(G, 2, 9)
    shortestPath(G, 3, 5)
    #check memoization table
    #print(memoTable)

####
"""
    if searchMemoTable(0, 4):
        print(memoTable[(0, 4)])
    # call dijkstra function
    mydict = single_source_dijkstra_path(G, 0, 4)
    print(mydict)
    addItemToMemoTable(0, mydict)

    print("search of memotable:")
    if searchMemoTable(0, 3):
        print(memoTable[(0, 3)])

    print("mydict (return from networkx djikstra):")
    print(mydict)
    print("memoTable:")
    print(memoTable)
    print("Edges in Graph:")
    print(G.edges)
    #print("dijkstra_path:"),
    #print(dijkstra_path(G, 0, 4))
    #print("dijkstra_path_length:"),
    #print(dijkstra_path_length(G, 0, 4))
    #print("bidirectional_dijkstra:"),
    #print(bidirectional_dijkstra(G, 0, 4))
    #print("single_source_dijkstra:"),
    #print(single_source_dijkstra(G, 0, 4))
    #print("single_source_dijkstra_path:"),
    #print(single_source_dijkstra_path(G, 0, 4)
   # mydict = single_source_dijkstra_path(G, 0, 4)
    #addItemToMemoTable(0, mydict)
    #print(mydict)
    #print(memoTable)
    #print("single_source_dijkstra_path_length:"),
    #print(single_source_dijkstra_path_length(G, 0, 4))
    #print("multi_source_dijkstra:"),
   # print(multi_source_dijkstra(G, 0))
    #print("multi_source_dijkstra_path:"),
   # print(multi_source_dijkstra_path(G, 0))
    #print("multi_source_dijkstra_path_length:"),
   # print(multi_source_dijkstra_path_length(G, 0))
    #print("all_pairs_dijkstra:"),
    #(x,y) = all_pairs_dijkstra(G)
    #print(x)
    #print(y)
    #print("all_pairs_dijkstra_path:"),
    #print(all_pairs_dijkstra_path(G))
    #print("all_pairs_dijkstra_path_length:"),
    #length = dict(nx.all_pairs_dijkstra_path_length(G))
        #for node in [0, 1, 2, 3, 4]:
         #   print(f"1 - {node}: {length[1][node]}")
    #print("dijkstra_predecessor_and_distance:")
  #  print(dijkstra_predecessor_and_distance(G))
"""
#TODO: experiments Is it faster to fun all_pairs first to fill memoization table and then just search for results?
#TODO: check other algorithms for possible faster run though I think they all start at the same algorithm..
#TODO: change save strategies for table: Use deduction to save more SPs per run, Subtables/smaller search areas (George
#referenced as possible option)
#TODO: save a smaller set of graphs, try to link them together for the search, use a non-key strategy for the table
#TODO: set weights
#TODO: sinlglesource dijkstra just save previous node, not node lists.


if __name__ == "__main__":
    main()