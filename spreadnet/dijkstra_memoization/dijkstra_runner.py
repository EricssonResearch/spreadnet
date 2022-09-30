from dijkstra_algorithm import *
import networkx as nx

#Holds all SP from all source node runs.
memoTable = {}

#Took out the check before adding as the table itself only holds unique values so it wasn't needed
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


#TODO: experiments Is it faster to fun all_pairs first to fill memoization table and then just search for results?
#TODO: check other algorithms for possible faster run though I think they all start at the same algorithm..
#TODO: change save strategies for table: Use deduction to save more SPs per run, Subtables/smaller search areas (George
#referenced as possible option)
#TODO: save a smaller set of graphs, try to link them together for the search, use a non-key strategy for the table
#TODO: set weights
#TODO: sinlglesource dijkstra just save previous node, not node lists.
#TODO: cite networkx djikstra algorithm


if __name__ == "__main__":
    main()