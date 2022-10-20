# from dijkstra_algorithm import single_source_dijkstra

from spreadnet.dijkstra_memoization import dijkstra_algorithm

import networkx as nx

from networkx import weisfeiler_lehman_graph_hash

"""
# Holds all SP from all source node runs.
"""
memoTable = {}


def add_item_to_memo_table(graph_hash, start_node, allPathsFromSource):
    """Fills in the memoization table (dictionary type) with the shortest paths
    The dictionary does not include duplicates so no need to do duplicate
    check.

    Args:
        start_node: node label
            Start node for SP
        item: dictionary items
            found shortest paths

    Returns:
        nothing
    """
    print("allPathsfromSource", allPathsFromSource)
    pathDic = {}
    memoTable.setdefault(graph_hash, {})[start_node] = pathDic

    key_list = allPathsFromSource.keys()
    for item_key in key_list:
        pathDic[item_key] = allPathsFromSource[item_key]
    memoTable[graph_hash][start_node] = pathDic

    # if memoTable[graph_hash] is None:
    #    memoTable[graph_hash] = {}
    # key_list = allPathsFromSource.keys()
    # for item_key in key_list:
    #    memoTable[graph_hash][(start_node, item_key)] = allPathsFromSource[item_key]


def search_memo_table(graph_hash, start_node, end_node):
    """Search function for memoization table.

    Args:
        start_node: node label
            Start node for SP
        end_node: node label
            Ending node

    Returns:
        return shortest path or -1 implying none known
    """
    # TODO: add a check for if search has been done and not in memo table
    #  (no path) so no second search
    # if memoTable.get(graph_hash, {}).get(start_node, end_node):
    # memoTable.get((graph_hash, start_node, end_node)):
    print(
        "WHAT IS THIS??",
        memoTable.get(graph_hash, {}).get(start_node, {}).get(end_node),
    )
    return memoTable.get(graph_hash, {}).get(start_node, {}).get(end_node)
    # else:
    #    return -1


def shortest_path(G, start_node, end_node, weight="weight"):
    """Runs the shortest path dijkstra algorithm, brings back lengths for all
    (unused at this point) and all paths which is printed and then added to the
    memoization table.

    Args:
        G: NetworkX graph
        start_node: node label
            Source node for path
        end_node: node label
            End node for SP
        weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns:
      The shortest path from the memoization table or from running Dijkstra algorithm.
    """
    hashed_graph = weisfeiler_lehman_graph_hash(G)
    # TODO make sure to implement that it is with unique hash
    #  (add edge features to graph function call)

    path = search_memo_table(hashed_graph, start_node, end_node)
    if path is not None:
        return path
    else:
        (
            all_lengths_from_source,
            all_paths_from_source,
        ) = dijkstra_algorithm.single_source_dijkstra(G, start_node, None, None, weight)
        add_item_to_memo_table(hashed_graph, start_node, all_paths_from_source)
        print("memotable from shortest path algorithm", memoTable)
        print("shortest path", all_paths_from_source[end_node])
        return all_paths_from_source[end_node]


def main():
    # build graph
    # G_ten = nx.DiGraph()
    G_ten = nx.path_graph(10)
    G_ten1 = nx.path_graph(20)

    # H = nx.path_graph(10)
    # G_ten.add_nodes_from(H)
    # G_ten.add_edges_from(H.edges)
    # = nx.complete_graph(100)
    # G_hundred = nx.path_graph(100)
    # G_thousand = nx.path_graph(1000)
    print("G_ten:", G_ten)
    print(memoTable)
    hashed_graph = weisfeiler_lehman_graph_hash(G_ten)
    print("hashed graph", hashed_graph)
    (
        all_lengths_from_source,
        all_paths_from_source,
    ) = dijkstra_algorithm.single_source_dijkstra(G_ten, 7, None, None, "weight")
    print("all lengths from source", all_lengths_from_source)
    print("all_paths form source", all_paths_from_source)
    add_item_to_memo_table(hashed_graph, 7, all_paths_from_source)
    print(memoTable)
    search_memo_table(hashed_graph, 7, 3)
    print("all lengths from source", all_lengths_from_source)
    print("all_paths form source", all_paths_from_source)

    shortest_path(G_ten, 1, 4, "weight")
    shortest_path(G_ten1, 1, 4, "weight")

    # tests here
    # x = shortest_path(G_ten, 1, 4)
    # print("Shortest path", x)

    # x = shortest_path(G_ten, 1, 8)
    # print("Shortest path", x)

    # x = shortest_path(G_ten, 2, 9)
    # print("Shortest path", x)

    # x = shortest_path(G_ten, 3, 5)

    # print("Shortest path", x)
    # check memoization table
    print("memotable final", memoTable)


if __name__ == "__main__":
    main()


# TODO: need to add in a comparison to the unchanged SP
#  run as well as some analysis.

# TODO: experiments Is it faster to find all_pairs first to fill
#  memoization table and then just search for results?
# TODO: check other algorithms for possible faster run though
#  I think they all start at the same algorithm..
# TODO: change save strategies for table: Use deduction to save
#  more SPs per run, Sub-tables/smaller search areas
#  (George may have references to something as possible option)
# TODO: save a smaller set of graphs, try to link them together for the
#  search, use a non-key strategy for the table
# TODO: set weights
# TODO: single source dijkstra just save previous node,
#  not node lists.
# TODO: test size of graph with hash functions to see if at some point
#  eisfeiler_lehman_subgraph_hashes will be needed
