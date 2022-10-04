from dijkstra_algorithm import single_source_dijkstra


"""
# Holds all SP from all source node runs.
"""
memoTable = {}


def add_item_to_memo_table(start_node, item):
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
    key_list = item.keys()
    for item_key in key_list:
        memoTable[(start_node, item_key)] = item[item_key]


def search_memo_table(start_node, end_node):
    """Search function for memoization table.

    Args:
        start_node: node label
            Start node for SP
        end_node: node label
            Ending node

    Returns:
        return shortest path or -1 implying none known
    """
    if memoTable.get((start_node, end_node)):
        return memoTable[(start_node, end_node)]
    else:
        return -1


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

    if search_memo_table(start_node, end_node) != -1:
        return memoTable[(start_node, end_node)]
    else:
        all_lengths_from_source, all_paths_from_source = single_source_dijkstra(
            G, start_node, None, None, weight
        )
        add_item_to_memo_table(start_node, all_paths_from_source)
        return all_paths_from_source[end_node]


# TODO: need to add in a comparison to the unchanged SP
#  run as well as some analysis.

# TODO: experiments Is it faster to fun all_pairs first to fill
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
# TODO: cite networkx dijkstra algorithm
