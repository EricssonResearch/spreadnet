from dijkstra_algorithm import single_source_dijkstra
import networkx as nx

# Holds all SP from all source node runs.
memoTable = {}


# Took out the check before adding as the table itself only holds unique
# values so it wasn't needed
def add_item_to_memo_table(start_node, item):
    key_list = item.keys()
    for item_key in key_list:
        memoTable[(start_node, item_key)] = item[item_key]


# memoization table search function
def search_memo_table(start_node, end_node):
    if memoTable.get((start_node, end_node)):
        print("shortest path:")
        print(memoTable[(start_node, end_node)])


# runs the shortest path dijkstra algorithm, brings back lengths
# for all (unused at this point)
# and all paths which is printed and then added to the memoization table.
def shortest_path(G, start_node, end_node, weight="weight"):
    if search_memo_table(start_node, end_node):
        print(memoTable[(start_node, end_node)])
    else:
        all_lengths_from_source, all_paths_from_source = single_source_dijkstra(
            G, start_node, None, None, weight
        )
        print(all_paths_from_source[end_node])
        add_item_to_memo_table(start_node, all_paths_from_source)


# TODO: need to add in a comparison to the unchanged SP
#  run as well as some analysis.
def main():
    # build graph
    G = nx.path_graph(10)
    # tests here
    shortest_path(G, 1, 4)
    shortest_path(G, 1, 8)
    shortest_path(G, 2, 9)
    shortest_path(G, 3, 5)
    # check memoization table
    # print(memoTable)


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


if __name__ == "__main__":
    main()
