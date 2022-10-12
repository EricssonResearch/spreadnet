import unittest
import networkx as nx

# import spreadnet.dijkstra_memoization.dijkstra_runner as sn

# from spreadnet.dijkstra_memoization import dijkstra_runner, dijkstra_algorithm


# import dijkstra_runner
# from heapq import heappop, heappush
# from itertools import count


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.graph_ten_node = nx.path_graph(10)
        self.memoTable = {}
        print(self.graph_ten_node)

    def test_shortest_path_found_nx(self):

        self.assertEqual(self.graph_ten_node, self.graph_ten_node)  # add assertion here
        self.assertEqual(True, True)

    #  self.assertEqual(
    #      nx.single_source_dijkstra_path(self.graph_ten_node, 1, 5),
    #      sn.shortest_path(self.graph_ten_node, 1, 5),
    #  )


if __name__ == "__main__":
    unittest.main()
