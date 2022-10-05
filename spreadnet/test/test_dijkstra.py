import unittest
import networkx as nx

# import dijkstra_runner
# from heapq import heappop, heappush
# from itertools import count


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.graph_ten_node = nx.path_graph(10)

    def test_shortest_path_found_nx(self):

        self.assertEqual(self.graph_ten_node, self.graph_ten_node)  # add assertion here
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()
