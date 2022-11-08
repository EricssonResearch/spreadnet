# Copyright 2021 The TensorFlow GNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modifications copyright (C) 2022 Haodong Zhao
# ==============================================================================

"""Graph generator to generate graphs randomly.The graphs are geographic
threshold graphs, but with added edges via a minimum spanning tree algorithm,
to ensure all nodes are connected.

This Graph generator is modified from the graph generation part of tensorflow/gnn
    https://github.com/tensorflow/gnn/blob/main/examples/notebooks/graph_network_shortest_path.ipynb

TODO: Code duplicated in the tf_gnn utils also. One copy has to be removed.
      Do all models require this type of data generation?
TODO: For the final spreadnet(aka non epxerimental).
      Get rid of all code that is not written by us so that
      we do not have to bother with their license.
"""

import itertools
from typing import Tuple
import networkx as nx
import numpy as np
from scipy import spatial  # Spatial algorithms and data structures


def _set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def _pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class GraphGenerator:
    """A graph generator that creates a connected graph."""

    def __init__(
        self,
        random_seed: int,
        num_nodes_min_max: Tuple[int, int],
        dimensions: int = 2,
        theta: float = 1000.0,
        min_length: int = 1,
        rate: float = 1.0,
    ):
        """

        Args:
            random_seed:  A random seed for the graph generator. Default= None.
            num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
            dimensions: (optional) An `int` number of dimensions for the positions.
                        Default= 2.
            theta: graph's threshold. Large values (1000+) make mostly trees.Try 20-60
                    for good non-trees. Default=1000.0.
            min_length: (optional) An `int` minimum number of edges in the shortest
                        path. Default= 1. sampling distribution. Default= 1.0.
        """

        self.random_state = np.random.RandomState(random_seed)
        self.num_nodes_min_max = num_nodes_min_max
        self.dimensions = dimensions
        self.theta = theta
        self.min_length = min_length
        self.rate = rate

    def set_theta(self, new_theta):
        self.theta = new_theta

    def task_graph_generator(self):
        """The graphs are geographic threshold graphs, but with added edges via
        a minimum spanning tree algorithm, to ensure all nodes are connected.
        The generated graph is a directed graph, and it contains path
        information.

        Returns:
            Generator of networkx.DiGraph
        """
        while True:
            yield self._generate_task_graph()

    def base_graph_generator(self):
        """The graphs are geographic threshold graphs, but with added edges via
        a minimum spanning tree algorithm, to ensure all nodes are connected.

        returns:
            Generator of networkx.Graph
        """
        while True:
            yield self._generate_base_graph()

    def _generate_task_graph(self):
        graph = self._generate_base_graph()
        graph = self.add_shortest_path(graph, self.min_length)
        return graph

    def _geo_diff(self, lat1, lon1, lat2, lon2):
        return abs(lat2 - lat1) + abs(lon2 - lon1)

    def _generate_base_graph(self):
        """Generate the base graph for the task.

        Returns:
            A basic graph.
        """
        # 1. Sample num_nodes.
        num_nodes = self.random_state.randint(*self.num_nodes_min_max)

        # 2. Create geographic threshold graph.
        pos_array = self.random_state.uniform(
            size=(num_nodes, self.dimensions)
        )  # Draw samples from a uniform distribution. num_node ✕ dimensions

        pos = dict(enumerate(pos_array))  # len(pos) = num_nodes, {0: [,], 1:[,], ...}
        weight = dict(
            enumerate(self.random_state.exponential(self.rate, size=num_nodes))
        )  # Draw samples from an exponential distribution.
        # weight: {0:num0, 1:num1, 2:num2, ...}, len(weight) = num_nodes
        geo_graph: nx.Graph = nx.geographical_threshold_graph(
            num_nodes, self.theta, pos=pos, weight=weight
        )

        # 3. Connect sub graphs
        components = sorted(nx.connected_components(geo_graph), key=len)
        max_connection = 3
        for idx, c in enumerate(components):
            connect_from_choices = list(c)
            connections = 1

            while connections <= max_connection and idx + connections < len(components):
                connect_from = self.random_state.choice(connect_from_choices)
                connect_from_pos = pos[connect_from]

                connect_to_nodes = list(components[idx + connections])
                closest_node = connect_to_nodes[0]
                closest_node_pos = pos[closest_node]
                closest_distance = self._geo_diff(
                    connect_from_pos[0],
                    connect_from_pos[1],
                    closest_node_pos[0],
                    closest_node_pos[1],
                )

                for node in connect_to_nodes:
                    if connect_from == node:
                        continue

                    npos = pos[node]
                    distance = self._geo_diff(
                        connect_from_pos[0], connect_from_pos[1], npos[0], npos[1]
                    )

                    if distance < closest_distance:
                        closest_node = node
                        closest_node_pos = npos
                        closest_distance = distance

                geo_graph.add_edge(connect_from, closest_node)
                connections += 1

        # pdist: Pairwise distances between observations in n-dimensional space.
        # squareform: Convert a vector-form distance vector to a square-form distance
        # matrix, and vice-versa.
        distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))

        # 4. Put all distance weights into edge attributes.
        for i, j in geo_graph.edges():
            geo_graph.get_edge_data(i, j).setdefault("weight", distances[i, j])

        return geo_graph

    def add_shortest_path(self, graph: nx.DiGraph, min_length=1):
        """
            Sample the shortest path in the graph
        Args:
            graph: A basic graph.
            min_length: the minimum length of the shortest path.

        Returns:
            A directed graph with the shortest path data.

        """
        # Choose the start and end points.
        num_nodes = graph.number_of_nodes()
        nodes_set = set(graph.nodes())

        start = self.random_state.choice(num_nodes)
        nodes_set = nodes_set - set([start])

        path = []
        while len(path) < min_length:
            if len(nodes_set) == 0:
                return self._generate_task_graph()

            end = self.random_state.choice(list(nodes_set))
            nodes_set = nodes_set - set([end])
            path = nx.shortest_path(graph, source=start, target=end, weight="weight")

        # Creates a directed graph, to store the directed path from start to end.
        digraph = graph.to_directed()

        # Add the "start", "end", and "solution" attributes to the nodes and edges.
        digraph.add_node(start, is_start=True)
        digraph.add_node(end, is_end=True)
        digraph.add_nodes_from(_set_diff(digraph.nodes(), [start]), is_start=False)
        digraph.add_nodes_from(_set_diff(digraph.nodes(), [end]), is_end=False)
        digraph.add_nodes_from(_set_diff(digraph.nodes(), path), is_in_path=False)
        digraph.add_nodes_from(path, is_in_path=True)
        path_edges = list(_pairwise(path))
        digraph.add_edges_from(_set_diff(digraph.edges(), path_edges), is_in_path=False)
        digraph.add_edges_from(path_edges, is_in_path=True)

        return digraph
