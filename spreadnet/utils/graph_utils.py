import collections
import functools
import itertools
from typing import Callable, Optional, Mapping, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import signal
from scipy import spatial
import tensorflow as tf
import tensorflow_gnn as tfgnn


# from build_utils_graph import *
from spreadnet.utils.gnn_utils import *


def generate_task_graph(
    random_state,
    num_nodes_min_max: Tuple[int, int],
    dimensions: int = 2,
    theta: float = 1000.0,
    min_length: int = 1,
    rate: float = 1.0,
) -> tfgnn.GraphTensor:
    """Creates a connected graph.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Args:
      random_state: A random seed for the graph generator. Default= None.
      num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
      dimensions: (optional) An `int` number of dimensions for the positions.
        Default= 2.
      theta: (optional) A `float` threshold parameters for the geographic
        threshold graph's threshold. Large values (1000+) make mostly trees. Try
        20-60 for good non-trees. Default=1000.0.
      min_length: (optional) An `int` minimum number of edges in the shortest
        path. Default= 1.
      rate: (optional) A rate parameter for the node weight exponential sampling
        distribution. Default= 1.0.

    Returns:
      The graph.
    """
    graph = _generate_base_graph(
        random_state,
        num_nodes_min_max=num_nodes_min_max,
        dimensions=dimensions,
        theta=theta,
        rate=rate,
    )
    graph = _add_shortest_path(random_state, graph, min_length=min_length)

    return _convert_to_graph_tensor(graph)


def task_graph_generator(random_seed: int, **task_kwargs):
    random_state = np.random.RandomState(random_seed)
    while True:
        yield generate_task_graph(random_state, **task_kwargs)


def get_dataset(random_seed: int, **task_kwargs):
    def generator_fn():
        return task_graph_generator(random_seed, **task_kwargs)

    graph_spec = graph_tensor_spec_from_sample_graph(next(generator_fn()))
    return tf.data.Dataset.from_generator(generator_fn, output_signature=graph_spec)


def _generate_base_graph(rand, num_nodes_min_max, dimensions, theta, rate):
    """Generates the base graph for the task."""
    # Sample num_nodes.
    num_nodes = rand.randint(*num_nodes_min_max)

    # Create geographic threshold graph.
    pos_array = rand.uniform(size=(num_nodes, dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(
        num_nodes, theta, pos=pos, weight=weight
    )

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight="weight")
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight="weight")
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault("weight", distances[i, j])
    return combined_graph


def _pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def _set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def _add_shortest_path(rand, graph, min_length=1):
    """Samples a shortest path in the graph."""
    # Map from node pairs to the length of their shortest path.
    pair_to_length_dict = {}
    lengths = list(nx.all_pairs_shortest_path_length(graph))
    for x, yy in lengths:
        for y, l in yy.items():
            if l >= min_length:
                pair_to_length_dict[x, y] = l
    if max(pair_to_length_dict.values()) < min_length:
        raise ValueError("All shortest paths are below the minimum length")
    # The node pairs which exceed the minimum length.
    node_pairs = list(pair_to_length_dict)

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [
        prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
    ]

    # Choose the start and end points.
    i = rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(graph, source=start, target=end, weight="length")

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


def _convert_to_graph_tensor(graph_nx):
    """Converts the graph to a GraphTensor."""
    number_of_nodes = graph_nx.number_of_nodes()
    nodes_data = [data for _, data in graph_nx.nodes(data=True)]
    node_features = tf.nest.map_structure(lambda *n: np.stack(n, axis=0), *nodes_data)

    number_of_edges = graph_nx.number_of_edges()
    source_indices, target_indices, edges_data = zip(*graph_nx.edges(data=True))
    source_indices = np.array(source_indices, dtype=np.int32)
    target_indices = np.array(target_indices, dtype=np.int32)
    edge_features = tf.nest.map_structure(lambda *e: np.stack(e, axis=0), *edges_data)
    context_features = dict(graph_nx.graph)
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "cities": tfgnn.NodeSet.from_fields(
                sizes=[number_of_nodes],
                features=node_features,
            )
        },
        edge_sets={
            "roads": tfgnn.EdgeSet.from_fields(
                sizes=[number_of_edges],
                features=edge_features,
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("cities", source_indices), target=("cities", target_indices)
                ),
            )
        },
        context=tfgnn.Context.from_fields(features=context_features),
    )


def draw_graph(ax, task_graph):

    tfgnn.check_scalar_graph_tensor(task_graph)
    assert task_graph.num_components == 1

    ax.set_xticks([])
    ax.set_yticks([])
    node_set = task_graph.node_sets["cities"]
    node_positions = node_set["pos"].numpy()

    start_node_mask = node_set["is_start"].numpy()
    end_node_mask = node_set["is_end"].numpy()
    other_nodes_mask = ~(start_node_mask + end_node_mask)

    in_path_node_mask = node_set["is_in_path"].numpy()

    for label, mask, kwargs in [
        ("Cities", other_nodes_mask, dict(color="lightgrey")),
        ("Start city", start_node_mask, dict(color="red")),
        ("End city", end_node_mask, dict(color="blue")),
        (
            "Cities in shortest path",
            in_path_node_mask,
            dict(color="None", markeredgecolor="black"),
        ),
    ]:
        ax.plot(
            node_positions[mask, 0],
            node_positions[mask, 1],
            "o",
            zorder=100,
            ms=10,
            markeredgewidth=2,
            label=label,
            **kwargs
        )

    edge_positions = np.stack(
        [
            node_positions[task_graph.edge_sets["roads"].adjacency.source.numpy()],
            node_positions[task_graph.edge_sets["roads"].adjacency.target.numpy()],
        ],
        axis=0,
    )

    in_path_edges_mask = task_graph.edge_sets["roads"]["is_in_path"].numpy()
    other_edges_mask = ~in_path_edges_mask
    for label, mask, kwargs in [
        ("Roads", other_edges_mask, dict(color="lightgrey", linewidth=2)),
        (
            "Roads in shortest path",
            in_path_edges_mask,
            dict(color="black", linewidth=5),
        ),
    ]:
        ax.plot(edge_positions[:, mask, 0], edge_positions[:, mask, 1], **kwargs)
        ax.plot(np.nan, np.nan, label=label, **kwargs)  # Single legend element.
