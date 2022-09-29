"""
Tensorflow GNN specifc utilities used during testing.  
    TODO: 
        1. nx to tensor used by tf_gnn function.
            1.1 All helper functions
        2. For license purposes mention that some of the code related to tf_gnn is taken from 
            the Relational Inductive iases... paper
Libraries:
    networkx, tensorflow, tensorflow_gnn


Details:
    If the tensorflow data generation is included it should be only 
    for testing purposes and removed at the end. 
    The final code should use the same DataLoader that pyg_gnn uses.
     
"""
import networkx as nx
from scipy import spatial
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import itertools
import collections


class TfGNNUtils:
    def __init__(self) -> None:
        pass

    def _convert_to_graph_tensor(self, graph_nx):
        """Converts the graph to a GraphTensor."""
        number_of_nodes = graph_nx.number_of_nodes()
        nodes_data = [data for _, data in graph_nx.nodes(data=True)]
        node_features = tf.nest.map_structure(
            lambda *n: np.stack(n, axis=0), *nodes_data
        )

        number_of_edges = graph_nx.number_of_edges()
        source_indices, target_indices, edges_data = zip(*graph_nx.edges(data=True))
        source_indices = np.array(source_indices, dtype=np.int32)
        target_indices = np.array(target_indices, dtype=np.int32)
        edge_features = tf.nest.map_structure(
            lambda *e: np.stack(e, axis=0), *edges_data
        )
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
                        source=("cities", source_indices),
                        target=("cities", target_indices),
                    ),
                )
            },
            context=tfgnn.Context.from_fields(features=context_features),
        )

    def tf_pred_tensor_graph_to_nx_graph(self, graph_tensor):
        """Predicted tensor graph to nx graph.

        Args:
            graph_tensor (_type_): _description_

        Returns:
            _type_: _description_
        """

        tfgnn.check_scalar_graph_tensor(graph_tensor)
        assert graph_tensor.num_components == 1

        # Empty graph to be populated
        G = nx.Graph()

        node_set = graph_tensor.node_sets["cities"]
        node_positions = node_set["pos"].numpy()

        start_node_mask = node_set["is_start"].numpy()
        end_node_mask = node_set["is_end"].numpy()
        other_nodes_mask = ~(start_node_mask + end_node_mask)
        node_weights = node_set["weight"].numpy()
        in_path_node_mask = node_set["is_in_path"].numpy()

        # Add nodes with specifc atributes
        for i in range(len(node_positions)):
            path_flag = False
            start_flag = False
            end_flag = False
            if start_node_mask[i] == True:
                start_flag = True
            if in_path_node_mask[i] == True:
                path_flag = True
            if end_node_mask[i] == True:
                end_flag = True

            G.add_node(
                i,
                pos=node_positions[i],
                is_start=start_flag,
                is_end=end_flag,
                is_in_path=path_flag,
                weight=node_weights[i],
            )

        # Add edges from tensor

        in_path_edges_mask = graph_tensor.edge_sets["roads"]["is_in_path"].numpy()
        edge_weights = graph_tensor.edge_sets["roads"]["weight"].numpy()
        edge_links = np.stack(
            [
                graph_tensor.edge_sets["roads"].adjacency.source.numpy(),
                graph_tensor.edge_sets["roads"].adjacency.target.numpy(),
            ],
            axis=0,
        )

        for i in range(len(edge_links[0])):
            path_flag = False

            if in_path_edges_mask[i] == True:
                path_flag = True

            G.add_edge(
                edge_links[0][i],
                edge_links[1][i],
                weight=edge_weights[i],
                is_in_path=path_flag,
            )
        return G

    def prob_labels(self, ot_graph) -> nx.Graph:
        """Gets a output tensorflow graph and computes probabilities as labels for nx graph.

        Used by the nx_draw_custom function.

        Args:
            ot_graph (_type_): Instead of features the network has a
                hidden state that only contains the outpus.

        Returns:
            node_labels (dict): {node_num: prob_is_in_path 0..1}
            edge_labels (dict): {(s, t): prob_is_in_path 0..1}
        """

        node_labels = {}
        edge_labels = {}

        node_logits = ot_graph.node_sets["cities"][tfgnn.HIDDEN_STATE]
        node_prob = tf.nn.softmax(node_logits)  # assume nodes are in order

        edge_logits = ot_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE]
        edge_prob = tf.nn.softmax(edge_logits)

        for n in range(0, len(ot_graph.node_sets["cities"][tfgnn.HIDDEN_STATE])):
            node_labels[n] = np.round(node_prob[n][1].numpy(), 2)  # prob is in path

        edge_links = np.stack(
            [
                ot_graph.edge_sets["roads"].adjacency.source.numpy(),
                ot_graph.edge_sets["roads"].adjacency.target.numpy(),
            ],
            axis=0,
        )

        for i in range(0, len(ot_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE])):
            edge_labels[(edge_links[0][i], edge_links[1][i])] = np.round(
                edge_prob[i][1].numpy(), 2
            )

        return node_labels, edge_labels

    def predict_from_final_hidden_state(self, task_graph, output_graph):
        """Transforms output logits into "is_in_path" prediction."""

        city_features = task_graph.node_sets["cities"].get_features_dict()
        node_logits = output_graph.node_sets["cities"][tfgnn.HIDDEN_STATE]
        city_features["is_in_path"] = tf.cast(tf.argmax(node_logits, axis=-1), tf.bool)

        road_features = task_graph.edge_sets["roads"].get_features_dict()
        edge_logits = output_graph.edge_sets["roads"][tfgnn.HIDDEN_STATE]
        road_features["is_in_path"] = tf.cast(tf.argmax(edge_logits, axis=-1), tf.bool)

        return task_graph.replace_features(
            node_sets={"cities": city_features}, edge_sets={"roads": road_features}
        )

    def _set_initial_node_state(self, cities_set, *, node_set_name):
        assert node_set_name == "cities"
        return tf.concat(
            [
                tf.cast(cities_set["weight"], tf.float32)[..., None],
                tf.cast(cities_set["is_start"], tf.float32)[..., None],
                tf.cast(cities_set["is_end"], tf.float32)[..., None],
                # Don't provide the position for better generalization.
                # tf.cast(city_features["pos"], tf.float32),
                # Do not give the answer, unless debugging!
                # tf.cast(city_features["is_in_path"], tf.float32)[..., None],
            ],
            axis=-1,
        )

    def _set_initial_edge_state(self, road_set, *, edge_set_name):
        assert edge_set_name == "roads"
        return tf.concat(
            [
                tf.cast(road_set["weight"], tf.float32)[..., None],
                # Do not give the answer, unless debugging!
                # tf.cast(road_features["is_in_path"], tf.float32)[..., None],
            ],
            axis=-1,
        )

    def _set_initial_context_state(self, context):
        return tfgnn.keras.layers.MakeEmptyFeature()(context)

    """ 
        The following functions are added to aid the integration of the visualization.
        The visualization was initially build for tf_gnn.  
        The following should be removed if the data loader will not use them for 
        equivalence testing between tf_gnn and pyg_gnn. 

        
    """

    def _generate_base_graph(self, rand, num_nodes_min_max, dimensions, theta, rate):
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

    def _pairwise(self, iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def _set_diff(self, seq0, seq1):
        """Return the set difference between 2 sequences as a list."""
        return list(set(seq0) - set(seq1))

    def _add_shortest_path(self, rand, graph, min_length=1):
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
        digraph.add_nodes_from(self._set_diff(digraph.nodes(), [start]), is_start=False)
        digraph.add_nodes_from(self._set_diff(digraph.nodes(), [end]), is_end=False)
        digraph.add_nodes_from(self._set_diff(digraph.nodes(), path), is_in_path=False)
        digraph.add_nodes_from(path, is_in_path=True)
        path_edges = list(self._pairwise(path))
        digraph.add_edges_from(
            self._set_diff(digraph.edges(), path_edges), is_in_path=False
        )
        digraph.add_edges_from(path_edges, is_in_path=True)

        return digraph
