"""
Tensorflow GNN specific utilities used during testing.
    TODO:
        1. nx to tensor used by tf_gnn function.
            1.1 All helper functions
        2. For license purposes mention that some of the code related to
        tf_gnn is taken from
            the Relational Inductive iases... paper
Libraries:
    networkx, tensorflow, tensorflow_gnn


Details:
    If the tensorflow data generation is included it should be only
    for testing purposes and removed at the end.
    The final code should use the same DataLoader that pyg_gnn uses.

"""
import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn


class TfGNNUtils:
    def __init__(self) -> None:

        self.build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=self._set_initial_node_state,
            edge_sets_fn=self._set_initial_edge_state,
            context_fn=self._set_initial_context_state,
        )

    def convert_to_graph_tensor(self, graph_nx):
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

    def nx_standard_format_from_tensor(
        self, ground_truth_graph=None, output_graph_tensor=None
    ):
        """Combines the ground truth with the logit predictions.

        Args:
            graph_tensor (_type_): Tensor graph forma

        Returns: Nx graph in the documentation defined format.
                TODO: Write the documentation. :)
        """

        tfgnn.check_scalar_graph_tensor(output_graph_tensor)
        assert output_graph_tensor.num_components == 1

        graph_updated = (
            ground_truth_graph  # Start from the grond truth and add the values
        )

        node_logits = output_graph_tensor.node_sets["cities"][
            tfgnn.HIDDEN_STATE
        ].numpy()
        edge_logits = output_graph_tensor.edge_sets["roads"][tfgnn.HIDDEN_STATE].numpy()
        # print(edge_logits)
        node_labels = {}
        for i in range(len(node_logits)):
            node_labels[i] = node_logits[i]
        nx.set_node_attributes(graph_updated, node_labels, "logits")

        edge_links = np.stack(
            [
                output_graph_tensor.edge_sets["roads"].adjacency.source.numpy(),
                output_graph_tensor.edge_sets["roads"].adjacency.target.numpy(),
            ],
            axis=0,
        )

        edge_labels = {}
        for i in range(0, len(edge_links[0])):
            edge_labels[(edge_links[0][i], edge_links[1][i])] = {
                "logits": edge_logits[i]
            }

        nx.set_edge_attributes(graph_updated, edge_labels)
        # loop through the output edges

        # print(graph_updated)

        return graph_updated

        # Append the logits of the outputs as a feature to the
        # original ground truth graph.
        # Assumes that the order of the nodes and the edges remains the same.

    def tf_pred_tensor_graph_to_nx_graph(self, graph_tensor):
        """Predicted tensor graph to nx graph for tf_gnn.

        Args:
            graph_tensor (_type_): Graph in tensor  format as used by the

        Returns:
            nx_graph: Returns an nx Graph in the same format used
            by the convert_to_graph tenosr function.
        """

        tfgnn.check_scalar_graph_tensor(graph_tensor)
        assert graph_tensor.num_components == 1

        # Empty graph to be populated
        G = nx.Graph()

        node_set = graph_tensor.node_sets["cities"]
        node_positions = node_set["pos"].numpy()

        start_node_mask = node_set["is_start"].numpy()
        end_node_mask = node_set["is_end"].numpy()

        node_weights = node_set["weight"].numpy()
        in_path_node_mask = node_set["is_in_path"].numpy()

        # Add nodes with specific attributes
        for i in range(len(node_positions)):
            path_flag = False
            start_flag = False
            end_flag = False
            if start_node_mask[i] is True:
                start_flag = True
            if in_path_node_mask[i] is True:
                path_flag = True
            if end_node_mask[i] is True:
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

            if in_path_edges_mask[i] is True:
                path_flag = True

            G.add_edge(
                edge_links[0][i],
                edge_links[1][i],
                weight=edge_weights[i],
                is_in_path=path_flag,
            )
        return G

    def prob_labels(self, ot_graph) -> nx.Graph:
        """Gets a output tensorflow graph and computes probabilities as labels
        for nx graph.

        Used by the nx_draw_custom function.

        Args:
            ot_graph (_type_): Instead of features the network has a
                hidden state that only contains the outputs.

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
