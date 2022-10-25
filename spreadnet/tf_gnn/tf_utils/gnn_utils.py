from typing import Mapping


import tensorflow as tf
import tensorflow_gnn as tfgnn


def sum_graphs(
    graph_1: tfgnn.GraphTensor,
    graph_2: tfgnn.GraphTensor,
) -> tfgnn.GraphTensor:
    """Sums all features in two identical graphs."""
    assert set(graph_1.edge_sets.keys()) == set(graph_2.edge_sets.keys())
    new_edge_set_features = {}
    for set_name in graph_1.edge_sets.keys():
        new_edge_set_features[set_name] = _sum_feature_dict(
            graph_1.edge_sets[set_name].get_features_dict(),
            graph_2.edge_sets[set_name].get_features_dict(),
        )

    assert set(graph_1.node_sets.keys()) == set(graph_2.node_sets.keys())
    new_node_set_features = {}
    for set_name in graph_1.node_sets.keys():
        new_node_set_features[set_name] = _sum_feature_dict(
            graph_1.node_sets[set_name].get_features_dict(),
            graph_2.node_sets[set_name].get_features_dict(),
        )

    new_context_features = _sum_feature_dict(
        graph_1.context.get_features_dict(), graph_2.context.get_features_dict()
    )
    return graph_1.replace_features(
        edge_sets=new_edge_set_features,
        node_sets=new_node_set_features,
        context=new_context_features,
    )


def _sum_feature_dict(
    features_1: Mapping[str, tf.Tensor], features_2: Mapping[str, tf.Tensor]
) -> Mapping[str, tf.Tensor]:
    tf.nest.assert_same_structure(features_1, features_2)
    return tf.nest.map_structure(lambda x, y: x + y, features_1, features_2)


def nest_to_numpy(nest):
    return tf.nest.map_structure(lambda x: x.numpy(), nest)


# TODO(b/205123804): Provide a library function for this.
def graph_tensor_spec_from_sample_graph(sample_graph):
    """Build variable node/edge spec given a sample graph without batch
    axes."""
    tfgnn.check_scalar_graph_tensor(sample_graph)
    sample_graph_spec = sample_graph.spec
    node_sets_spec = {}
    for node_set_name, node_set_spec in sample_graph_spec.node_sets_spec.items():
        new_features_spec = {}
        for feature_name, feature_spec in node_set_spec.features_spec.items():
            new_features_spec[feature_name] = _to_none_leading_dim(feature_spec)
        node_sets_spec[node_set_name] = tfgnn.NodeSetSpec.from_field_specs(
            features_spec=new_features_spec,
            sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
        )

    edge_sets_spec = {}
    for edge_set_name, edge_set_spec in sample_graph_spec.edge_sets_spec.items():
        new_features_spec = {}
        for feature_name, feature_spec in edge_set_spec.features_spec.items():
            new_features_spec[feature_name] = _to_none_leading_dim(feature_spec)

        adjacency_spec = tfgnn.AdjacencySpec.from_incident_node_sets(
            source_node_set=edge_set_spec.adjacency_spec.source_name,
            target_node_set=edge_set_spec.adjacency_spec.target_name,
            index_spec=_to_none_leading_dim(edge_set_spec.adjacency_spec.target),
        )

        edge_sets_spec[edge_set_name] = tfgnn.EdgeSetSpec.from_field_specs(
            features_spec=new_features_spec,
            sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
            adjacency_spec=adjacency_spec,
        )

    context_spec = sample_graph_spec.context_spec

    return tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec=node_sets_spec,
        edge_sets_spec=edge_sets_spec,
        context_spec=context_spec,
    )


def _to_none_leading_dim(spec):
    new_shape = list(spec.shape)
    new_shape[0] = None
    return tf.TensorSpec(shape=new_shape, dtype=spec.dtype)
