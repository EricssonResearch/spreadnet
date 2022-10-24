import collections
import functools
from typing import Callable, Optional
import tensorflow as tf
import tensorflow_gnn as tfgnn
from spreadnet.tf_gnn.tf_utils.gnn_utils import sum_graphs


def GraphNetworkGraphUpdate(
    *,
    edges_next_state_factory: Callable[..., tf.keras.layers.Layer],
    nodes_next_state_factory: Callable[..., tf.keras.layers.Layer],
    context_next_state_factory: Optional[Callable[..., tf.keras.layers.Layer]],
    receiver_tag: Optional[tfgnn.IncidentNodeTag] = tfgnn.TARGET,
    reduce_type_to_nodes: str = "sum",
    reduce_type_to_context: str = "sum",
    use_input_context_state: bool = True,
    name: str = "graph_network",
):
    """Returns a GraphUpdate to run a GraphNetwork on all node sets and edge
    sets.

    The returned layer implements a Graph Network, as described by
    Battaglia et al.: ["Relational inductive biases, deep learning, and
    graph networks"](https://arxiv.org/abs/1806.01261), 2018, generalized
    to heterogeneous graphs.

    It expects an input GraphTensor with a `tfgnn.HIDDEN_STATE` feature on all
    node sets and edge sets, and also context if `use_input_context_state=True`.
    It runs edge, node, and context updates, in this order, separately for each
    edge set, node set (regardless whether it has an incoming edge set), and also
    context if `context_next_state_factory` is set. Finally, it returns a
    GraphTensor with updated hidden states, incl. a context state, if
    `context_next_state_factory` is set.

    The model can also behave as an Interaction Network ([Battaglia et al., NIPS
    2016](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html))
    by setting
      * `use_input_context_state = False`
      * `context_next_state_factory = None`

    Args:
      edges_next_state_factory: Called with keyword argument `edge_set_name=`
        for each edge set to return the NextState layer for use in the respective
        `tfgnn.keras.layers.EdgeSetUpdate`.
      nodes_next_state_factory: Called with keyword argument `node_set_name=`
        for each node set to return the NextState layer for use in the respective
        `tfgnn.keras.layers.NodeSetUpdate`.
      context_next_state_factory: If set, a `tfgnn.keras.layers.ContextUpdate`
        is included with the NextState layer returned by calling this factory.
      receiver_tag: The incident node tag at which each edge set is used to
        update node sets. Defaults to `tfgnn.TARGET`.
      reduce_type_to_nodes: Controls how incident edges at a node are aggregated
        within each EdgeSet. Defaults to `"sum"`. (The aggregates of the various
        incident EdgeSets are then concatenated.)
      reduce_type_to_context: Controls how the nodes of a NodeSet or the edges of
        an EdgeSet are aggregated for the context update. Defaults to `"sum"`.
        (The aggregates of the various NodeSets/EdgeSets are then concatenated.)
      use_input_context_state: If true, the input `GraphTensor.context` must have
        a `tfgnn.HIDDEN_STATE` feature that gets used as input in all edge, node
        and context updates.
      name: A name for the returned layer.
    """

    def deferred_init_callback(graph_spec):
        context_input_feature = tfgnn.HIDDEN_STATE if use_input_context_state else None

        # To keep track node types that receive each edge type.
        incoming_edge_sets = collections.defaultdict(list)

        # For every edge set, create an EdgeSetUpdate.
        edge_set_updates = {}
        for edge_set_name in sorted(graph_spec.edge_sets_spec.keys()):
            next_state = edges_next_state_factory(edge_set_name=edge_set_name)
            edge_set_updates[edge_set_name] = tfgnn.keras.layers.EdgeSetUpdate(
                next_state=next_state,
                edge_input_feature=tfgnn.HIDDEN_STATE,
                node_input_feature=tfgnn.HIDDEN_STATE,
                context_input_feature=context_input_feature,
            )
            # Keep track of which node set is the receiver for this edge type
            # as we will need it later.
            target_name = graph_spec.edge_sets_spec[
                edge_set_name
            ].adjacency_spec.node_set_name(receiver_tag)
            incoming_edge_sets[target_name].append(edge_set_name)

        # For every node set, create a NodeSetUpdate.
        node_set_updates = {}
        for node_set_name in sorted(graph_spec.node_sets_spec.keys()):
            # Apply a node update, after summing *all* of the received edges
            # for that node set.
            next_state = nodes_next_state_factory(node_set_name=node_set_name)
            node_set_updates[node_set_name] = tfgnn.keras.layers.NodeSetUpdate(
                next_state=next_state,
                edge_set_inputs={
                    edge_set_name: tfgnn.keras.layers.Pool(
                        receiver_tag,
                        reduce_type_to_nodes,
                        feature_name=tfgnn.HIDDEN_STATE,
                    )
                    for edge_set_name in incoming_edge_sets[node_set_name]
                },
                node_input_feature=tfgnn.HIDDEN_STATE,
                context_input_feature=context_input_feature,
            )

        # Create a ContextUpdate, if requested.
        context_update = None
        if context_next_state_factory is not None:
            next_state = context_next_state_factory()
            context_update = tfgnn.keras.layers.ContextUpdate(
                next_state=next_state,
                edge_set_inputs={
                    edge_set_name: tfgnn.keras.layers.Pool(
                        tfgnn.CONTEXT,
                        reduce_type_to_context,
                        feature_name=tfgnn.HIDDEN_STATE,
                    )
                    for edge_set_name in sorted(graph_spec.edge_sets_spec.keys())
                },
                node_set_inputs={
                    node_set_name: tfgnn.keras.layers.Pool(
                        tfgnn.CONTEXT,
                        reduce_type_to_context,
                        feature_name=tfgnn.HIDDEN_STATE,
                    )
                    for node_set_name in sorted(graph_spec.node_sets_spec.keys())
                },
                context_input_feature=context_input_feature,
            )
        return dict(
            edge_sets=edge_set_updates,
            node_sets=node_set_updates,
            context=context_update,
        )

    return tfgnn.keras.layers.GraphUpdate(
        deferred_init_callback=deferred_init_callback, name=name
    )


def build_mlp(
    num_hidden_layers: int,
    hidden_size: int,
    output_size: int,
    use_layer_norm: bool,
    activation: str = "relu",
    activate_final: bool = False,
    name: str = "mlp",
):
    """Builds an MLP."""
    output_sizes = [hidden_size] * num_hidden_layers + [output_size]
    mlp = tf.keras.Sequential(name="mlp")
    for layer_i, size in enumerate(output_sizes):
        layer_activation = activation
        if not activate_final and layer_i == len(output_sizes) - 1:
            layer_activation = None
        mlp.add(
            tf.keras.layers.Dense(
                size,
                activation=layer_activation,
                use_bias=True,
                kernel_initializer="variance_scaling",
                bias_initializer="zeros",
                name=f"{name}/dense_{layer_i}",
            )
        )
    if use_layer_norm:
        mlp.add(tf.keras.layers.LayerNormalization(name=f"{name}/layer_norm"))
    return mlp


class EncodeProcessDecode(tf.keras.layers.Layer):
    """Implements EncodeProcessDecode from https://arxiv.org/abs/2002.09405.

    It is extended to:
    * run with global updates.
    * use shared or unshared weights when using message passing.
    * produce per node, per edge and/or context outputs.

    It requires node sets, edge sets and context with the
    `tfgnn.HIDDEN_STATE` feature.
    """

    def __init__(
        self,
        edge_output_size: Optional[int],
        node_output_size: Optional[int],
        context_output_size: Optional[int],
        num_message_passing_steps: int,
        num_mlp_hidden_layers: int,
        mlp_hidden_size: int,
        latent_size: int,
        use_layer_norm: bool,
        shared_processors: bool,
        reduce_type_to_nodes: str = "sum",
        reduce_type_to_context: str = "sum",
        name: str = "encode_process_decode",
    ):
        super().__init__(name=name)

        # Build graph encoder.
        def encoder_fn(graph_piece, *, edge_set_name=None, node_set_name=None):
            piece_name = (
                f"edges_{edge_set_name}"
                if edge_set_name
                else f"nodes_{node_set_name}"
                if node_set_name
                else "context"
            )
            mlp = build_mlp(
                num_hidden_layers=num_mlp_hidden_layers,
                hidden_size=mlp_hidden_size,
                output_size=latent_size,
                use_layer_norm=use_layer_norm,
                name=f"{self.name}/encoder/{piece_name}",
            )
            return mlp(graph_piece[tfgnn.HIDDEN_STATE])

        self._encoder = tfgnn.keras.layers.MapFeatures(
            edge_sets_fn=encoder_fn, node_sets_fn=encoder_fn, context_fn=encoder_fn
        )

        # Build graph processor(s).
        # We will just concatenate all inputs to each edge update, node update
        # and context upcate, and run an MLP on it.

        def processor_fn(*, processor_name, edge_set_name=None, node_set_name=None):
            if edge_set_name is not None:
                mlp_name = f"{processor_name}/edges_{edge_set_name}"
            elif node_set_name is not None:
                mlp_name = f"{processor_name}/nodes_{node_set_name}"
            else:
                mlp_name = f"{processor_name}/context"
            mlp = build_mlp(
                name=mlp_name,
                num_hidden_layers=num_mlp_hidden_layers,
                hidden_size=mlp_hidden_size,
                output_size=latent_size,
                use_layer_norm=use_layer_norm,
            )
            return tfgnn.keras.layers.NextStateFromConcat(mlp)

        num_processors = 1 if shared_processors else num_message_passing_steps

        processors = []
        for processor_i in range(num_processors):
            processor_name = f"{self.name}/processor_{processor_i}"
            processor_fn_named = functools.partial(
                processor_fn, processor_name=processor_name
            )
            processors.append(
                GraphNetworkGraphUpdate(
                    edges_next_state_factory=processor_fn_named,
                    nodes_next_state_factory=processor_fn_named,
                    context_next_state_factory=processor_fn_named,
                    reduce_type_to_nodes=reduce_type_to_nodes,
                    reduce_type_to_context=reduce_type_to_context,
                    name=processor_name,
                )
            )

        if shared_processors:
            self._processors = processors * num_message_passing_steps
        else:
            self._processors = processors

        # Build graph decoder.
        def decoder_fn(graph_piece, *, edge_set_name=None, node_set_name=None):
            piece_name = (
                f"edges_{edge_set_name}"
                if edge_set_name
                else f"nodes_{node_set_name}"
                if node_set_name
                else "context"
            )
            if edge_set_name:
                output_size = edge_output_size
            elif node_set_name:
                output_size = node_output_size
            else:
                output_size = context_output_size
            mlp = build_mlp(
                num_hidden_layers=num_mlp_hidden_layers,
                hidden_size=mlp_hidden_size,
                output_size=output_size,
                use_layer_norm=False,  # Never LayerNorm for the outputs.
                name=f"{self.name}/decoder/{piece_name}",
            )
            return mlp(graph_piece[tfgnn.HIDDEN_STATE])

        self._decoder = tfgnn.keras.layers.MapFeatures(
            edge_sets_fn=decoder_fn if edge_output_size else None,
            node_sets_fn=decoder_fn if node_output_size else None,
            context_fn=decoder_fn if context_output_size else None,
        )

    def call(self, input_graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
        latent_graph = self._encoder(input_graph)
        for processor in self._processors:
            residual_graph = processor(latent_graph)
            latent_graph = sum_graphs(residual_graph, latent_graph)
        output_graph = self._decoder(latent_graph)
        return output_graph
