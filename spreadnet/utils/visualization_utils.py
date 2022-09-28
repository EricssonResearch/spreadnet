"""
    Utilities for visualizing graphs. 
    The code was initially designed for tf_gnn. 

    TODO: Add utilities for pyg_gnn if needed. 

"""


class VisualUtils:
    def __init__(self) -> None:
        pass

    def nx_draw(self, G, label_w_weights=False, output_graph=None) -> None:
        """Draws an nx graph with weights for the ground truth and probabilities for predicted graph.

        If label_w_weights is set to false then an output_graph is needed.
        Args:
            G (_type_): Ground Truth nx graph.
            label_w_weights (bool, optional): Use weights or probabilities Defaults to False.
            output_graph (_type_, optional): tf_gnn output graph. Defaults to None.

        Raises:
            Exception: tf_gnn output graph needs to be given if graph weights are not used as labels.

        TODO: Adjust to also use pyg_gnn probabilities.
        """
        if label_w_weights == False and output_graph == None:
            # Make sure we can get use the output values for each node as
            # a label for the network.
            raise Exception(
                "output graph is needed if graph weights are not used for labels"
            )

        is_start_index = 0
        is_end_index = 0

        sp_path = []
        sp_path_edges = []

        pos = {}

        for i in range(0, G.number_of_nodes()):
            # Construct node positions and find the source and target
            # of the querry.
            if G.nodes[i]["is_start"] == True:
                is_start_index = i
            if G.nodes[i]["is_end"] == False:
                is_end_index = i
            pos[i] = G.nodes[i]["pos"]

        for i in range(0, len(G.nodes)):
            # Get shortest path nodes ground truth.
            if G.nodes[i]["is_in_path"] == True:
                sp_path.append(i)

        edges_list = list(G.edges(data=True))

        for e in edges_list:
            if e[2]["is_in_path"] == True:
                sp_path_edges.append([e[0], e[1]])

        if label_w_weights:
            # Use the graph weights or the output of the network.
            labels_edge = nx.get_edge_attributes(G, "weight")
            node_labels = {}

            for key in labels_edge:
                labels_edge[key] = np.round(labels_edge[key], 2)

            for i in range(0, len(G.nodes)):
                node_labels[i] = round(G.nodes[i]["weight"], 2)
        else:
            node_labels, labels_edge = prob_labels(output_graph)

        nx.draw_networkx(G, pos=pos, labels=node_labels)

        nx.draw_networkx(
            G,
            pos=pos,
            nodelist=sp_path,
            edgelist=sp_path_edges,
            node_color="r",
            width=2,
            edge_color="r",
            # with_labels=True,
            labels=node_labels,
        )

        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edge)

        plt.show()

    def prob_labels(self, ot_graph) -> nx.Graph:
        """
        Takes a tf_gnn output graph and extracts the probabilities in a dict format used by nx.draw


        Args:
            ot_graph (_type_): tf_gnn output graph as given by the decoder.

        Returns:
            node_labels (dict): {node_num: prob_is_in_path 0..1}
            edge_labels (dict): {(s, t): prob_is_in_path 0..1}

        Those labels are passed to custom draw.

        """

        node_labels = {}
        edge_labels = {}
        print(ot_graph.node_sets["cities"][tfgnn.HIDDEN_STATE].numpy())
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
