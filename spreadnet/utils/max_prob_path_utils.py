import tensorflow as tf
import numpy as np


class MaxProbWalk:
    def max_probability_walk(self, G, start_node, end_node, prob_treshold):
        """Takes an output graph with a start and end node, outputs the nodes
        and edges if we take the maximum probability, either node or edge.

        Args:
            G (_type_): Oruput Graph
            start_node int: Start node.
            end_node int: End node.
            prob_treshold: float (0,1]
        Returns:
            _type_: Max probability nodes and edges list.

        Notes: The path can be incomplete.
        """

        max_prob_walk_nodes = []
        max_prob_walk_edges = []

        current_node = start_node
        max_prob_walk_nodes.append(start_node)

        while current_node != end_node:
            edges = G.out_edges(current_node, data=True)

            max_probability_edge = prob_treshold
            chosen_edge = None
            for e in edges:

                probability_edge = np.round(tf.nn.softmax(e[2]["logits"]).numpy()[1], 2)
                if (
                    probability_edge > max_probability_edge
                    and e[1] not in max_prob_walk_nodes
                    and (e[0], e[1]) not in max_prob_walk_edges
                    and (e[1], e[0]) not in max_prob_walk_edges
                ):
                    max_probability_edge = probability_edge

                    chosen_edge = (e[0], e[1])

            if chosen_edge is None:
                chosen_node = None
                max_probability_node = prob_treshold
                for e in edges:

                    probability_node = np.round(
                        tf.nn.softmax(dict(G.nodes(data=True))[e[1]]["logits"]).numpy()[
                            1
                        ],
                        2,
                    )

                    if (
                        probability_node > max_probability_node
                        and e[1] not in max_prob_walk_nodes
                    ):
                        max_probability_node = probability_node
                        chosen_node = e[1]

                if chosen_node is not None:
                    max_prob_walk_edges.append((current_node, chosen_node))
                    max_prob_walk_nodes.append(chosen_node)
                    current_node = chosen_node
                else:
                    return max_prob_walk_nodes, max_prob_walk_edges
            else:
                max_prob_walk_edges.append(chosen_edge)
                max_prob_walk_nodes.append(chosen_edge[1])
                current_node = chosen_edge[1]

        return max_prob_walk_nodes, max_prob_walk_edges
