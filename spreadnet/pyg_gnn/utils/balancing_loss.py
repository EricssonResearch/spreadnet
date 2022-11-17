import torch


class BalancingLoss:
    """Adding additional penalties to the nodes and edges losses."""

    def __init__(
        self,
        node_true,
        node_pred,
        node_losses,
        edge_true,
        edge_pred,
        edge_losses,
        edge_data,
    ):
        self.node_true = node_true
        self.node_pred = node_pred
        self.node_losses = node_losses
        self.edge_true = edge_true
        self.edge_pred = edge_pred
        self.edge_losses = edge_losses
        self.edge_data = edge_data

        # getting the node and edge multiplication factor:
        # nodes ratio = nr_nodes_not_in_path / nr_nodes_in_path
        count_nodes_in_path = torch.bincount(
            node_true
        )  # tensor([nr of 0's, nr of 1's])
        self.node_mult_factor = count_nodes_in_path[0] / count_nodes_in_path[1]

        # edges ratio = nr_edges_not_in_path / nr_edges_in_path
        count_edges_in_path = torch.bincount(edge_true)
        self.edge_mult_factor = count_edges_in_path[0] / count_edges_in_path[1]

        # converts logits to their classes
        node_0 = node_pred[:, 0]
        node_1 = node_pred[:, 1]
        self.node_pred_classes = torch.where(node_0 > node_1, 0, 1)

        edge_0 = edge_pred[:, 0]
        edge_1 = edge_pred[:, 1]
        self.edge_pred_classes = torch.where(edge_0 > edge_1, 0, 1)

        self.updated_node_losses = node_losses
        self.updated_edge_losses = edge_losses

        # adds penalties to the losses with a node wrongly classified as "not in path"
        self.updated_node_losses = torch.where(
            torch.logical_and(self.node_pred_classes == 0, self.node_true == 1),
            self.node_losses * self.node_mult_factor,
            self.node_losses,
        )

        # adds penalties to the losses with an edge wrongly classified as "not in path"
        self.updated_edge_losses = torch.where(
            torch.logical_and(self.edge_pred_classes == 0, self.edge_true == 1),
            self.edge_losses * self.edge_mult_factor,
            self.edge_losses,
        )

    def get_weighted_loss(self):
        return {
            "nodes": torch.mean(self.updated_node_losses),
            "edges": torch.mean(self.updated_edge_losses),
        }

    def get_sequence_weighted_loss(self):
        for i in range(len(self.edge_data)):
            (source_node, target_node) = self.edge_data[i]

            # 1: if edge is correctly classified as "in path"
            edge_prediction = self.edge_pred_classes[i]
            edge_truth = self.edge_true[i]

            if edge_prediction == 1 and (edge_prediction == edge_truth):

                # 1a: penalize the wrongly classified source node
                # (which should be in path)
                if self.node_pred_classes[source_node] == 0:
                    self.updated_node_losses[source_node] *= self.node_mult_factor

                # 1b: penalize the wrongly classified target node
                # (which should be in path)
                if self.node_pred_classes[target_node] == 0:
                    self.updated_node_losses[target_node] *= self.node_mult_factor

            # 2: if 2 neighbouring nodes are correctly classified to be "in path"
            # penalize the wrongly classified edge (which should be in path)
            if (
                (self.node_pred_classes[source_node] == self.node_true[source_node])
                and (self.node_pred_classes[target_node] == self.node_true[target_node])
                and (self.node_true[source_node] == 1)
                and (self.node_true[target_node] == 1)
            ):
                if self.edge_pred_classes[i] == 0:
                    self.updated_edge_losses[i] *= self.edge_mult_factor

        return {
            "nodes": torch.mean(self.updated_node_losses),
            "edges": torch.mean(self.updated_edge_losses),
        }
