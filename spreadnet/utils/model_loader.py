from spreadnet.pyg_gnn.models import SPCoDeepGCNet, MPNN
from spreadnet.pyg_gnn.models.deepGCN.sp_deepGCN import SPDeepGCN
from spreadnet.pyg_gnn.models.graph_attention_network.sp_gat import SPGATNet


def load_model(model_name, model_configs, device):
    """load the entire model.

    :param model_name: the name of the model
    :return: the loaded model
    """
    if model_name == "MPNN":
        return MPNN(
            node_in=model_configs["node_in"],
            edge_in=model_configs["edge_in"],
            node_out=model_configs["node_out"],
            edge_out=model_configs["edge_out"],
            latent_size=model_configs["latent_size"],
            num_message_passing_steps=model_configs["num_message_passing_steps"],
            num_mlp_hidden_layers=model_configs["num_mlp_hidden_layers"],
            mlp_hidden_size=model_configs["mlp_hidden_size"],
        ).to(device)
    if model_name == "DeepCoGCN":
        return SPCoDeepGCNet(
            node_in=model_configs["node_in"],
            edge_in=model_configs["edge_in"],
            gcn_hidden_channels=model_configs["gcn_hidden_channels"],
            gcn_num_layers=model_configs["gcn_num_layers"],
            mlp_hidden_channels=model_configs["mlp_hidden_channels"],
            mlp_hidden_layers=model_configs["mlp_hidden_layers"],
            node_out=model_configs["node_out"],
            edge_out=model_configs["edge_out"],
        ).to(device)
    if model_name == "GAT":
        return SPGATNet(
            num_hidden_layers=model_configs["num_hidden_layers"],
            in_channels=model_configs["in_channels"],
            hidden_channels=model_configs["hidden_channels"],
            out_channels=model_configs["out_channels"],
            heads=model_configs["heads"],
            # dropout=model_configs[""],
            add_self_loops=model_configs["add_self_loops"],
            bias=model_configs["bias"],
            edge_hidden_channels=model_configs["edge_hidden_channels"],
            edge_out_channels=model_configs["edge_out_channels"],
            edge_num_layers=model_configs["edge_num_layers"],
            edge_bias=model_configs["edge_bias"],
            encode_node_in=model_configs["encode_node_in"],
            encode_edge_in=model_configs["encode_edge_in"],
            encode_node_out=model_configs["encode_node_out"],
            encode_edge_out=model_configs["encode_edge_out"],
        ).to(device)
    if model_name == "DeepGCN":
        return SPDeepGCN(
            node_in=model_configs["node_in"],
            edge_in=model_configs["edge_in"],
            encoder_hidden_channels=model_configs["encoder_hidden_channels"],
            encoder_layers=model_configs["encoder_layers"],
            gcn_hidden_channels=model_configs["gcn_hidden_channels"],
            gcn_layers=model_configs["gcn_layers"],
            decoder_hidden_channels=model_configs["decoder_hidden_channels"],
            decoder_layers=model_configs["decoder_layers"],
        ).to(device)
    raise Exception("Invalid model name")
