"""Train the models.

Usage:
    python train.py --model model_name [--wandb]

    model_name:
        1. MPNN: EncodeProcessDecode
        2. GCN: Co-embedding Deep GCN

Example:
    python train.py --model="MPNN" --wandb
    python train.py --model="GCN" --wandb
    python train.py --model="GAT" --wandb

@Time    : 10/27/2022 8:45 PM
@Author  : Haodong Zhao
"""
import argparse
import os

from spreadnet.utils import yaml_parser
from spreadnet.utils.model_trainer import ModelTrainer, WAndBModelTrainer

default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "dataset_configs.yaml"
)

default_dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

parser = argparse.ArgumentParser(description="Train the model.")

parser.add_argument(
    "--model", help="Specify which model you want to train. ", required=True
)

parser.add_argument(
    "--wandb", help="Specify if train the model with wandb", action="store_true"
)

parser.add_argument(
    "--dataset-config",
    default=default_dataset_yaml_path,
    help="Specify the path of the dataset config file. ",
)

parser.add_argument(
    "--dataset-path",
    default=default_dataset_path,
    help="Specify the path of the dataset",
)

args = parser.parse_args()
model = args.model

yaml_path, model_save_path = "", ""
if model == "MPNN":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "encode_process_decode", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "encode_process_decode", "weights"
    )

elif model == "GCN":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "co_graph_conv_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "co_graph_conv_network", "weights"
    )
elif model == "GAT":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "graph_attention_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "graph_attention_network", "weights"
    )
else:
    print("Please check the model name. -h for more details.")
    exit()

dataset_yaml_path = args.dataset_config
dataset_path = args.dataset_path
dataset_path = dataset_path.replace("\\", "/")

configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)

train_configs = configs.train
model_configs = configs.model
data_configs = dataset_configs.data

use_wandb = args.wandb
print("use_wandb: ", use_wandb)

trainer = None
if use_wandb:
    trainer = WAndBModelTrainer(
        project_name="UU-SpreadNet-Train",
        model_configs=model_configs,
        train_configs=train_configs,
        dataset_configs=data_configs,
        dataset_path=dataset_path,
        model_save_path=model_save_path,
    )
else:
    trainer = ModelTrainer(
        model_configs=model_configs,
        train_configs=train_configs,
        dataset_configs=data_configs,
        dataset_path=dataset_path,
        model_save_path=model_save_path,
    )

trainer.train()
