"""Train the models.

Usage:
    python train.py --model model_name [--wandb] [--resume]

    model_name:
        1. MPNN: Message Passing Network
        2. DeepCoGCN: Co-embedding Deep GCN
        3. DeepGCN: DeepGCN
        4. GAT: Graph Attention Network

Example:
    use wandb:
        python train.py --model="MPNN" --wandb --project=PROJECT_NAME
        python train.py --model="DeepGCN" --wandb --project=PROJECT_NAME
        python train.py --model="GAT" --wandb --project=PROJECT_NAME
        python train.py --model="DeepCoGCN" --wandb --project=PROJECT_NAME
        python train.py --model="AdaptiveMPNN" --wandb --project=PROJECT_NAME

    on local machine:
        python train.py --model="MPNN"
        python train.py --model="DeepGCN"
        python train.py --model="GAT"
        python train.py --model="DeepCoGCN"
        python train.py --model="AdaptiveMPNN"

@Time    : 10/27/2022 8:45 PM
@Author  : Haodong Zhao
"""
import argparse
import os

from spreadnet.utils import yaml_parser
from spreadnet.utils.model_trainer import ModelTrainer, WAndBModelTrainer
import spreadnet.utils.log_utils as log_utils

default_dataset_yaml_path = os.path.join(
    os.path.dirname(__file__), "dataset_configs.yaml"
)

default_dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

log_save_path = os.path.join(os.path.dirname(__file__), "logs")

default_loss_type = "d"

parser = argparse.ArgumentParser(description="Train the model.")

parser.add_argument(
    "--model", help="Specify which model you want to train. ", required=True
)

parser.add_argument(
    "--wandb", help="Specify if train the model with wandb", action="store_true"
)

parser.add_argument("--project", help="Specify the wandb project.")

parser.add_argument(
    "--resume",
    help="Specify if it should resume training if training state is found",
    action="store_true",
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

parser.add_argument(
    "--loss-type",
    default=default_loss_type,
    help="Specify if you want to use the original loss (d), \
    weighted loss (w), \
    or euclidean weighted loss (e)",
)

args = parser.parse_args()
use_wandb = args.wandb
project_name = args.project

if use_wandb:
    if project_name is None:
        print(
            "ERROR: You need to specify the project name,"
            "if you want to train models with wandb."
        )
        exit(0)

model = args.model
yaml_path, model_save_path = "", ""
if model == "MPNN":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "message_passing_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "message_passing_network", "weights"
    )
elif model == "DeepCoGCN":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "co_graph_conv_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "co_graph_conv_network", "weights"
    )
elif model == "DeepGCN":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "deep_graph_conv_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "deep_graph_conv_network", "weights"
    )
elif model == "GAT":
    yaml_path = os.path.join(
        os.path.dirname(__file__), "graph_attention_network", "configs.yaml"
    )
    model_save_path = os.path.join(
        os.path.dirname(__file__), "graph_attention_network", "weights"
    )
elif model == "AdaptiveMPNN":
    yaml_path = os.path.join(os.path.dirname(__file__), "adaptive_mpnn", "configs.yaml")
    model_save_path = os.path.join(
        os.path.dirname(__file__), "adaptive_mpnn", "weights"
    )
else:
    print("Please check the model name. -h for more details.")
    exit()

resume = args.resume
dataset_yaml_path = args.dataset_config
dataset_path = args.dataset_path
dataset_path = dataset_path.replace("\\", "/")

train_console_logger = log_utils.init_console_only_logger(
    logger_name="train_console_logger"
)

loss_type = args.loss_type

configs = yaml_parser(yaml_path)
dataset_configs = yaml_parser(dataset_yaml_path)

train_configs = configs.train
model_configs = configs.model
data_configs = dataset_configs.data

train_console_logger.info(f"use_wandb: {use_wandb}")
if use_wandb:
    train_console_logger.info(f"project name: {project_name}")
train_console_logger.info(f"Training {model}...")

trainer = None
if use_wandb:
    trainer = WAndBModelTrainer(
        entity_name="uu-spreadnet",
        project_name=project_name,
        model_configs=model_configs,
        train_configs=train_configs,
        dataset_configs=data_configs,
        dataset_path=dataset_path,
        model_save_path=model_save_path,
        loss_type=loss_type,
    )
else:
    train_local_logger = log_utils.init_file_console_logger(
        logger_name="train_local_logger",
        log_save_path=log_save_path,
        exp_type=f"train_{model}",
    )
    trainer = ModelTrainer(
        model_configs=model_configs,
        train_configs=train_configs,
        dataset_configs=data_configs,
        dataset_path=dataset_path,
        model_save_path=model_save_path,
        loss_type=loss_type,
    )

trainer.train(resume)
