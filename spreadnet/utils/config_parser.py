"""Parse the configuration file.

@Time    : 9/22/2022 6:35 PM
@Author  : Haodong Zhao
"""
import argparse

import yaml


def yaml_parser(yaml_path: str):
    """Parse `.yaml` config file.

    Args:
        yaml_path: The path of the `.yaml` config file.

    Returns:
        A dictionary that contains the configs.
    """
    with open(yaml_path, "r") as file:
        configs = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))

    return configs
