import logging
from datetime import datetime
import sys
import os


def create_log_dir(file_path: str, exp_type):
    """Creates a directory for logs with date in filename.

    Returns:
        log_file_name: str
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_name_format = datetime.today().strftime("%d_%m_%y__%H_%M")

    log_file_name = f"{file_path}/{exp_type}_log_{file_name_format}.log"

    return log_file_name


def init_logger(
    logger_name="gnn_logger", exp_type="unspecified", file_path="unspecified"
):
    """Initializes logger instance with experiment details.

    Args:
        logger_name (str, optional): _description_. Defaults to "gnn_logger".
        exp_type (str, optional):Experiment type. Defaults to "unspecified".
        file_path (str, optional):Log destination. Defaults to "unspecified".
        author (str, optional): Author of the experiment. Defaults to "anon".
    Returns:
        gnn_logger: logger instance
    """
    gnn_logger = logging.getLogger(logger_name)
    try:
        # create a logger with name
        gnn_logger = logging.getLogger(logger_name)
        # set level to info
        gnn_logger.setLevel(logging.INFO)
        format_str = "%(asctime)s | %(levelname)s | %(message)s"
        formatter = logging.Formatter(format_str)

        # prints logs as terminal/cell output
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        # writes to a file

        log_path = create_log_dir(file_path, exp_type)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # attaching logger to the output streams
        gnn_logger.addHandler(file_handler)
        gnn_logger.addHandler(stdout_handler)
    except Exception as e:
        print(e)
    return gnn_logger


def shutdown_logger():
    """Shuts down logging instance and related files."""
    return logging.shutdown()
