import logging
from datetime import datetime
import sys
import os


format_str = "%(asctime)s | %(levelname)s | %(message)s"
formatter = logging.Formatter(format_str)


def init_console_only_logger(logger_name):
    only_console_logger = logging.getLogger(logger_name)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    only_console_logger.addHandler(stdout_handler)
    only_console_logger.setLevel(logging.INFO)

    return only_console_logger


def create_log_dir(log_save_path, exp_type):
    """Creates a directory for logs with date in filename.

    Returns:
            log_file_name: str
    """
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)

    file_name_format = datetime.now().strftime("%H-%M-%S_%d-%m-%y")

    log_dir_path = f"{log_save_path}/{exp_type}_log_{file_name_format}.log"

    return log_dir_path


def init_file_only_logger(logger_name, log_save_path, exp_type):
    file_only_logger = logging.getLogger(logger_name)

    log_dir_path = create_log_dir(log_save_path, exp_type)
    file_handler = logging.FileHandler(log_dir_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    file_only_logger.addHandler(file_handler)
    file_only_logger.setLevel(logging.INFO)

    return file_only_logger


def init_file_console_logger(logger_name, log_save_path, exp_type):
    file_console_logger = logging.getLogger(logger_name)
    file_console_logger.setLevel(logging.INFO)
    log_dir_path = create_log_dir(log_save_path, exp_type)
    file_handler = logging.FileHandler(log_dir_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_console_logger.addHandler(file_handler)
    file_console_logger.addHandler(stdout_handler)

    return file_console_logger


def shutdown_logger():
    """Shuts down logging instance and related files."""
    return logging.shutdown()
