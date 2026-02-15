import logging
import os
from multiagent_rag.utils.constants import LoggingConstants

def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(LoggingConstants.DEFAULT_LOG_LEVEL)

    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter(LoggingConstants.LOG_FORMAT, datefmt=LoggingConstants.DATE_FORMAT))
    logger.addHandler(c_handler)

    log_dir = os.path.dirname(LoggingConstants.LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    f_handler = logging.FileHandler(LoggingConstants.LOG_FILE_PATH)
    f_handler.setFormatter(logging.Formatter(LoggingConstants.LOG_FORMAT, datefmt=LoggingConstants.DATE_FORMAT))
    logger.addHandler(f_handler)

    return logger
