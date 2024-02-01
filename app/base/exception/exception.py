import os

from loguru import logger


def show_log(message, level: str = "info"):
    if level == "debug" and os.getenv('DEBUG'):
        logger.debug(str(message))
    elif level == "error":
        logger.error(str(message))
    else:
        logger.info(str(message))
