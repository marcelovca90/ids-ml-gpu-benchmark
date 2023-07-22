import inspect
import logging
import sys

import colorlog
from colorama import Fore, Style

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s")

handler_stdout = logging.StreamHandler(stream=sys.stdout)
handler_stdout.setFormatter(formatter)

handler_file = logging.FileHandler(f"data_loader.log", mode="w")
handler_file.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler_stdout)
logger.addHandler(handler_file)


def function_call_logger(func):

    def wrapper(*args, **kwargs):
        log_print(f"Function started.", func.__name__)
        return func(*args, **kwargs)

    return wrapper


def log_print(message, calling_function=None):
    if not calling_function:
        calling_function = inspect.stack()[1].function
    logger.info(
        f'{Fore.RED}[ {calling_function} ]{Style.RESET_ALL} {message}')
