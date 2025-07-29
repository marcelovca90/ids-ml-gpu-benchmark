import inspect
import logging
import os
import sys

import colorlog
from colorama import Fore, Style
from datetime import datetime

formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s")

handler_stdout = logging.StreamHandler(stream=sys.stdout)
handler_stdout.setFormatter(formatter)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
handler_file = logging.FileHandler(f"logs/{timestamp}.log", mode="w")
handler_file.setFormatter(formatter)

logger = logging.Logger("iot-threat-classifier")
# logger = logging.getLogger("iot-threat-classifier")
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
