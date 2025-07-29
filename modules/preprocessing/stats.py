from typing import List, Union
from pandas import DataFrame

from modules.logging.logger import function_call_logger, log_print


@function_call_logger
def log_data_types(df: DataFrame) -> None:
    for line in df.dtypes.to_string().split('\n'):
        log_print(line)


@function_call_logger
def log_memory_usage(df: DataFrame) -> None:
    total_memory = df.memory_usage(deep=True).sum()
    log_print(f"{total_memory / (1024 ** 2):.2f} MB")


@function_call_logger
def log_col_data(df: DataFrame, col: Union[str, List[str]]):
    log_print(f"{'Column':<40}{'#Unique':<12}{'Dtype':<12}{'MemoryMB'}")
    if isinstance(col, str):
        mem = df[col].memory_usage(deep=True) / 1024 / 1024
        log_print(f'{col:<40}{df[col].nunique():<12}{str(df[col].dtype):<12}{mem:.2f}')
    elif isinstance(col, list):
        for c in col:
            mem = df[c].memory_usage(deep=True) / 1024 / 1024
            log_print(f'{c:<40}{df[c].nunique():<12}{str(df[c].dtype):<12}{mem:.2f}')


@function_call_logger
def log_value_counts(df: DataFrame, col: str) -> None:
    abs_counts = df[col].value_counts().to_string().split('\n')
    rel_counts = df[col].value_counts(normalize=True).to_string().split('\n')
    for line_abs, line_rel in zip(abs_counts, rel_counts):
        log_print(line_abs + '\t' + line_rel.split(' ')[-1])
