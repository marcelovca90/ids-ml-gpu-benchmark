import json
import numbers
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from operator import getitem
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
# from featurewiz import FeatureWiz
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import InstanceHardnessThreshold, TomekLinks
from pytictoc import TicToc
from scipy import stats
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import ExtraTreeClassifier

sys.path.append(Path(__file__).absolute().parent.parent)

from modules.logging.logger import log_print
from modules.logging.webhook import post_disc

t = TicToc()
JOBS = 4
SEED = 10

def _convert_to_int(x):
    try:
        if '.' in x:
            return int(float(x))
        elif '0x' in x:
            return int(x, 16)
        else:
            return int(x)
    except:
        return 0

def _replace_values(df, column, old_value, new_value):
    df.loc[(df[column] == old_value), column] = new_value

def now():
    now = datetime.now()
    yyyymmdd_hhmmss_part = now.strftime('%Y-%m-%d %H:%M:%S')
    ms_part = f'{int(now.microsecond / 1000):03d}'
    return f'{yyyymmdd_hhmmss_part},{ms_part}'

def safe_exec(runnable, msg_prefix, dataset_name, msg_suffix):
    """
    Wraps execution with standard logging and error handling.
    
    Args:
        runnable (callable): A lambda or function to execute (e.g., lambda: obj.pipeline())
        msg_prefix (str): Log prefix (e.g., "[01/05]")
        dataset_name (str): Name of the dataset class
        msg_suffix (str): Log suffix (e.g., "b=False f=1.0 s=17")
    
    Returns:
        bool: True if successful, False if an exception occurred.
    """

    def _log_event(prefix, dataset_name, suffix, stage):
        """
        Unified logging for any pipeline stage.
        stage ∈ {"start", "finish", "error"}
        """
        if stage == "start":
            msg = f"{prefix} Started {dataset_name} ({suffix})."
        elif stage == "finish":
            msg = f"{prefix} Finished {dataset_name} ({suffix})."
        elif stage == "error":
            msg = f"{prefix} ERROR in {dataset_name} ({suffix})."
        else:
            raise ValueError(f"Unknown logging stage: {stage}")

        log_print(msg)
        post_disc(msg)

    try:
        _log_event(msg_prefix, dataset_name, msg_suffix, "start")
        runnable()
        _log_event(msg_prefix, dataset_name, msg_suffix, "finish")
        return True
    except Exception as e:
        _log_event(msg_prefix, dataset_name, f"{msg_suffix} — {e}", "error")
        exit()
        return False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)