import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from pytictoc import TicToc

from modules.logging.logger import log_event

sys.path.append(Path(__file__).absolute().parent.parent)

t = TicToc()
JOBS = 4
SEED = 10

def _clean_and_expand_kmg_suffix(val):
    if isinstance(val, str):
        val = re.sub(r'\s', '', val.strip().upper())
        if val.endswith('%'):
            try:
                return str(float(val[:-1]) / 100)
            except ValueError:
                return val  # fallback if not a valid float
        elif val.endswith('K'):
            return val[:-1] + '000'
        elif val.endswith('M'):
            return val[:-1] + '000000'
        elif val.endswith('G'):
            return val[:-1] + '000000000'
    return str(val)  # ensure return is string

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
    Prints full stack trace on error for debugging.

    Returns: 
        dict: {'success': bool, 'error': str | None}
    """
    result = {
        "success": False,
        "error": None
    }

    try:
        log_event(msg_prefix, dataset_name, msg_suffix, "start")

        # Run the function
        runnable()

        log_event(msg_prefix, dataset_name, msg_suffix, "finish")

        result["success"] = True
        return result

    except Exception as e:
        # 1. Log the short error (for history/discord)
        log_event(msg_prefix, dataset_name, f"{msg_suffix} — {e}", "error")

        # 2. Print the FULL Stack Trace (Critical for debugging)
        log_event("--- STACK TRACE START ---")
        log_event(traceback.format_exc())
        log_event("--- STACK TRACE END ---")

        result["error"] = str(e)
        return result

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