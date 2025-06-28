import json
import math
import numpy as np
import os
import re
from abc import ABC, abstractmethod

import pandas as pd
from dtype_diet import optimize_dtypes, report_on_dataframe
from scipy.stats import skew
from type_infer.api import infer_types
from typing_extensions import Self
from ydata_profiling import ProfileReport

from modules.logging.logger import function_call_logger, log_print
from modules.preprocessing.stats import (
    log_col_data, log_data_types, log_memory_usage, log_value_counts
)


class BasePreprocessingPipeline(ABC):

    def __init__(self, binarize=False) -> None:
        self.data: pd.DataFrame = None
        self.folder: str = None
        self.name: str = None
        self.target: str = None
        self.seed = 42
        self.binarize = binarize
        self.kind = 'Binary' if self.binarize else 'Multiclass'
        self.metadata: dict = {}
        self.complexity: dict = {}
        self.profile: ProfileReport = None

    @function_call_logger
    def preload(self) -> None:
        base_path = os.path.join(os.getcwd(), self.folder, 'generated')
        log_print(f'Loading cached files from \'{base_path}\'...')
        self.data = pd.read_parquet(os.path.join(base_path, self.name + '.parquet'))
        with open(os.path.join(base_path, 'metadata.json')) as json_file:
            self.metadata = json.load(json_file)

    @abstractmethod
    def prepare(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @function_call_logger
    def sanitize(self) -> None:
        log_print('Value counts before sanitization:')
        log_value_counts(self.data, self.target)
        for col in self.data.columns:
            dtype = self.data[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                self.data[col] = self.data[col].fillna(0)
            elif pd.api.types.is_float_dtype(dtype):
                self.data[col] = self.data[col].fillna(0.0)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                self.data[col] = self.data[col].fillna('0')
            else:
                log_print(f"Skipping column {col} with unsupported dtype: {dtype}")
        log_print('Value counts after sanitization:')
        log_value_counts(self.data, self.target)

    @function_call_logger
    def infer_dtypes(self) -> None:
        log_print("Data types and memory usage before type inference:")
        log_data_types(self.data)
        log_memory_usage(self.data)

        # Map from inferred type labels to pandas dtypes
        type_map = {
            'integer': 'int32',
            'float': 'float32',
            'categorical': 'category',
            'binary': 'category',
            'tags': 'category'
        }

        try:
            inferred_dtypes = infer_types(self.data)
        except Exception as e:
            log_print(f"Failed to infer types: {e}")
            return

        for col in self.data.columns:
            current_dtype = self.data[col].dtype
            inferred_label = inferred_dtypes.dtypes.get(col)
            target_dtype = type_map.get(inferred_label)

            series = self.data[col]

            # Promote low-precision numerics before conversion
            if pd.api.types.is_integer_dtype(series) and series.dtype in ('int8', 'int16', 'uint8', 'uint16'):
                series = series.astype('int32')
            elif pd.api.types.is_float_dtype(series) and series.dtype == 'float16':
                series = series.astype('float32')

            log_print(f"{col:<40} {str(current_dtype):<8} → {inferred_label or 'unknown':<11} ({target_dtype or 'skip'})")

            if target_dtype:
                try:
                    self.data[col] = series.astype(target_dtype)
                except Exception as e:
                    log_print(f"Warning: Could not convert column '{col}' to {target_dtype}: {e}")

        # Drop identifier columns
        identifier_cols = list(
            [x for x in inferred_dtypes.identifiers.keys() if x != self.target]
        )
        if identifier_cols:
            log_print(f"Dropping identifier columns: {identifier_cols}")
            self.data = self.data.drop(columns=identifier_cols)
        else:
            log_print("No identifier columns to drop.")

        log_print("Data types and memory usage after type inference:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def convert_to_numeric(self) -> None:
        log_print("Data types and memory usage before numeric conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)

        for col in self.data.drop(columns=[self.target]).select_dtypes(include=["object", "string"]).columns:
            original_dtype = self.data[col].dtype
            try:
                self.data[col] = pd.to_numeric(self.data[col], errors='raise')
                new_dtype = self.data[col].dtype
                log_print(f"Column '{col}' converted from {original_dtype} to {new_dtype}")
            except Exception:
                log_print(f"Column '{col}' could not be converted from {original_dtype}")

        log_print("Data types and memory usage after numeric conversion:")
        log_data_types(self.data)
        log_memory_usage(self.data)

    @function_call_logger
    def round_floats(self, round_decimals=3) -> None:
        log_print(f"Number of unique values per column before rounding to {round_decimals} decimal places:")
        for col in self.data.columns:
            log_print(f"{col}\t{self.data[col].nunique()}")
        # Select only float columns
        float_cols = self.data.drop(columns=[self.target]).select_dtypes(include='float').columns
        self.data[float_cols] = self.data[float_cols].round(decimals=round_decimals)
        log_print(f"Number of unique values per column after rounding to {round_decimals} decimal places:")
        for col in self.data.columns:
            log_print(f"{col}\t{self.data[col].nunique()}")

    @function_call_logger
    def handle_hc_numeric_columns(self, handle_num_mode='discretize') -> None:
        HIGH_CARD = 2**16       # 65,536
        VERY_HIGH_CARD = 2**20  # 1,048,576
        if handle_num_mode == 'discretize':
            numeric_cols = self.data.drop(columns=[self.target]).select_dtypes(include=["number"]).columns
            numeric_cols = [col for col in numeric_cols if self.data[col].nunique(dropna=False) > HIGH_CARD]
            log_print("Uniques, dtypes, and memory usage before processing:")
            log_col_data(self.data, numeric_cols)
            for col in numeric_cols:
                nunique = self.data[col].nunique(dropna=False)
                if nunique > VERY_HIGH_CARD:
                    qcut_result = pd.qcut(self.data[col], q=100, duplicates='drop')
                    labels = [f'Q{i+1}' for i in range(qcut_result.cat.categories.size)]
                    self.data[col] = qcut_result.cat.rename_categories(labels)
                elif nunique > HIGH_CARD:
                    cut_result = pd.cut(self.data[col], bins=100, duplicates='drop')
                    labels = [f'Q{i+1}' for i in range(cut_result.cat.categories.size)]
                    self.data[col] = cut_result.cat.rename_categories(labels)
            log_print("Uniques, dtypes, and memory usage after processing:")
            log_col_data(self.data, numeric_cols)

    @function_call_logger
    def handle_port_columns(self) -> None:

        def is_probably_port_column(series: pd.Series, colname: str) -> bool:
            include_filter = [
                'srcport', 'dstport', 'src_port', 'dst_port', 'sport', 'dport',
                'srcp', 'dstp', 'src-p', 'dst-p', 'tcp.srcport', 'tcp.dstport',
                'udp.srcport', 'udp.dstport', 'port', 'prt', 'orig_p', 'resp_p',
                'srcprt', 'dstprt'
            ]
            exclude_filter = [
                'rate', 'ltm', 'is_', 'ct_', 'flag', 'type', 'pkts', 'rtt', 'time',
                'ttl', 'len', 'bytes', 'flow', 'duration'
            ]
            colname_lc = colname.lower()
            if not any(kw in colname_lc for kw in include_filter):
                return False
            if any(kw in colname_lc for kw in exclude_filter):
                return False
            if not pd.api.types.is_numeric_dtype(series):
                return False
            return series.dropna().between(0, 65535).mean() > 0.95

        def semantically_bin_port(port_series: pd.Series) -> pd.Series:

            port_map = {
                # Common Application Protocols
                20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
                53: "DNS", 67: "DHCP", 68: "DHCP", 69: "TFTP", 80: "HTTP",
                110: "POP3", 123: "NTP", 137: "NetBIOS", 143: "IMAP",
                161: "SNMP", 443: "HTTPS", 465: "SMTPS", 993: "IMAPS", 995: "POP3S",

                # Databases & Admin Access
                3306: "MySQL", 3389: "RDP", 5900: "VNC",

                # IoT Messaging Protocols
                1883: "MQTT", 8883: "MQTT-TLS", 8080: "MQTT-WS", 8081: "MQTT-WSS",
                5060: "SIP",  # Session Initiation Protocol for VoIP/IoT

                # Industrial Control Systems
                502: "Modbus",          # SCADA
                20000: "DNP3",          # SCADA/Power
                2222: "EtherNet/IP",    # Rockwell
                44818: "CIP",           # Common Industrial Protocol
                2404: "IEC-104",        # European automation
                789: "CrimsonV3",       # HMI software

                # Security & Logging
                514: "Syslog",

                # Known Malware / Suspicious
                12345: "NetBus",        # Malware
                31337: "BackOrifice"    # Malware
            }

            def label_port(p):
                if pd.isna(p) or not (0 <= p <= 65535):
                    return "Other"
                try:
                    p = int(p)
                except:
                    return "Other"
                if p in port_map:
                    return port_map[p]
                elif p <= 255:
                    return "System_0_255"
                elif p <= 1023:
                    return "System_256_1023"
                elif p <= 49151:
                    return "Registered_1024_49151"
                elif p <= 57343:
                    return "Dynamic_49152_57343"
                else:
                    return "Dynamic_57344_65535"

            return port_series.map(label_port).astype("category")

        # Port binning check
        prob_port_cols = []
        for col in self.data.drop(columns=[self.target]).columns:
            if is_probably_port_column(self.data[col], col):
                prob_port_cols.append(col)

        # Port binning for probable cols
        if prob_port_cols:
            log_print(f"Port-like columns identified: {prob_port_cols}")
            log_print("Uniques, dtypes, and memory usage before semantic binning:")
            log_col_data(self.data, prob_port_cols)
            for col in prob_port_cols:
                self.data[col] = semantically_bin_port(self.data[col])
            log_print("Uniques, dtypes, and memory usage after semantic binning:")
            log_col_data(self.data, prob_port_cols)
        else:
            log_print("No probable port columns detected.")

    @function_call_logger
    def handle_object_columns(self, handle_obj_mode='auto') -> None:
        HIGH_CARD = 2**14       # 16,384
        VERY_HIGH_CARD = 2**18  # 262,144
        MAX_BUCKETS = 2**16     # 65,536

        if handle_obj_mode != 'keep':
            log_print("Data types and memory usage before encoding:")
            log_data_types(self.data)
            log_memory_usage(self.data)
            object_cols = self.data.drop(columns=[self.target]).select_dtypes(include='object').columns
            if len(object_cols) == 0:
                log_print("No object columns to process.")
                return

            for col in object_cols:
                n_unique = self.data[col].nunique(dropna=False)
                log_print(f"Processing column '{col}' with {n_unique} unique values")

                if handle_obj_mode == 'drop':
                    self.data = self.data.drop(columns=[col])
                    log_print(f"Dropped object column '{col}'")
                    continue

                # Standard encoding strategies
                elif handle_obj_mode == 'encode_cat' or (handle_obj_mode == 'auto' and n_unique < HIGH_CARD):
                    # Low-cardinality: use pandas categorical dtype for memory and model efficiency
                    self.data[col] = self.data[col].astype('category')
                    log_print(f"Encoded '{col}' as category")

                elif handle_obj_mode == 'auto' and HIGH_CARD <= n_unique <= VERY_HIGH_CARD:
                    # Medium-cardinality: choose encoding based on value distribution skewness
                    value_counts = self.data[col].value_counts(dropna=False)
                    skewness = skew(value_counts.values.astype(float))

                    if abs(skewness) > 1:
                        # Highly skewed -> Frequency encoding (proportion of dataset)
                        freq_map = value_counts / len(self.data)
                        self.data[col] = self.data[col].map(freq_map).fillna(0).astype('float32')
                        log_print(f"Applied frequency encoding to '{col}' (skew = {skewness:.2f})")
                    else:
                        # Balanced distribution -> Count encoding (absolute frequency)
                        self.data[col] = self.data[col].map(value_counts).fillna(0).astype('int32')
                        log_print(f"Applied count encoding to '{col}' (skew = {skewness:.2f})")

                else:
                    # Very high cardinality: use hashing trick to bucket unique values
                    col_str = self.data[col].astype(str)
                    n_buckets_raw = 2 ** math.ceil(math.log2(n_unique * 1.1))
                    n_buckets = min(n_buckets_raw, MAX_BUCKETS)
                    hashed = pd.util.hash_pandas_object(col_str, index=False).astype('int64') + self.seed
                    self.data[col] = (hashed % n_buckets).astype('int32')
                    log_print(f"Hashed '{col}' to {n_buckets} buckets")

            log_print("Data types and memory usage after encoding:")
            log_data_types(self.data)
            log_memory_usage(self.data)

    @function_call_logger
    def drop_high_unique_columns(self) -> None:
        UNIQUE_THRESHOLD = 0.999
        n_rows = len(self.data)
        high_unique_cols = [
            col for col in self.data.drop(columns=[self.target]).columns
            if self.data[col].nunique(dropna=False) / n_rows >= UNIQUE_THRESHOLD
            and self.data[col].nunique(dropna=False) < n_rows  # exclude already handled all-unique
        ]
        if high_unique_cols:
            log_print(f"Dropped high-unique columns (≥{UNIQUE_THRESHOLD:.0%} unique): {high_unique_cols}")
            self.data = self.data.drop(columns=high_unique_cols)

    @function_call_logger
    def drop_infinite_rows(self) -> None:
        mask = self.data.isin([np.inf, -np.inf]).any(axis=1)
        self.data = self.data[~mask]
        log_print(f"Dropped {mask.sum()} rows containing ±inf")

    @function_call_logger
    def drop_na_duplicates(self) -> None:
        num_nas_before = self.data.isna().sum().sum()
        num_duplicates_before = self.data.duplicated().sum()
        log_print(f"NAs before cleaning: {num_nas_before}")
        log_print(f"Duplicates before cleaning: {num_duplicates_before}")
        if num_nas_before == 0 and num_duplicates_before == 0:
            log_print("No NAs or duplicates found; skipping cleanup.")
        else:
            self.data.dropna(axis='columns', how='all', inplace=True)
            self.data.dropna(axis='index', how='any', inplace=True)
            self.data.drop_duplicates(inplace=True)
            num_nas_after = self.data.isna().sum().sum()
            num_duplicates_after = self.data.duplicated().sum()
            log_print(f"NAs after cleaning: {num_nas_after}")
            log_print(f"Duplicates after cleaning: {num_duplicates_after}")
            log_print(f"Memory usage after cleaning:")
            log_memory_usage(self.data)

    @function_call_logger
    def shrink_numeric_dtypes(self, shrink_mode='conservative') -> None:
        if shrink_mode:
            log_print("Data types and memory usage before shrinkage:")
            log_data_types(self.data)
            log_memory_usage(self.data)
            df_report = report_on_dataframe(self.data, unit="MB", optimize="computation")
            if shrink_mode == 'aggressive':
                self.data = optimize_dtypes(self.data, df_report)
                for col in self.data.drop(columns=[self.target]).select_dtypes(include=['integer', 'float']).columns:
                    self.data[col] = pd.to_numeric(
                        self.data[col], downcast='integer' if self.data[col].dtype.kind == 'i' else 'float')
            elif shrink_mode == 'conservative':
                for col in self.data.drop(columns=[self.target]).select_dtypes(include=["number"]).columns:
                    if pd.api.types.is_integer_dtype(self.data[col]):
                        self.data[col] = self.data[col].astype('int32')
                    elif pd.api.types.is_float_dtype(self.data[col]):
                        self.data[col] = self.data[col].astype('float32')
            log_print("Data types and memory usage after shrinkage:")
            log_data_types(self.data)
            log_memory_usage(self.data)

    @function_call_logger
    def clean_and_sort_columns(self) -> None:
        # Rename all columns except the target
        rename_map = {
            col: re.sub(r"[^A-Za-z0-9]", "_", col).upper()
            for col in self.data.columns if col != self.target
        }
        self.data = self.data.rename(columns=rename_map)
        # Sort: features first, target last
        cols = [col for col in self.data.columns if col != self.target]
        cols.append(self.target)
        log_print(f'Columns sorted according to {cols}.')
        self.data = self.data.reindex(columns=cols)

    @function_call_logger
    def reset_index(self) -> None:
        self.data.reset_index(drop=True, inplace=True)

    @function_call_logger
    def update_metadata(self) -> None:
        self.metadata = {
            'kind': self.kind,
            'columns': self.data.columns.tolist(),
            'description': self.data.describe().to_dict(),
            'dtypes': self.data.dtypes.apply(str).to_dict(),
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'value_counts': self.data[self.target].value_counts().to_dict(),
            'complexity': self.complexity.get(self.name, {})
        }

    @function_call_logger
    def compute_complexity(self, complexity_mode=None) -> None:
        if complexity_mode:
            if complexity_mode == 'cpu':
                from modules.preprocessing.complexity_cpu import (
                    compute_all_complexity_measures, smart_categorical_encode)
            elif complexity_mode == 'gpu':
                from modules.preprocessing.complexity_gpu_v1 import (
                    compute_all_complexity_measures, smart_categorical_encode)
            X, y = self.data.drop(columns=[self.target]), self.data[self.target]
            X, y = smart_categorical_encode(X, y)
            self.complexity = compute_all_complexity_measures(X, y)

    @function_call_logger
    def compute_profile(self, profile_mode='minimal') -> None:
        if profile_mode:
            profile_kwargs = {'df': self.data, 'minimal': profile_mode=='minimal'}
            self.profile = ProfileReport(**profile_kwargs)

    @function_call_logger
    def save(self, csv=False, parquet=True, metadata=True, profile=True) -> None:
        kind_suffix = f'_{self.kind}' if f'_{self.kind}' not in self.name else ''
        base_dir = os.path.join(os.getcwd(), self.folder, 'generated')
        os.makedirs(base_dir, exist_ok=True)

        if csv:
            csv_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.csv')
            log_print(f'Persisting dataset to \'{csv_filename}\'...')
            self.data.to_csv(csv_filename, index=False)

        if parquet:
            parquet_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.parquet')
            log_print(f'Persisting dataset to \'{parquet_filename}\'...')
            self.data.to_parquet(parquet_filename, index=False)

        if metadata:
            metadata_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.json')
            log_print(f'Persisting metadata to \'{metadata_filename}\'...')
            with open(metadata_filename, 'w') as fp:
                json.dump(self.metadata, fp, indent=4, default=str)

        if profile:
            profile_filename = os.path.join(base_dir, f'{self.name}{kind_suffix}.html')
            log_print(f'Persisting profile to \'{profile_filename}\'...')
            self.profile.to_file(profile_filename)

    @function_call_logger
    def pipeline(
        self,
        preload: bool = False,
        round_decimals: int = 3,
        shrink_num_mode: str | None = 'conservative',
        handle_num_mode: str | None = 'discretize',
        handle_obj_mode: str | None = 'auto',
        complexity_mode: str | None = None,
        profile_mode: str | None = 'minimal'
    ) -> Self:
        if preload:
            self.preload()
        else:
            self.prepare()                                    # Setup paths, configs, logging
            self.load()                                       # Load raw data
            self.sanitize()                                   # Clean up column names, fix encoding
            self.infer_dtypes()                               # Guess column types (object -> number, etc.)
            self.convert_to_numeric()                         # Convert object columns that look like numbers
            self.drop_infinite_rows()                         # Remove rows with inf/-inf
            self.round_floats(round_decimals)                 # Round float precision (e.g., to 3 decimals)
            self.handle_port_columns()                        # Semantically bin port columns
            self.handle_hc_numeric_columns(handle_num_mode)   # Discretize high-cardinality numeric cols
            self.handle_object_columns(handle_obj_mode)       # Encode or transform object columns
            self.shrink_numeric_dtypes(shrink_num_mode)       # Downcast float64/int64 to float32/int32
            self.drop_high_unique_columns()                   # Drop columns where mostly all values are unique
            self.clean_and_sort_columns()                     # Clean names and sort columns
            self.reset_index()                                # Reset index after row ops
            self.drop_na_duplicates()                         # Drop rows with all-NaNs and/or duplicates
            self.compute_complexity(complexity_mode)          # Compute metrics like ANOVA, KDN, etc.
            self.compute_profile(profile_mode)                # Summarize stats, distributions
            self.update_metadata()                            # Save dataset summary info
            self.save()                                       # Persist cleaned data + metadata
        return self
