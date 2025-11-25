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
from modules.preprocessing.splitting import (
    sample_train_subset, train_val_test_split_fn
)
from modules.preprocessing.preproc_utils import NumpyEncoder
from modules.evaluation.resource_monitor import ResourceMonitor

class BasePreprocessingPipeline(ABC):

    def __init__(self, sample_frac=1.0, seed=42, binarize=False) -> None:
        self.data: pd.DataFrame = None
        self.folder: str = None
        self.name: str = None
        self.target: str = None
        self.sample_frac: float = sample_frac
        self.seed : int = seed
        self.binarize : bool = binarize
        self.kind = 'Binary' if self.binarize else 'Multiclass'
        self.metadata: dict = {}

    @function_call_logger
    def check_attrs(self, caller_fn: str, include_sampled: bool = True):
        """
        Ensure required dataset attributes exist before data-driven steps.

        Required always:
            - df_train_full
            - df_val_full
            - df_test_full

        Required optionally:
            - df_train_sampled
            - df_val_sampled
            - df_test_sampled

        (The sampled versions are needed only after sample_train_subset().)
        """

        required_full = ["df_train_full", "df_val_full", "df_test_full"]
        required_sampled = ["df_train_sampled", "df_val_sampled", "df_test_sampled"]

        subsets = required_full + (required_sampled if include_sampled else [])

        missing = [
            attr for attr in subsets
            if not hasattr(self, attr) or getattr(self, attr) is None
        ]

        if missing:
            raise AttributeError(
                f"{caller_fn} requires the following dataset attributes: {missing}. "
                "Run train_val_test_split() first, and sample_train_subset() if needed."
            )

    # -------------------------------------------------------------------------
    # 1. NEW HELPER: Centralize Universe Definitions
    # -------------------------------------------------------------------------
    def _get_universe_definition(self, universe_name: str) -> dict:
        """
        Returns the attribute names associated with a specific universe.
        """
        if universe_name.upper() == "FULL":
            return {
                "train": "df_train_full",
                "val": "df_val_full",
                "test": "df_test_full",
                "bins_numeric": "_numeric_discretization_bins_full",
                "encoders_obj": "_object_encoders_full",
                "metadata_key": "full",
                "profile_attr": "profile_full"
            }
        elif universe_name.upper() == "SAMPLED":
            return {
                "train": "df_train_sampled",
                "val": "df_val_sampled",
                "test": "df_test_sampled",
                "bins_numeric": "_numeric_discretization_bins_sampled",
                "encoders_obj": "_object_encoders_sampled",
                "metadata_key": "sampled",
                "profile_attr": "profile_sampled"
            }
        else:
            raise ValueError(f"Unknown universe: {universe_name}")

    # -------------------------------------------------------------------------
    # 2. COMMON FUNCTIONS
    # -------------------------------------------------------------------------

    @function_call_logger
    def preload(self) -> None:
        """
        Preload ONLY the FULL train/val/test splits from disk, if they exist.
        """
        base_root = (
            self.base_dir
            if hasattr(self, "base_dir") and self.base_dir is not None
            else os.path.join(os.getcwd(), self.folder, "generated")
        )

        full_dir = os.path.join(base_root, f"seed_{self.seed}", "full")

        if not os.path.isdir(full_dir):
            raise FileNotFoundError(
                f"FULL universe not found for seed={self.seed} at: {full_dir}. "
                f"Run the pipeline once with preload=False to create it."
            )

        fname_base = f"{self.name}_{self.kind}"

        train_path = os.path.join(full_dir, f"{fname_base}_train.parquet")
        val_path   = os.path.join(full_dir, f"{fname_base}_val.parquet")
        test_path  = os.path.join(full_dir, f"{fname_base}_test.parquet")

        for path in (train_path, val_path, test_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected FULL split file missing: {path}")

        log_print(f"Preloading FULL universe from {full_dir}")
        self.df_train_full = pd.read_parquet(train_path)
        self.df_val_full   = pd.read_parquet(val_path)
        self.df_test_full  = pd.read_parquet(test_path)

        # Optionally load full metadata
        meta_full_path = os.path.join(full_dir, f"{self.name}.metadata.json")
        if os.path.exists(meta_full_path):
            with open(meta_full_path) as fp:
                meta_full = json.load(fp)
        else:
            meta_full = {}

        self.metadata = {
            "kind": self.kind,
            "full": meta_full,
            "sampled": None,
        }
        log_print("Preload completed: FULL universe ready.")

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

            log_print(f"{col:<40} {str(current_dtype):<8} => {inferred_label or 'unknown':<11} ({target_dtype or 'skip'})")

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
    def drop_infinite_rows(self) -> None:
        mask = self.data.isin([np.inf, -np.inf]).any(axis=1)
        self.data = self.data[~mask]
        log_print(f"Dropped {mask.sum()} rows containing ±inf")

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
    def rule_based_drop_ip_columns(self) -> None:
        ipv4_pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
        ipv6_pattern = re.compile(r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$')

        def is_ip(val: str) -> bool:
            val = val.strip()
            return bool(ipv4_pattern.fullmatch(val)) or bool(ipv6_pattern.fullmatch(val))

        def column_is_mostly_ip(col: pd.Series) -> bool:
            non_null = col.dropna()
            if non_null.empty:
                return False
            sample = non_null.sample(n=int(0.01 * len(non_null)), random_state=self.seed).astype(str)
            return sample.apply(is_ip).mean() > 0.95

        ip_cols = [col for col in self.data.columns if col != self.target and column_is_mostly_ip(self.data[col])]

        if ip_cols:
            log_print(f"Dropping columns with IP-like values: {ip_cols}")
            self.data = self.data.drop(columns=ip_cols)
        else:
            log_print("No IP-like columns detected.")

    @function_call_logger
    def rule_based_drop_mac_columns(self) -> None:
        # Matches MACs with separators (:, -, _)
        mac_sep_pattern = re.compile(r'^([0-9A-F]{2}[:\-_]){5}([0-9A-F]{2})$', re.IGNORECASE)
        # Matches MACs without separators (12 hex pairs)
        mac_nosep_pattern = re.compile(r'^[0-9A-F]{12}$', re.IGNORECASE)

        def is_mac(val: str) -> bool:
            val = val.strip()
            return bool(mac_sep_pattern.fullmatch(val)) or bool(mac_nosep_pattern.fullmatch(val))

        def column_is_mostly_mac(col: pd.Series) -> bool:
            if not pd.api.types.is_object_dtype(col):
                return False
            non_null = col.dropna()
            if non_null.empty:
                return False
            sample_size = max(10, min(int(0.01 * len(non_null)), 100))
            sample = non_null.sample(n=sample_size, random_state=self.seed).astype(str)
            return sample.apply(is_mac).mean() > 0.95

        mac_cols = [col for col in self.data.columns if col != self.target and column_is_mostly_mac(self.data[col])]

        if mac_cols:
            log_print(f"Dropping columns with MAC-like values: {', '.join(mac_cols)}")
            self.data = self.data.drop(columns=mac_cols)
        else:
            log_print("No MAC-like columns detected.")

    @function_call_logger
    def rule_based_discretize_port_columns(self) -> None:

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
    def rule_based_handle_object_columns(self, handle_obj_mode: str = "auto") -> None:
        """
        Rule-based handling for object columns.

        Safe to run BEFORE train/val/test split, because it does NOT learn from
        data distributions (no value_counts, no target-based stats).

        - mode 'keep': do nothing
        - mode 'drop': drop all non-target object columns
        - mode 'encode_cat':
            -> cast all object columns to 'category'
        - mode 'auto':
            -> low-cardinality: cast to 'category'
            -> very high-cardinality: apply hashing trick
            -> medium-cardinality: leave as object for train-only encoding later
        """

        if handle_obj_mode == "keep":
            log_print("handle_object_columns_rule_based: mode='keep', nothing to do.")
            return

        HIGH_CARD = 2**14       # 16,384
        VERY_HIGH_CARD = 2**18  # 262,144
        MAX_BUCKETS = 2**16     # 65,536

        log_print("Rule-based object handling: data types and memory BEFORE:")
        log_data_types(self.data)
        log_memory_usage(self.data)

        object_cols = (
            self.data
            .drop(columns=[self.target])
            .select_dtypes(include="object")
            .columns
        )

        if len(object_cols) == 0:
            log_print("No object columns to process (rule-based).")
            return

        for col in object_cols:
            n_unique = self.data[col].nunique(dropna=False)
            log_print(f"[RULE-BASED] Column '{col}' has {n_unique} unique values")

            if handle_obj_mode == "drop":
                self.data = self.data.drop(columns=[col])
                log_print(f"[RULE-BASED] Dropped object column '{col}'")
                continue

            # Force everything to category if requested
            if handle_obj_mode == "encode_cat":
                self.data[col] = self.data[col].astype("category")
                log_print(f"[RULE-BASED] Encoded '{col}' as category (encode_cat mode)")
                continue

            # AUTO mode logic (rule-based branches only)
            if handle_obj_mode == "auto":
                if n_unique < HIGH_CARD:
                    # Low-cardinality: categorical dtype is safe and rule-based
                    self.data[col] = self.data[col].astype("category")
                    log_print(f"[RULE-BASED] Encoded '{col}' as category (low-card)")
                elif n_unique > VERY_HIGH_CARD:
                    # Very high-cardinality: hashing trick (rule-based)
                    col_str = self.data[col].astype(str)
                    n_buckets_raw = 2 ** math.ceil(math.log2(n_unique * 1.1))
                    n_buckets = min(n_buckets_raw, MAX_BUCKETS)
                    hashed = pd.util.hash_pandas_object(col_str, index=False).astype("int64") + self.seed
                    self.data[col] = (hashed % n_buckets).astype("int32")
                    log_print(f"[RULE-BASED] Hashed '{col}' to {n_buckets} buckets (very high-card)")
                else:
                    # Medium-cardinality: leave as object for train-only encoding
                    log_print(f"[RULE-BASED] Leaving '{col}' as object (medium-card, will encode train-only)")

        log_print("Rule-based object handling: data types and memory AFTER:")
        log_data_types(self.data)
        log_memory_usage(self.data)

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
    def train_val_test_split(self) -> None:
        """
        Split self.data into FULL train/val/test sets.

        MEMORY OPTIMIZATION:
        Once the split is done, self.data is redundant. We delete it 
        and force garbage collection to free up RAM immediately.
        """

        # 1. Perform the split
        self.df_train_full, self.df_val_full, self.df_test_full = train_val_test_split_fn(
            self.data,
            target_column=self.target,
            min_samples_per_class=1,
            random_state=self.seed,
        )

        # 2. Free Memory
        import gc
        # Optional: Log how much we are freeing (if pandas is available)
        if hasattr(self.data, 'memory_usage'):
             mem_usage = self.data.memory_usage(deep=True).sum() / 1024**2
             log_print(f"Memory Optimization: Freeing self.data (~{mem_usage:.2f} MB)")

        self.data = None
        gc.collect()

        # 3. Verify
        self.check_attrs("train_val_test_split", include_sampled=False)

    @function_call_logger
    def drop_high_unique_columns(self) -> None:
        """
        Drop columns that are almost unique based on FULL TRAIN data only.

        - Compute high-unique columns on df_train_full (DATA-DRIVEN; FIT ON TRAIN).
        - Drop the same columns from df_train_full, df_val_full, df_test_full.
        """

        UNIQUE_THRESHOLD = 0.999

        # Require only full splits at this stage
        self.check_attrs("drop_high_unique_columns", include_sampled=False)

        n_rows_train = len(self.df_train_full)
        if n_rows_train == 0:
            log_print("drop_high_unique_columns: df_train_full is empty; nothing to do.")
            return

        candidate_cols = [col for col in self.df_train_full.columns if col != self.target]

        high_unique_cols = [
            col
            for col in candidate_cols
            if (
                self.df_train_full[col].nunique(dropna=False) / n_rows_train >= UNIQUE_THRESHOLD
                and self.df_train_full[col].nunique(dropna=False) < n_rows_train
            )
        ]

        if not high_unique_cols:
            log_print(
                f"No high-unique columns found (>={UNIQUE_THRESHOLD:.0%} unique) in df_train_full."
            )
            return

        log_print(
            f"Dropped high-unique columns (>={UNIQUE_THRESHOLD:.0%} unique in train_full): "
            f"{high_unique_cols}"
        )

        # Drop from all FULL splits
        for df_attr in ("df_train_full", "df_val_full", "df_test_full"):
            df = getattr(self, df_attr)
            cols_to_drop_here = [c for c in high_unique_cols if c in df.columns]
            if cols_to_drop_here:
                df.drop(columns=cols_to_drop_here, inplace=True)
            setattr(self, df_attr, df)

    ############

    @function_call_logger
    def sample_train_subset(self) -> None:
        """
        Create the sampled-universe train/val/test sets.

        - df_train_full / df_val_full / df_test_full already exist.
        - This function creates:
            df_train_sampled
            df_val_sampled
            df_test_sampled
        - Only df_train_sampled is downsampled.
        - df_val_sampled and df_test_sampled remain IDENTICAL to full versions.
        """

        # 1) Safety check: full sets must exist
        self.check_attrs("sample_train_subset", include_sampled=False)

        # If frac=1.0 → sampled universe identical to full → avoid duplicate work downstream
        if self.sample_frac == 1.0:
            self.df_train_sampled = self.df_train_full
            self.df_val_sampled   = self.df_val_full
            self.df_test_sampled  = self.df_test_full
            log_print("sample_frac=1.0 → sampled universe identical to FULL. No sampling applied.")
            return

        # Otherwise create true sampled universe
        self.df_train_sampled = self.df_train_full.copy()
        self.df_val_sampled   = self.df_val_full.copy()
        self.df_test_sampled  = self.df_test_full.copy()

        # 3) Apply sampling to the TRAIN subset only
        if self.sample_frac is None:
            log_print("No sampling parameter given => df_train_sampled is identical to df_train_full.")
            return

        # sample_train_subset must return a sampled DataFrame
        self.df_train_sampled = sample_train_subset(
            df_train_full=self.df_train_full,
            frac=self.sample_frac,
            random_state=self.seed,
        )

        log_print(
            f"df_train_sampled now has {len(self.df_train_sampled)} rows "
            f"(full={len(self.df_train_full)}, frac={self.sample_frac})"
        )

    @function_call_logger
    def data_driven_discretize_hc_numeric_columns(self, handle_num_mode: str = "discretize", target_universes: list = None) -> None:
        if handle_num_mode != "discretize":
            return

        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        HIGH_CARD = 2**16       # 65,536
        VERY_HIGH_CARD = 2**20  # 1,048,576

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']):
                continue

            train_df = getattr(self, u_def['train'])

            # Identify numeric columns (Exclude Target and _ROW_ID)
            numeric_cols = train_df.select_dtypes(include=["number"]).columns
            numeric_cols = [
                col for col in numeric_cols
                if col != self.target 
                and col != "_ROW_ID" 
                and train_df[col].nunique(dropna=False) > HIGH_CARD
            ]

            if not numeric_cols:
                log_print(f"[{u_name}] No high-cardinality numeric columns to discretize.")
                setattr(self, u_def['bins_numeric'], {})
                continue

            log_print(f"[{u_name}] BEFORE numeric discretization (train):")
            log_col_data(train_df, numeric_cols)

            bins_dict = {}

            # --- FIT & TRANSFORM TRAIN ---
            # We transform TRAIN here directly to get the bins and labels
            for col in numeric_cols:
                nunique = train_df[col].nunique(dropna=False)

                if nunique > VERY_HIGH_CARD:
                    # Quantile-based
                    qcut_result = pd.qcut(train_df[col], q=100, duplicates="drop")
                    intervals = qcut_result.cat.categories
                    labels = [f"Q{i+1}" for i in range(len(intervals))]

                    bins_dict[col] = {"type": "qcut", "bins": intervals, "labels": labels}
                    # Transform Train immediately
                    train_df[col] = qcut_result.cat.rename_categories(labels)

                elif nunique > HIGH_CARD:
                    # Equal-width
                    cut_result = pd.cut(train_df[col], bins=100, duplicates="drop")
                    intervals = cut_result.cat.categories
                    labels = [f"Q{i+1}" for i in range(len(intervals))]

                    bins_dict[col] = {"type": "cut", "bins": intervals, "labels": labels}
                    # Transform Train immediately
                    train_df[col] = cut_result.cat.rename_categories(labels)

            # Update Train in self (It is now Categorical)
            setattr(self, u_def['train'], train_df)
            setattr(self, u_def['bins_numeric'], bins_dict)

            # --- TRANSFORM VAL & TEST ---
            # FIX: Do NOT include 'train' here, because it was already transformed above!
            # Re-running pd.cut on the now-categorical train data causes the 'float vs str' crash.
            for col, cfg in bins_dict.items():
                intervals = cfg["bins"]
                labels = cfg["labels"]
                bin_edges = [iv.left for iv in intervals] + [intervals[-1].right]

                for split_key in ['val', 'test']:
                    df_attr = u_def[split_key]
                    if not hasattr(self, df_attr): continue

                    df = getattr(self, df_attr)
                    if col not in df.columns: continue

                    # Apply cuts using the edges derived from Train
                    df[col] = pd.cut(df[col], bins=bin_edges, labels=labels)
                    df[col] = df[col].astype("category")
                    setattr(self, df_attr, df)

            log_print(f"[{u_name}] AFTER numeric discretization (train):")
            log_col_data(getattr(self, u_def['train']), numeric_cols)

    @function_call_logger
    def data_driven_handle_object_columns(self, handle_obj_mode: str = "auto", target_universes: list = None) -> None:
        if handle_obj_mode == "keep":
            return

        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        HIGH_CARD = 2**14       # 16,384
        VERY_HIGH_CARD = 2**18  # 262,144

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            train_df = getattr(self, u_def['train'])

            object_cols = (
                train_df.drop(columns=[self.target])
                .select_dtypes(include="object")
                .columns
            )

            if len(object_cols) == 0:
                log_print(f"[{u_name}] No object columns to process (data-driven).")
                setattr(self, u_def['encoders_obj'], {})
                continue

            log_print(f"[{u_name}] Data-driven object handling: dtypes/memory BEFORE:")
            log_data_types(train_df)
            log_memory_usage(train_df)

            encoders_info = {}
            split_keys = ['train', 'val', 'test']

            for col in object_cols:
                n_unique = train_df[col].nunique(dropna=False)
                log_print(f"[{u_name}][DATA-DRIVEN] Column '{col}' has {n_unique} unique values")

                # --- MODE: DROP ---
                if handle_obj_mode == "drop":
                    for sk in split_keys:
                        df_attr = u_def[sk]
                        if hasattr(self, df_attr):
                            df = getattr(self, df_attr)
                            if col in df.columns:
                                df.drop(columns=[col], inplace=True)
                                setattr(self, df_attr, df)
                    continue

                # --- MODE: ENCODE_CAT ---
                if handle_obj_mode == "encode_cat":
                    for sk in split_keys:
                        df_attr = u_def[sk]
                        if hasattr(self, df_attr):
                            df = getattr(self, df_attr)
                            if col in df.columns:
                                df[col] = df[col].astype("category")
                                setattr(self, df_attr, df)
                    continue

                # --- MODE: AUTO ---
                if handle_obj_mode == "auto":
                    if not (HIGH_CARD <= n_unique <= VERY_HIGH_CARD):
                        log_print(f"[{u_name}] Skipping '{col}' (not medium-card)")
                        continue

                    value_counts = train_df[col].value_counts(dropna=False)
                    skewness = skew(value_counts.values.astype(float))

                    if abs(skewness) > 1:
                        # Frequency Encoding
                        freq_map = (value_counts / len(train_df)).astype("float32")
                        encoders_info[col] = {"type": "frequency", "skewness": float(skewness)}

                        for sk in split_keys:
                            df_attr = u_def[sk]
                            if hasattr(self, df_attr):
                                df = getattr(self, df_attr)
                                if col in df.columns:
                                    df[col] = df[col].map(freq_map).fillna(0).astype("float32")
                                    setattr(self, df_attr, df)
                    else:
                        # Count Encoding
                        count_map = value_counts.astype("int32")
                        encoders_info[col] = {"type": "count", "skewness": float(skewness)}

                        for sk in split_keys:
                            df_attr = u_def[sk]
                            if hasattr(self, df_attr):
                                df = getattr(self, df_attr)
                                if col in df.columns:
                                    df[col] = df[col].map(count_map).fillna(0).astype("int32")
                                    setattr(self, df_attr, df)

            setattr(self, u_def['encoders_obj'], encoders_info)

            log_print(f"[{u_name}] Data-driven object handling: dtypes/memory AFTER (train):")
            log_data_types(getattr(self, u_def['train']))
            log_memory_usage(getattr(self, u_def['train']))

    @function_call_logger
    def data_driven_handle_object_columns(self, handle_obj_mode: str = "auto", target_universes: list = None) -> None:
        if handle_obj_mode == "keep":
            return

        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        HIGH_CARD = 2**14       # 16,384
        VERY_HIGH_CARD = 2**18  # 262,144

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            train_df = getattr(self, u_def['train'])

            object_cols = (
                train_df.drop(columns=[self.target])
                .select_dtypes(include="object")
                .columns
            )

            if len(object_cols) == 0:
                log_print(f"[{u_name}] No object columns to process (data-driven).")
                setattr(self, u_def['encoders_obj'], {})
                continue

            log_print(f"[{u_name}] Data-driven object handling: dtypes/memory BEFORE:")
            log_data_types(train_df)
            log_memory_usage(train_df)

            encoders_info = {}
            split_keys = ['train', 'val', 'test']

            for col in object_cols:
                n_unique = train_df[col].nunique(dropna=False)
                log_print(f"[{u_name}][DATA-DRIVEN] Column '{col}' has {n_unique} unique values")

                # --- MODE: DROP ---
                if handle_obj_mode == "drop":
                    for sk in split_keys:
                        df_attr = u_def[sk]
                        if hasattr(self, df_attr):
                            df = getattr(self, df_attr)
                            if col in df.columns:
                                df.drop(columns=[col], inplace=True)
                                setattr(self, df_attr, df)
                    continue

                # --- MODE: ENCODE_CAT ---
                if handle_obj_mode == "encode_cat":
                    for sk in split_keys:
                        df_attr = u_def[sk]
                        if hasattr(self, df_attr):
                            df = getattr(self, df_attr)
                            if col in df.columns:
                                df[col] = df[col].astype("category")
                                setattr(self, df_attr, df)
                    continue

                # --- MODE: AUTO ---
                if handle_obj_mode == "auto":
                    if not (HIGH_CARD <= n_unique <= VERY_HIGH_CARD):
                        log_print(f"[{u_name}] Skipping '{col}' (not medium-card)")
                        continue

                    value_counts = train_df[col].value_counts(dropna=False)
                    skewness = skew(value_counts.values.astype(float))

                    if abs(skewness) > 1:
                        # Frequency Encoding
                        freq_map = (value_counts / len(train_df)).astype("float32")
                        encoders_info[col] = {"type": "frequency", "skewness": float(skewness)}

                        for sk in split_keys:
                            df_attr = u_def[sk]
                            if hasattr(self, df_attr):
                                df = getattr(self, df_attr)
                                if col in df.columns:
                                    df[col] = df[col].map(freq_map).fillna(0).astype("float32")
                                    setattr(self, df_attr, df)
                    else:
                        # Count Encoding
                        count_map = value_counts.astype("int32")
                        encoders_info[col] = {"type": "count", "skewness": float(skewness)}

                        for sk in split_keys:
                            df_attr = u_def[sk]
                            if hasattr(self, df_attr):
                                df = getattr(self, df_attr)
                                if col in df.columns:
                                    df[col] = df[col].map(count_map).fillna(0).astype("int32")
                                    setattr(self, df_attr, df)

            setattr(self, u_def['encoders_obj'], encoders_info)

            log_print(f"[{u_name}] Data-driven object handling: dtypes/memory AFTER (train):")
            log_data_types(getattr(self, u_def['train']))
            log_memory_usage(getattr(self, u_def['train']))

    @function_call_logger
    def shrink_numeric_dtypes(self, shrink_mode: str | None = "conservative", target_universes: list = None) -> None:
        if not shrink_mode:
            return
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            if shrink_mode == "conservative":
                # Use THIS universe's train as reference
                train_df = getattr(self, u_def['train'])
                ref_numeric_cols = train_df.drop(columns=[self.target]).select_dtypes(include=["number"]).columns

                for split_key in ['train', 'val', 'test']:
                    df_attr = u_def[split_key]
                    if hasattr(self, df_attr):
                        df = getattr(self, df_attr)
                        for col in ref_numeric_cols:
                            if col in df.columns:
                                if pd.api.types.is_integer_dtype(df[col]):
                                    df[col] = df[col].astype("int32")
                                elif pd.api.types.is_float_dtype(df[col]):
                                    df[col] = df[col].astype("float32")
                        setattr(self, df_attr, df)

            elif shrink_mode == "aggressive":
                # Build combined view for THIS universe
                dfs = []
                for split_key in ['train', 'val', 'test']:
                    if hasattr(self, u_def[split_key]):
                        dfs.append(getattr(self, u_def[split_key]))

                if not dfs: continue

                df_all = pd.concat(dfs, axis=0, ignore_index=True)
                df_report = report_on_dataframe(df_all, unit="MB", optimize="computation")
                df_all_opt = optimize_dtypes(df_all, df_report)
                dtype_map = df_all_opt.dtypes.to_dict()

                for split_key in ['train', 'val', 'test']:
                    df_attr = u_def[split_key]
                    if hasattr(self, df_attr):
                        df = getattr(self, df_attr)
                        for col, dtype in dtype_map.items():
                            if col != self.target and col in df.columns:
                                df[col] = df[col].astype(dtype)
                        setattr(self, df_attr, df)

    @function_call_logger
    def clean_and_sort_columns(self, target_universes: list = None) -> None:
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            # Canonical source is the TRAIN set of THIS universe
            ref_df = getattr(self, u_def['train'])
            base_cols = list(ref_df.columns)

            if self.target not in base_cols:
                # Fallback or Error if target missing in train (unlikely)
                continue

            # Build Rename Map
            rename_map = {
                col: re.sub(r"[^A-Za-z0-9]", "_", col).upper()
                for col in base_cols if col != self.target
            }

            # Build Final Order
            renamed_feature_cols = [rename_map.get(col, col) for col in base_cols if col != self.target]
            final_cols = sorted(renamed_feature_cols) + [self.target]

            # Apply to all splits
            for split_key in ['train', 'val', 'test']:
                df_attr = u_def[split_key]
                if hasattr(self, df_attr):
                    df = getattr(self, df_attr)
                    # Rename
                    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                    # Reorder
                    cols_present = [c for c in final_cols if c in df.columns]
                    df = df.reindex(columns=cols_present)
                    setattr(self, df_attr, df)

    @function_call_logger
    def assert_disjoint_subsets(self, id_col: str = "_ROW_ID") -> None:
        """
        Assert that train/val/test subsets are disjoint based on a row-id column.
        Checks ANY universe (Full or Sampled) that currently exists.
        Skips the check if the ID column is missing (e.g., after loading cleaned data).
        """

        def _assert_disjoint(df_train, df_val, df_test, universe_name):
            # 1. Safety Check: Does the ID column exist?
            # If we loaded pre-cleaned data, _ROW_ID might be gone. That's fine.
            if id_col not in df_train.columns:
                log_print(f"[{universe_name}] '{id_col}' not found. Skipping disjoint check (assumed safe from previous run).")
                return

            # 2. Perform the check
            train_ids = set(df_train[id_col])
            val_ids   = set(df_val[id_col])
            test_ids  = set(df_test[id_col])

            inter_train_val  = train_ids & val_ids
            inter_train_test = train_ids & test_ids
            inter_val_test   = val_ids & test_ids

            assert not inter_train_val,  (
                f"[{universe_name}] Train and Val share {len(inter_train_val)} IDs. Leakage detected!"
            )
            assert not inter_train_test, (
                f"[{universe_name}] Train and Test share {len(inter_train_test)} IDs. Leakage detected!"
            )
            assert not inter_val_test,   (
                f"[{universe_name}] Val and Test share {len(inter_val_test)} IDs. Leakage detected!"
            )
            log_print(f"[{universe_name}] Disjoint check passed (No ID overlap).")

        # Check FULL Universe
        if hasattr(self, "df_train_full") and hasattr(self, "df_val_full") and hasattr(self, "df_test_full"):
            _assert_disjoint(
                self.df_train_full, 
                self.df_val_full, 
                self.df_test_full, 
                "FULL"
            )

        # Check SAMPLED Universe
        if hasattr(self, "df_train_sampled") and hasattr(self, "df_val_sampled") and hasattr(self, "df_test_sampled"):
            _assert_disjoint(
                self.df_train_sampled, 
                self.df_val_sampled, 
                self.df_test_sampled, 
                "SAMPLED"
            )

    @function_call_logger
    def drop_row_id(self, target_universes: list = None) -> None:
        """
        Removes the _ROW_ID helper column from all subsets in the target universes.
        Should be called immediately after assert_disjoint_subsets().
        """
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)

            for split_key in ['train', 'val', 'test']:
                df_attr = u_def[split_key]
                if hasattr(self, df_attr):
                    df = getattr(self, df_attr)
                    if "_ROW_ID" in df.columns:
                        df = df.drop(columns=["_ROW_ID"])
                        setattr(self, df_attr, df)
                        log_print(f"[{u_name}] Dropped _ROW_ID from {split_key}.")

    @function_call_logger
    def reset_index(self, target_universes: list = None) -> None:
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            for split_key in ['train', 'val', 'test']:
                df_attr = u_def[split_key]
                if hasattr(self, df_attr):
                    df = getattr(self, df_attr)
                    if "_ROW_ID" in df.columns:
                        df = df.drop(columns=["_ROW_ID"])
                    df.reset_index(drop=True, inplace=True)
                    setattr(self, df_attr, df)

    @function_call_logger
    def compute_profile(self, profile_mode: str = "minimal", target_universes: list = None) -> None:
        if not profile_mode:
            return
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        minimal_flag = (profile_mode == "minimal")

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            log_print(f"Computing profile for {u_name} universe...")
            df = getattr(self, u_def['train'])
            profile = ProfileReport(df=df, minimal=minimal_flag)
            setattr(self, u_def['profile_attr'], profile)

    @function_call_logger
    def update_metadata(self, target_universes: list = None) -> None:
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        for u_name in target_universes:
            u_def = self._get_universe_definition(u_name)
            if not hasattr(self, u_def['train']): continue

            # Gather dict for this universe
            meta_dict = {
                "folder": self.folder,
                "name": self.name,
                "target": self.target,
                "sample_frac": self.sample_frac,
                "seed": self.seed,
                "binarize": self.binarize,
                "columns": getattr(self, u_def['train']).columns.tolist(),
                "dtypes": getattr(self, u_def['train']).dtypes.apply(str).to_dict(),
                "description": getattr(self, u_def['train']).describe(include="all").to_dict(),
                "shapes": {},
                "memory_usage_mb": {},
                "target_value_counts": {}
            }

            for split_key in ['train', 'val', 'test']:
                if hasattr(self, u_def[split_key]):
                    df = getattr(self, u_def[split_key])
                    suffix = f"{split_key}_{u_def['metadata_key']}" # e.g. train_full

                    meta_dict["shapes"][suffix] = df.shape

                    mem_bytes = df.memory_usage(deep=True).sum()
                    meta_dict["memory_usage_mb"][suffix] = round(mem_bytes / 1024**2, 2)

                    if self.target in df.columns:
                        meta_dict["target_value_counts"][suffix] = df[self.target].value_counts().to_dict()

            # Update self.metadata
            self.metadata[u_def['metadata_key']] = meta_dict

    @function_call_logger
    def _save_universe_artifacts(self, u_name: str, out_dir: str, fname_base: str) -> None:
        """
        Internal helper to save splits, metadata, and profile for a single universe.
        """
        u_def = self._get_universe_definition(u_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1. Save Splits
        for split_key in ['train', 'val', 'test']:
            if hasattr(self, u_def[split_key]):
                df = getattr(self, u_def[split_key])
                df.to_parquet(os.path.join(out_dir, f"{fname_base}_{split_key}.parquet"))

        # 2. Save Metadata
        meta_data = self.metadata.get(u_def['metadata_key'], {})
        with open(os.path.join(out_dir, f"{self.name}.metadata.json"), "w") as f:
            json.dump(meta_data, f, indent=2, cls=NumpyEncoder)

        # 3. Save Profile
        if hasattr(self, u_def['profile_attr']):
            profile = getattr(self, u_def['profile_attr'])
            if profile is not None:
                profile.to_file(os.path.join(out_dir, "profile.html"))

        log_print(f"Saved {u_name} artifacts to {out_dir}")

    @function_call_logger
    def save(self, target_universes: list = None) -> None:
        if target_universes is None:
            target_universes = ["FULL", "SAMPLED"]

        base_dir = os.path.join(os.getcwd(), self.folder, "generated", f"seed_{self.seed}")
        fname = f"{self.name}_{self.kind}"

        # Save FULL
        if "FULL" in target_universes and hasattr(self, "df_train_full"):
            full_dir = os.path.join(base_dir, "full")
            self._save_universe_artifacts("FULL", full_dir, fname)

        # Save SAMPLED
        if "SAMPLED" in target_universes and hasattr(self, "df_train_sampled"):
            # Only save sampled if it's actually a subset (frac != 1.0)
            if self.sample_frac != 1.0:
                sample_frac_str = f"{int(100 * self.sample_frac):02}"
                sampled_dir = os.path.join(base_dir, f"sampled_{sample_frac_str}")
                self._save_universe_artifacts("SAMPLED", sampled_dir, fname)

    @function_call_logger
    def pipeline(
        self,
        preload: bool = False,
        round_decimals: int = 3,
        shrink_num_mode: str | None = 'conservative',
        handle_num_mode: str | None = 'discretize',
        handle_obj_mode: str | None = 'auto',
        profile_mode: str | None = 'minimal'
    ) -> Self:

        # 1. Start Monitoring (Non-blocking)
        monitor = ResourceMonitor(interval=0.1)
        monitor.start()

        # 2. Run Execution Logic (Flat)
        target = []

        if not preload:
            log_print("--- PIPELINE MODE: FULL GENERATION ---")

            # a. Raw Loading & Rule-Based Cleaning
            self.prepare()
            self.load()
            self.sanitize()
            self.infer_dtypes()
            self.convert_to_numeric()
            self.drop_infinite_rows()
            self.round_floats(round_decimals)
            self.rule_based_drop_ip_columns()
            self.rule_based_drop_mac_columns()
            self.rule_based_discretize_port_columns()
            self.rule_based_handle_object_columns(handle_obj_mode)
            self.drop_na_duplicates()

            monitor.checkpoint("loading_and_cleaning")

            # b. Split into FULL universe
            self.train_val_test_split()
            self.drop_high_unique_columns() 

            # c. Determine Targets
            target = ["FULL"]
            # If user asked for a sample (e.g., 0.2) but started from raw data (preload=False),
            # we generate BOTH the Full universe and the Sampled universe in this single run.
            if self.sample_frac < 1.0:
                log_print(f"sample_frac={self.sample_frac} < 1.0 detected in Full Mode. Generating Sampled universe as well.")
                self.sample_train_subset()
                self.assert_disjoint_subsets()
                target.append("SAMPLED")

            # d. Drop ID immediately after checking disjointness
            self.drop_row_id(target_universes=target)

            monitor.checkpoint("splitting")

            # e. Process Targets (FULL, plus SAMPLED if applicable)
            self.data_driven_discretize_hc_numeric_columns(handle_num_mode, target_universes=target)
            self.data_driven_handle_object_columns(handle_obj_mode, target_universes=target)
            self.shrink_numeric_dtypes(shrink_num_mode, target_universes=target)
            self.clean_and_sort_columns(target_universes=target)
            self.reset_index(target_universes=target)

            monitor.checkpoint("transformation")

            # f. Profile & Save
            self.compute_profile(profile_mode, target_universes=target)
            self.update_metadata(target_universes=target)

            monitor.checkpoint("profiling")

        else:
            log_print(f"--- PIPELINE MODE: SAMPLED (frac={self.sample_frac}) ---")

            # a. Load the already-processed FULL universe
            self.preload()

            # b. Create the Sampled Universe (Derived from Full)
            self.sample_train_subset()
            self.assert_disjoint_subsets()

            # c. Process SAMPLED Universe ONLY
            target = ["SAMPLED"]

            # d. Drop ID immediately after checking disjointness
            self.drop_row_id(target_universes=target)

            monitor.checkpoint("loading_and_sampling")

            # e. Process Targets (SAMPLED only)
            self.data_driven_discretize_hc_numeric_columns(handle_num_mode, target_universes=target)
            self.data_driven_handle_object_columns(handle_obj_mode, target_universes=target)
            self.shrink_numeric_dtypes(shrink_num_mode, target_universes=target)
            self.clean_and_sort_columns(target_universes=target)
            self.reset_index(target_universes=target)

            monitor.checkpoint("transformation")

            # f. Profile & Save SAMPLED
            self.compute_profile(profile_mode, target_universes=target)
            self.update_metadata(target_universes=target)

            monitor.checkpoint("profiling")

        # 3. Stop Monitoring
        execution_stats = monitor.stop()

        # 4. Inject Profiling Stats
        for u_name in target:
            key = self._get_universe_definition(u_name)['metadata_key']
            if key in self.metadata:
                self.metadata[key]['execution_stats'] = execution_stats

        # 4. Save
        self.save(target_universes=target)

        return self