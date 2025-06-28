import chardet
import json
import numpy as np
import os
import pandas as pd

from datetime import datetime
from dtype_diet import optimize_dtypes, report_on_dataframe
from pathlib import Path
from tqdm import tqdm

ONE_HUNDRED_MB = 100 * 1024 * 1024 # 100 MB in bytes

ONE_GB = 1 * 1024 * 1024 * 1024 # 1 GB in bytes

def now():
    now = datetime.now()
    yyyymmdd_hhmmss_part = now.strftime('%Y-%m-%d %H:%M:%S')
    ms_part = f'{int(now.microsecond / 1000):03d}'
    return f'{yyyymmdd_hhmmss_part},{ms_part}'

def log_memory_usage(df: pd.DataFrame) -> None:
    total_memory = df.memory_usage(deep=True).sum()
    tqdm.write(f"[{now()}] {total_memory / (1024 ** 2):.2f} MB")

def round_all():

    candidate_files = list(Path("ready").rglob("*.parquet"))

    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):

        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):

            try:
                if src_path.is_file() and kind in src_path.name:
                    abs_path = str(src_path.absolute().resolve())
                    dst_path = Path(abs_path.replace('ready', 'rounded_3'))
                    if src_path.is_file() and src_path.exists():

                        # Load
                        tqdm.write(f"[{now()}] Loading {str(src_path)}...")
                        df = pd.read_parquet(src_path)

                        # Round
                        df = df.round(3)

                        # Drop NAs and duplicates
                        num_nas_before = df.isna().sum().sum()
                        num_duplicates_before = df.duplicated().sum()
                        tqdm.write(f"[{now()}] NAs before cleaning: {num_nas_before}")
                        tqdm.write(f"[{now()}] Duplicates before cleaning: {num_duplicates_before}")
                        df.dropna(axis='columns', how='all', inplace=True)
                        df.dropna(axis='index', how='any', inplace=True)
                        df.drop_duplicates(inplace=True)
                        num_nas_after = df.isna().sum().sum()
                        num_duplicates_after = df.duplicated().sum()
                        tqdm.write(f"[{now()}] NAs after cleaning: {num_nas_after}")
                        tqdm.write(f"[{now()}] Duplicates after cleaning: {num_duplicates_after}")
                        tqdm.write(f"[{now()}] Memory usage after cleaning:")
                        log_memory_usage(df)

                        # Shrink dtypes
                        tqdm.write(f"[{now()}] Data types and memory usage before shrinkage:")
                        log_memory_usage(df)
                        df_report = report_on_dataframe(df, unit="MB", optimize="computation")
                        df = optimize_dtypes(df, df_report)
                        for col in df.select_dtypes(include=['integer', 'float']).columns:
                            df[col] = pd.to_numeric(
                                df[col], downcast='integer' if df[col].dtype.kind == 'i' else 'float')
                        tqdm.write(f"[{now()}] Data types and memory usage after shrinkage:")
                        log_memory_usage(df)

                        # Save
                        tqdm.write(f"[{now()}] Saving {str(dst_path)}...")
                        os.makedirs(dst_path.parent, exist_ok=True)
                        df.to_parquet(dst_path)

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")

def inspect_size():

    candidate_files = list(Path("rounded_3").rglob("*.parquet"))

    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):

        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):

            try:
                if src_path.is_file() and kind in src_path.name:
                    if src_path.is_file() and src_path.exists() and src_path.stat().st_size > ONE_HUNDRED_MB:
                        # Load
                        tqdm.write(f"[{now()}] Loading {str(src_path)} ({src_path.stat().st_size / 1024 / 1024:.2f} MB)...")
                        df = pd.read_parquet(src_path)
                        pass

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")

def inspect_labels():

    candidate_files = list(Path("ready").rglob("*.json"))

    results_dict = []

    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):

        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):

            try:
                if src_path.is_file() and src_path.exists() and kind in src_path.name:
                    with open(src_path, 'rb') as file:
                        raw = file.read()
                        enc = chardet.detect(raw)['encoding']
                    with open(src_path, encoding=enc) as file:
                        data = json.load(file)
                    size = Path(str(src_path).replace('.json', '.parquet')).stat().st_size / 1024 / 1024
                    vc_len = len(data['value_counts'])
                    vc_sum = np.sum(list(data['value_counts'].values()))
                    vc_uniques = list(data['value_counts'].keys())
                    status = ('Binary' in str(src_path) and vc_len == 2) or ('Multiclass' in str(src_path) and vc_len > 2)
                    results_dict.append({
                        'kind': kind, 'path': str(src_path.stem), 'size': f'{size:.2f} MB',
                        'status': 'OK' if status else 'ERROR', 'samples': f'{vc_sum:,}',
                        'classes': vc_len, 'uniques': ", ".join(vc_uniques)
                    })

            except Exception as e:
                tqdm.write(f"[{now()}] Error in {src_path}: {e}")
    
    results_df = pd.DataFrame(results_dict)
    print(results_df)

if __name__ == "__main__":
    inspect_labels()