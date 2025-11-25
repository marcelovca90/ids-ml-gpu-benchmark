import json
import math
import os
import re
import shutil
from pathlib import Path
from pprint import pformat
from tqdm import tqdm

try:
    from modules.logging.logger import log_event
except ImportError:
    def log_event(**kwargs):
        print(kwargs)

# Recursively replaces NaN values with None (JSON null)
def _replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: _replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_nan_with_none(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def copy_files():

    root_src_folder = "datasets"
    root_dst_folder = "2025-11-17/Input_Zip_v3a"
    candidate_files = list(Path(root_src_folder).rglob("*"))
    copied_files = {}

    for j, src_path in enumerate(tqdm(candidate_files, desc="File", leave=False)):
        msg_prefix = f"[{j+1:02}/{len(candidate_files):02}]"
        if (src_path.is_file() and
            ("generated" in str(src_path.absolute().resolve())) and
            (src_path.name.lower().endswith(".metadata.json"))):
            with open(src_path, mode='r', encoding='utf-8') as fp:
                metadata = json.load(fp)
            src_folder = src_path.parent
            dst_folder_suffix = re.sub(r'.*generated/', '', str(src_folder))
            dst_folder = os.path.join(root_dst_folder, metadata['name'], dst_folder_suffix)
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            os.makedirs(dst_folder, exist_ok=True)
            for src_file_name in os.listdir(src_folder):
                if metadata['name'] in src_file_name:
                    src_file_path = os.path.join(src_folder, src_file_name)
                    dst_file_path = os.path.join(dst_folder, src_file_name)
                    tqdm.write(f"{msg_prefix} Copying {src_file_path} to {dst_file_path}...")
                    shutil.copy2(src_file_path, dst_file_path)
                    copied_files[src_file_path] = dst_file_path

    if copied_files:
        log_event(stage="finish", prefix=f"The following files were copied:\n```json\n{pformat(copied_files, indent=2)}\n```")
        dispatcher_filename = Path(os.path.join(root_dst_folder, "start"))
        dispatcher_filename.touch(exist_ok=True)
        log_event(stage="finish", prefix=f"Dispatcher file created at:\n```json\n{dispatcher_filename}\n```")

def rename_files():

    root_src_folder = "2025-11-17/Input_Zip_v3a"
    root_dst_folder = "2025-11-17/Input_Zip_v3b"
    metadata_files = list(Path(root_src_folder).rglob("*.json"))
    renamed_files = {}

    # Locate *.metadata.json files
    for i, src_metadata_file in enumerate(tqdm(metadata_files, desc="Metadata", leave=False)):

        src_folder = str(src_metadata_file.parent)

        # Read all *.metadata.json
        if src_metadata_file.is_file() and src_metadata_file.name.lower().endswith(".metadata.json"):
            with open(src_metadata_file, mode='r', encoding='utf-8') as fp:
                metadata = json.load(fp)

            # Remove Binary/Multiclass suffixes
            pattern = r'(_Binary)+' if metadata['binarize'] else r'_Multiclass+'
            metadata['name'] = re.sub(pattern, '', metadata['name'])
            metadata['folder'] = os.path.join(root_dst_folder, metadata['name'])
            metadata = _replace_nan_with_none(metadata)

            # Clean and prepare destination folder
            dst_folder_suffix = 'Binary' if metadata['binarize'] else 'Multiclass'
            dst_folder_suffix = os.path.join(metadata['name'], dst_folder_suffix)
            dst_base_folder = src_folder.replace(root_src_folder, root_dst_folder)
            dst_base_folder = re.sub(pattern, '', dst_base_folder)
            dst_sub_folder = dst_base_folder.replace(metadata['name'], dst_folder_suffix)
            os.makedirs(dst_sub_folder, exist_ok=True)

            # Persist updated metadata file
            dst_metadata_file = re.sub(pattern, '', src_metadata_file.name)
            dst_metadata_file = Path(os.path.join(dst_sub_folder, dst_metadata_file))
            tqdm.write(f"Copying {src_metadata_file} to {dst_metadata_file}...")
            with open(dst_metadata_file, mode='w', encoding='utf-8') as fp:
                json.dump(metadata, fp, indent=4)
            renamed_files[src_metadata_file] = dst_metadata_file

            parquet_files = list(Path(src_folder).rglob("*.parquet"))
            for j, src_parquet_file in enumerate(tqdm(parquet_files, desc="Parquet", leave=False)):
                dst_parquet_file = Path(re.sub(pattern, '', str(src_parquet_file)))
                dst_parquet_file = Path(os.path.join(dst_sub_folder, dst_parquet_file.name))
                tqdm.write(f"Copying {src_parquet_file} to {dst_parquet_file}...")
                shutil.copy2(src_parquet_file, dst_parquet_file)
                renamed_files[src_parquet_file] = dst_parquet_file

    if renamed_files:
        log_event(stage="finish", prefix=f"The following files were renamed:\n```json\n{pformat(renamed_files, indent=2)}\n```")
        dispatcher_filename = Path(os.path.join(root_dst_folder, "start"))
        dispatcher_filename.touch(exist_ok=True)
        log_event(stage="finish", prefix=f"Dispatcher file created at:\n```json\n{dispatcher_filename}\n```")

# PYTHONPATH=. python file_utils.py
if __name__ == "__main__":
    rename_files()