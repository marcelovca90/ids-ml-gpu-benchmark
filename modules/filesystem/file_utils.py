import json
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

# PYTHONPATH=. python copy_files.py
def copy_files():

    root_src_folder = "datasets"

    root_dst_folder = "2025-11-17/Input"

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

if __name__ == "__main__":
    copy_files()