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

    copied_folders = {}

    for j, src_path in enumerate(tqdm(candidate_files, desc="File", leave=False)):
        msg_prefix = f"[{j+1:02}/{len(candidate_files):02}]"
        if (src_path.is_file() and
            ("generated" in str(src_path.absolute().resolve())) and
            (src_path.name.lower() == "metadata.json")):
            with open(src_path, mode='r', encoding='utf-8') as fp:
                metadata = json.load(fp)
            src_folder = src_path.parent
            dst_folder_suffix = re.sub(r'.*generated/', '', str(src_folder))
            dst_folder = os.path.join(root_dst_folder, metadata['name'], dst_folder_suffix)
            if os.path.exists(dst_folder):
                shutil.rmtree(dst_folder)
            tqdm.write(f"{msg_prefix} Copying {src_folder} to {dst_folder}...")
            shutil.copytree(src_folder, dst_folder)
            copied_folders[src_folder] = dst_folder

    if copied_folders:
        log_event(stage="finish", prefix=f"The following folders were copied:\n```json\n{pformat(copied_folders, indent=2)}\n```")
        dispatcher_filename = Path(os.path.join(root_dst_folder, "start"))
        dispatcher_filename.touch(exist_ok=True)
        log_event(stage="finish", prefix=f"Dispatcher file created at:\n```json\n{dispatcher_filename}\n```")

if __name__ == "__main__":
    copy_files()