import os
import shutil
from pathlib import Path

from pprint import pformat
from tqdm import tqdm

from modules.logging.webhook import post_disc

# PYTHONPATH=. python move_files.py
if __name__ == "__main__":

    src_folder = "datasets"

    dst_folder = "2025-07-05"

    dataset_kinds = ["Multiclass"]

    candidate_files = list(Path(src_folder).rglob("*"))

    for i, kind in enumerate(tqdm(dataset_kinds, desc="Kind", leave=False)):
        moved_files = {}
        for j, src_path in enumerate(tqdm(candidate_files, desc="File", leave=False)):
            msg_prefix = f"[{i+1:02}/{len(dataset_kinds):02}] [{j+1:02}/{len(candidate_files):02}]"
            if (src_path.is_file() and str(kind) in src_path.name and
                ("generated" in str(src_path.absolute().resolve())) and
                (src_path.name.lower().endswith((".parquet", ".json", ".html")))):
                dst_path = Path(os.path.join(dst_folder, f"Input_{kind}", src_path.name))
                if dst_path.is_file() and dst_path.exists():
                    dst_path.unlink()
                tqdm.write(f"{msg_prefix} Moving {src_path} to {dst_path}...")
                os.makedirs(Path(dst_path).parent, exist_ok=True)
                shutil.move(src_path, dst_path)
                moved_files[str(src_path)] = str(dst_path)
        if moved_files:
            dispatcher_filename = Path(os.path.join(dst_folder, f"Input_{kind}", "start"))
            dispatcher_filename.touch(exist_ok=True)
            post_disc(f"{kind} dispatcher file created at ```json\n{dispatcher_filename}```")
    post_disc(f"The following files were moved:\n```json\n{pformat(moved_files, indent=2)}```")
