from pathlib import Path
import os
import shutil

input_folder = Path("/home/automl/git/iot-threat-classifier/2025-07-05/Input_Multiclass")
output_folder = Path("/home/automl/git/iot-threat-classifier/2025-07-05/Input_Multiclass_3600")


# Find all .parquet files recursively
parquet_files = list(input_folder.rglob("*.parquet"))

for curr_file in parquet_files:

    curr_input_file_src_parquet = curr_file
    curr_input_file_src_json = Path(str(curr_file).replace('.parquet', '.json'))
    curr_input_file_src_html = Path(str(curr_file).replace('.parquet', '.html'))

    curr_output_folder = os.path.join(output_folder, curr_file.name.replace('.parquet', ''))
    os.makedirs(curr_output_folder, exist_ok=True)

    curr_input_file_dst_parquet = Path(str(curr_file).replace(str(input_folder), curr_output_folder))
    curr_input_file_dst_json = Path(str(curr_input_file_src_json).replace(str(input_folder), curr_output_folder))
    curr_input_file_dst_html = Path(str(curr_input_file_src_html).replace(str(input_folder), curr_output_folder))

    shutil.copyfile(curr_input_file_src_parquet, curr_input_file_dst_parquet)
    shutil.copyfile(curr_input_file_src_json, curr_input_file_dst_json)
    shutil.copyfile(curr_input_file_src_html, curr_input_file_dst_html)
