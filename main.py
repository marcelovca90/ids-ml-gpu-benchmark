import os
import shutil
from pathlib import Path

from tqdm import tqdm

from modules.logging.logger import log_print
from modules.preprocessing.custom.bot_iot_macro import BoT_IoT_Macro
from modules.preprocessing.custom.bot_iot_micro import BoT_IoT_Micro
from modules.preprocessing.custom.iot_23 import IoT_23
from modules.preprocessing.custom.iot_network_intrusion_macro import \
    IoT_Network_Intrusion_Macro
from modules.preprocessing.custom.iot_network_intrusion_micro import \
    IoT_Network_Intrusion_Micro
from modules.preprocessing.custom.kdd_cup_1999 import KDD_Cup_1999
from modules.preprocessing.custom.mqtt_iot_ids2020_biflow import \
    MQTT_IoT_IDS2020_BiflowFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_packet import \
    MQTT_IoT_IDS2020_PacketFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_uniflow import \
    MQTT_IoT_IDS2020_UniflowFeatures
from modules.preprocessing.custom.nids import NIDS
from modules.preprocessing.custom.unsw_nb15 import UNSW_NB15
from modules.preprocessing.utils import now

if __name__ == "__main__":

    binarize_flags = [True, False]

    dataset_classes = [
        BoT_IoT_Micro,
        BoT_IoT_Macro,
        IoT_23,
        IoT_Network_Intrusion_Macro,
        IoT_Network_Intrusion_Micro,
        KDD_Cup_1999,
        MQTT_IoT_IDS2020_PacketFeatures,
        MQTT_IoT_IDS2020_UniflowFeatures,
        MQTT_IoT_IDS2020_BiflowFeatures,
        NIDS,
        UNSW_NB15
    ]

    for binarize_flag in tqdm(binarize_flags, desc='Binarize', leave=False):

        for dataset_cls in tqdm(dataset_classes, desc='Dataset', leave=False):

            try:
                log_print(f'Started processing {dataset_cls.__name__}.')
                dataset_cls(binarize=binarize_flag).pipeline()
                log_print(f'Finished processing {dataset_cls.__name__}.')
            except Exception as e:
                log_print(f'Error processing {dataset_cls.__name__}: {str(e)}')

    candidate_files = list(Path("datasets").rglob("*"))
    for src_path in tqdm(candidate_files, desc='Candidate', leave=False):
        for kind in tqdm(['Binary', 'Multiclass'], desc='Kind', leave=False):
            if (src_path.is_file() and kind in src_path.name and
                ('generated' in str(src_path.absolute().resolve())) and
                ('.parquet' in src_path.name or '.json' in src_path.name)):
                dst_path = Path(os.path.join('ready', kind, src_path.name))
                if dst_path.is_file() and dst_path.exists():
                    dst_path.unlink()
                tqdm.write(f"[{now()}] Moving {src_path} to {dst_path}...")
                os.makedirs(Path(dst_path).parent, exist_ok=True)
                shutil.move(src_path, dst_path)