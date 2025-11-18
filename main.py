
from tqdm import tqdm

from constants import BINARIZE_FLAGS, SAMPLE_FRACS, SEEDS
from modules.preprocessing.preproc_utils import safe_exec
from modules.filesystem.file_utils import copy_files

from modules.preprocessing.custom.bot_iot_macro import BoT_IoT_Macro
from modules.preprocessing.custom.bot_iot_micro import BoT_IoT_Micro
from modules.preprocessing.custom.cic_ids_2017 import CIC_IDS_2017
from modules.preprocessing.custom.cic_iot_dataset2023 import \
    CIC_IOT_Dataset2023
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
from modules.preprocessing.custom.CICIoMT2024_Bluetooth import \
    CICIoMT2024_Bluetooth
from modules.preprocessing.custom.CICIoMT2024_WiFi_and_MQTT import \
    CICIoMT2024_WiFi_and_MQTT
from modules.preprocessing.custom.unsw_nb15 import UNSW_NB15

# PYTHONPATH=. python main.py
if __name__ == "__main__":

    dataset_classes = [
        # BoT_IoT_Macro,
        # BoT_IoT_Micro,
        # CIC_IDS_2017,
        # CICIoMT2024_Bluetooth,
        # CICIoMT2024_WiFi_and_MQTT,
        # CIC_IOT_Dataset2023,
        # IoT_23,
        # IoT_Network_Intrusion_Macro,
        # IoT_Network_Intrusion_Micro,
        KDD_Cup_1999,
        # MQTT_IoT_IDS2020_BiflowFeatures,
        # MQTT_IoT_IDS2020_PacketFeatures,
        # MQTT_IoT_IDS2020_UniflowFeatures,
        # UNSW_NB15
    ]

    # independent datasets (must be run separately):
    # - BCCC
    # - CICAPT_IIoT
    # - CICEVSE2024
    # - CICIoV2024
    # - EDGE_IIOTSET
    # - N_BaIoT
    # - NIDS
    # - ToN_IoT

    # ~/.bashrc
    # alias regen='PYTHONPATH=. python main.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/bccc.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/CICAPT_IIoT.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/CICEVSE2024.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/CICIoV2024.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/edge_iiotset.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/n_baiot.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/nids.py ; \
    # PYTHONPATH=. python modules/preprocessing/custom/ton_iot.py ; \
    # PYTHONPATH=. python modules/filesystem/utils.py ; \
    # PYTHONPATH=. python modules/preprocessing/complexity_gpu.py'

    for d, dataset_cls in enumerate(tqdm(dataset_classes, desc='Dataset', leave=False)):

        msg_prefix = f"[{d+1:02}/{len(dataset_classes):02}]"
        ds_name = dataset_cls.__name__

        for b, binarize_flag in enumerate(tqdm(BINARIZE_FLAGS, desc='Binarize', leave=False)):

            for s, seed in enumerate(tqdm(SEEDS, desc='Seed', leave=False)):

                # --- 1. Full Run (frac == 1.0) ---
                # We run this first to generate the base artifacts.
                suffix_full = f"binarize={binarize_flag} sample_frac=1.0 seed={seed}"
                
                success = safe_exec(
                    runnable=lambda: dataset_cls(sample_frac=1.0, seed=seed, binarize=binarize_flag).pipeline(preload=False),
                    msg_prefix=msg_prefix,
                    dataset_name=ds_name,
                    msg_suffix=suffix_full
                )

                # If the full run failed, we MUST skip the sampled runs for this seed
                # because the base artifacts won't exist.
                if not success:
                    continue

                # --- 2. Sampled Runs (frac < 1.0) ---
                for sample_frac in tqdm(SAMPLE_FRACS, desc='Fraction', leave=False):
                    if sample_frac == 1.0:
                        continue # Already done in step 1

                    suffix_sampled = f"binarize={binarize_flag} sample_frac={sample_frac} seed={seed}"

                    safe_exec(
                        runnable=lambda: dataset_cls(sample_frac=sample_frac, seed=seed, binarize=binarize_flag).pipeline(preload=True),
                        msg_prefix=msg_prefix,
                        dataset_name=ds_name,
                        msg_suffix=suffix_sampled
                    )
    
    # 3; Final Cleanup / Organization
    safe_exec(
        runnable=lambda: copy_files(),
        msg_prefix="[FINAL]",
        dataset_name="Main",
        msg_suffix="Copying Files"
    )