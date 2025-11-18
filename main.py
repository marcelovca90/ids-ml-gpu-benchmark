
from tqdm import tqdm

from modules.logging.logger import log_print
from modules.logging.webhook import post_disc
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

def _log_event(prefix, dataset_name, suffix, stage):
    """
    Unified logging for any pipeline stage.
    stage ∈ {"start", "finish", "error"}
    """
    if stage == "start":
        msg = f"{prefix} Started {dataset_name} ({suffix})."
    elif stage == "finish":
        msg = f"{prefix} Finished {dataset_name} ({suffix})."
    elif stage == "error":
        msg = f"{prefix} ERROR in {dataset_name} ({suffix})."
    else:
        raise ValueError(f"Unknown logging stage: {stage}")

    log_print(msg)
    post_disc(msg)

# PYTHONPATH=. python main.py
if __name__ == "__main__":

    binarize_flags = [False]

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

    sample_fracs = [1.0, 0.2, 0.1, 0.05]

    seeds = [17, 23, 37, 53, 89]
    
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
    # PYTHONPATH=. python move_files.py ; \
    # PYTHONPATH=. python modules/preprocessing/complexity_gpu.py'

    for d, dataset_cls in enumerate(tqdm(dataset_classes, desc='Dataset', leave=False)):
        for b, binarize_flag in enumerate(tqdm(binarize_flags, desc='Binarize', leave=False)):
            for s, seed in enumerate(tqdm(seeds, desc='Seed', leave=False)):

                # Full run (always first)
                sample_frac = 1.0
                msg_prefix = f"[{d+1:02}/{len(dataset_classes):02}]"
                msg_suffix = f"b={binarize_flag} f=1.0 s={seed}"

                try:
                    _log_event(msg_prefix, dataset_cls.__name__, msg_suffix, "start")
                    dataset_cls(sample_frac=1.0, seed=seed, binarize=binarize_flag).pipeline(preload=False)
                    _log_event(msg_prefix, dataset_cls.__name__, msg_suffix, "finish")
                except Exception as e:
                    _log_event(msg_prefix, dataset_cls.__name__, f"{msg_suffix} — {e}", "error")
                    continue

                # Sampled runs (all other fracs)
                for sample_frac in tqdm(sample_fracs, desc='Fraction', leave=False):
                    if sample_frac == 1.0:
                        continue  # skip; already done

                    msg_suffix = f"b={binarize_flag} f={sample_frac} s={seed}"

                    try:
                        _log_event(msg_prefix, dataset_cls.__name__, msg_suffix, "start")
                        dataset_cls(sample_frac=sample_frac, seed=seed, binarize=binarize_flag).pipeline(preload=True)
                        _log_event(msg_prefix, dataset_cls.__name__, msg_suffix, "finish")
                    except Exception as e:
                        _log_event(msg_prefix, dataset_cls.__name__, f"{msg_suffix} — {e}", "error")
