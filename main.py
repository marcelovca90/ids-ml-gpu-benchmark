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
from modules.preprocessing.custom.unsw_nb15 import UNSW_NB15

if __name__ == "__main__":

    for cls in [
        # IoT_23,
        # MQTT_IoT_IDS2020_PacketFeatures,
        # MQTT_IoT_IDS2020_UniflowFeatures,
        # MQTT_IoT_IDS2020_BiflowFeatures,
        # IoT_Network_Intrusion_Macro,
        # IoT_Network_Intrusion_Micro,
        # KDD_Cup_1999,
        # BoT_IoT_Micro,
        # BoT_IoT_Macro
        UNSW_NB15
    ]:

        try:
            cls().pipeline()
        except Exception as e:
            log_print(e)
