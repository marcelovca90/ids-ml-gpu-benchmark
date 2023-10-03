from modules.evaluation import evaluator
from modules.logging.logger import log_print
from modules.preprocessing.custom.mqtt_iot_ids2020_biflow import \
    MQTT_IoT_IDS2020_BiflowFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_packet import \
    MQTT_IoT_IDS2020_PacketFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_uniflow import \
    MQTT_IoT_IDS2020_UniflowFeatures

if __name__ == "__main__":

    for cls in [MQTT_IoT_IDS2020_BiflowFeatures,
                MQTT_IoT_IDS2020_UniflowFeatures,
                MQTT_IoT_IDS2020_PacketFeatures]:

        for use_weights in [False, True]:

            try:

                pipe = cls()

                pipe.prepare()
                pipe.load()
                pipe.sanitize()
                pipe.set_dtypes()
                pipe.encode()
                pipe.shrink_dtypes()
                pipe.select_features()
                pipe.sort_columns()
                pipe.remove_na_duplicates()
                pipe.reset_index()

                evaluator.baseline(pipe, use_class_weights=use_weights)

            except Exception as e:

                log_print(e)
