from modules.evaluation import evaluator
from modules.logging.logger import log_print
from modules.preprocessing.custom.mqtt_iot_ids2020_biflow import \
    MQTT_IoT_IDS2020_BiflowFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_packet import \
    MQTT_IoT_IDS2020_PacketFeatures
from modules.preprocessing.custom.mqtt_iot_ids2020_uniflow import \
    MQTT_IoT_IDS2020_UniflowFeatures

if __name__ == "__main__":

    for cls in [
        MQTT_IoT_IDS2020_BiflowFeatures,
        MQTT_IoT_IDS2020_UniflowFeatures,
        # MQTT_IoT_IDS2020_PacketFeatures
    ]:

        for fix_imbalance in [False, True]:

            for use_weights in [False, True]:

                try:
                    pipe = cls()
                    pipe.prepare()
                    pipe.load()
                    pipe.set_dtypes()
                    pipe.sanitize()
                    pipe.encode()
                    pipe.shrink_dtypes()
                    pipe.drop_irrelevant_features()
                    pipe.remove_na_duplicates()
                    pipe.sort_columns()
                    pipe.reset_index()
                    pipe.update_metadata()
                    pipe.save()

                    evaluator.baseline(pipe, fix_imbalance, use_weights)

                except Exception as e:
                    log_print(e)
