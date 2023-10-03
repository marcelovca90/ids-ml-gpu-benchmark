from modules.evaluation import evaluator
from modules.evaluation.hierarchical_sampled_batch import \
    CustomHierarchicalClassifier
from modules.evaluation.lce import LCE
from modules.preprocessing.custom.iot_23 import IoT_23
from modules.preprocessing.custom.mqtt_iot_ids2020 import \
    MQTT_IoT_IDS2020_BiflowFeatures

if __name__ == "__main__":

    # iot_23 = IoT_23().pipeline(preload=False, prepare=True)

    biflow = MQTT_IoT_IDS2020_BiflowFeatures()

    biflow.prepare()
    biflow.load()

    print('wait')
    pass

    # evaluator.warmup(iot_23)
    # evaluator.baseline(iot_23)

    # iot_23 = IoT_23().pipeline(preload=True, prepare=False)
    # # BR.run(iot_23)
    # # LCE.run(iot_23)
    # CustomHierarchicalClassifier.run(iot_23)

    pass
