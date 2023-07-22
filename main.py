from modules.evaluation.evaluator import baseline, warmup
from modules.preprocessing.custom.iot_23 import IoT_23

if __name__ == "__main__":

    iot_23 = IoT_23().setup()

    warmup(iot_23)

    baseline(iot_23)

    pass
