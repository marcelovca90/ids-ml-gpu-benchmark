from modules.evaluation import evaluator
from modules.preprocessing.custom.iot_23 import IoT_23

if __name__ == "__main__":

    iot_23 = IoT_23().pipeline(preload=False, prepare=True)

    evaluator.warmup(iot_23)

    evaluator.baseline(iot_23)

    pass
