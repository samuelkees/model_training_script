from ignite.engine import Engine
from azureml.core import Run


def azure_log_metrics(engine: Engine, run: Run, name = "", x= None):
    print("log metrics")
    if x is None:
        x = engine.state.epoch
    for key, value in engine.state.metrics.items():
        run.log_row(name + key, x=x, y=value)

    if "class_acc" in engine.state.metrics.keys():
        run.log("class_acc", engine.state.metrics["class_acc"])
