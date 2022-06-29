from azureml.core import Run
import click
import torch
from pathlib import Path
import faiss
import os
import numpy as np
from azureml.core import Model


@click.command()
@click.option("--root_data_path", type=str)
@click.option("--lr", type=float)
def main(**kwargs):

    # logging 
    run = Run.get_context()
    for x in np.linspace(0, 10):
        y = np.sin(x)
        run.log_row('sine', x=x, y=y, z=y)

    print(kwargs)

    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not avaliable")

    print("faiss version", faiss.__version__)
    print("click version", click.__version__)

    print(os.listdir(kwargs['root_data_path'])[:25])

    # save to ./outputs
    try: 
        os.makedirs("./outputs/test/")
    except OSError as e:
        print("error by saveing data")
        print(e)
    with open("./outputs/test/file.txt", "w") as f:
        f.write("this is a test of the output folder")

    with open("./outputs/test/file.txt", "r") as f:
        print(f.read())

    print("register model")
    # register model:

    ws = run.experiment.workspace
    Model.register(ws, "./outputs/test/file.txt", "not_a_model_just_a_test_1")
    print("download model...")
    model = Model(ws, "not_a_model_just_a_test_1")
    model.download(target_dir="./")

    with open("./file.txt", "r") as f:
        print(f.read())


    


if __name__ == "__main__":
    print("start script", Path(__file__).name, "...")
    try:
        main()
    except Exception as e:
        print("Hey Exception \n ", e)