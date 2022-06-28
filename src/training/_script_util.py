
from pathlib import Path
import torch
import pandas as pd
import GPUtil

from azureml.core import Run, Workspace
from typing import Dict
from azureml.core import Model
from ignite.engine import Engine
from ignite.handlers import Events
from torch.utils.data import DataLoader
from azureml.exceptions import WebserviceException
from pydantic import BaseModel


from ._model import save_model, load_model, LogoModel, get_resnet18
from ._util import save_ss_index, load_ss_index, gen_ss_index, flatten
from indexmodel import SSIndex
from ._azure_util import azure_log_metrics
from ._dataset import SimpleDataset, trans, TripletIndexDataset


class TrainConf:

    class _TrainConf(BaseModel):
        ss_index_name: str
        ss_index_version: int
        model_name: str
        model_version: int
        validation_dataset_path: str
        test_dataset_path: str
        train_dataset_path: str
        root_data_path: str
        k: int
        gamma: float
        device: str
        batch_size: int
        num_workers: int
        lr: float
        margin: float
        max_epochs: int
        num_samples_epoch: int
        interval_update_index: int
        max_loss: float
        max_dis: float
        loss_weight: float
        freeze: int

    def __init__(self, **kwargs) -> None:
        self._conf = self._TrainConf(**kwargs)
        self.run = Run.get_context()
        self.ws = self._get_ws()
        self.ssindex = self._get_index()
        self.model = self._get_model()


    @property
    def triplet_batch_size(self):
        batch_size = int(self._conf.batch_size / 3)
        return batch_size if batch_size > 1 else 1
    
    def _get_ws(self):
        try:
            ws = self.run.experiment.workspace
        except AttributeError as e:
            print("no workspace azure offline run mode?")
            #ws = Workspace.from_config("./notebooks/azure/.azureml/config.json")
            return None
        else:
            return ws

    def _get_model(self):
        if self._conf.model_name:
            return retrieve_model(self._conf.model_name, self._conf.model_version,
                                  self.ws, LogoModel(get_resnet18(cut=1)))
        else:
            model = LogoModel(get_resnet18(cut=1))
            model_meta_data = {"tags": {"type": "logo_model"}}
            register_model(model, f"./outputs/model/", f"{self.run._run_id}.ptm", self.ws, **model_meta_data)
            return model

    def _get_index(self):
        if self._conf.ss_index_name:
            index, _= retrieve_ss_index(self._conf.ss_index_name, self._conf.ss_index_version,
                                        self.ws, "class_id")
            return index


def register_model(model, path, name, workspace, **kwargs):
    save_model(model, path, name)
    try:
        Model.register(workspace, str(Path(path) / name), name, **kwargs)
    except Exception as e:
        print(e)
        print("no workspace azure offline run mode?")
        


def retrieve_model(name, version, workspace, class_of_model, cpu=False, target_dir=".", exist_ok=True):
    if version == -1:
        version = None
    try:
        Model(workspace, name, version=version).download(target_dir= target_dir, exist_ok=exist_ok)
    except WebserviceException:
        pass
    return load_model(f"{target_dir}/{name}", class_of_model, cpu)


def register_ss_index(meta, tensor, path, workspace, **kwargs):
    save_ss_index(meta, tensor, path)
    try:
        Model.register(workspace, path, Path(path).name, **kwargs)
    except Exception as e:
        print(e)
        print("no workspace azure offline run mode?")
        

def retrieve_ss_index(name, version, workspace, mapping, target_dir=".", exist_ok=True):
    if version == -1:
        version = None
    try:
        azure_index = Model(workspace, name, version=version)
        azure_index.download(exist_ok=exist_ok, target_dir= target_dir)
    except WebserviceException:
        pass
    return load_ss_index(f"{target_dir}/{name}", mapping), azure_index


def gen_new_ssindex(dataloader, meta_data, engine: Engine, eos, mapping: str) -> SSIndex:
    # run the evaluator
    engine.run(dataloader)
    # gen ssindex
    tensors = torch.stack(flatten([x['embeddings'] for x in eos.data])).view(-1, 64)
    eos.reset()
    new_ssindex = gen_ss_index(tensors.cpu().numpy(), meta_data, mapping)
    # reset outptut?
    print("gen index:")
    GPUtil.showUtilization()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    GPUtil.showUtilization()
    return new_ssindex, tensors


def _evaluate_event(_, evaluator, run, dataloader: Dict):
        for name, loader in dataloader.items():
            with evaluator.add_event_handler(Events.COMPLETED, azure_log_metrics, run, f"{name}_"):
                evaluator.run(loader)


def _update_ssindex(trainer, engine, eos, dataloader, index_meta_data, run, ws, _meta_data = {}, update_index= False):
    print("update index")
    ssindex, tensors =  gen_new_ssindex(dataloader, index_meta_data, engine, eos, "class_id")
    if update_index:
        trainer.state.dataloader.dataset.model = ssindex
        trainer.set_data(trainer.state.dataloader)
    register_ss_index(index_meta_data, tensors, f"./outputs/{run._run_id}", ws, **_meta_data)


def _init_index(engine, eos, dataloader, index_meta_data, run, ws, _meta_data = {}):
    print("update index")
    ssindex, tensors =  gen_new_ssindex(dataloader, index_meta_data, engine, eos, "class_id")
    register_ss_index(index_meta_data, tensors, f"./outputs/{run._run_id}", ws, **_meta_data)


def _register_model(trainer, model, run, ws, _meta_data = {}):
    print("register model")
    register_model(model,"./outputs/model/", f"{run._run_id}.ptm", ws, **_meta_data)


def _transform_labels(output):
    # https://github.com/pytorch/ignite/issues/971
    y = [torch.tensor(1) for _ in range(len(output["y"]))]
    y_pred = [torch.tensor(1) if a == b[0] else torch.tensor(0) for a, b in zip(output["y"], output["y_pred"]) ]
    return torch.stack(y), torch.stack(y_pred)


def _get_special_classes(output):
    return flatten([[y, y_pred[0]] for y, y_pred in  zip(output["y"], output["y_pred"]) if y != y_pred[0]])

def _transform_labels_(output):
    pass



def get_index(conf, dataloader, meta_data, engine ,eos):
    if conf.ssindex is None:
        # generate index:          
        ssindex, _ =  gen_new_ssindex(dataloader, meta_data, engine ,eos, "class_id")
        return ssindex
    else:
        return conf.ssindex


def gen_logo_dataloader(meta_data_path, root_data_path, batch_size, num_workers, label="class_id"):
    df = pd.read_parquet(meta_data_path)
    logo_data = SimpleDataset(df["path"], trans, df['geometry'], df[label], root_data_path, False)
    logo_dataloader = DataLoader(logo_data, batch_size, num_workers=num_workers)
    return df, logo_data, logo_dataloader


def gen_logo_triplet_dataloader(index, k, len, root_path, batch_size, num_workers):
    logo_triplet_dataset = TripletIndexDataset(index, k, trans, len, root_path, True)
    triplet_loader = DataLoader(logo_triplet_dataset, batch_size, num_workers=num_workers)
    return triplet_loader


def get_param_to_train(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def toggel_freeze(model: torch.nn.Module, num: int,
                  requires_grad: bool = False):

    layers = get_children(model)
    if num > 0:
        layers = layers[: -num]

    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = requires_grad


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def score_factory(trainer):
    def score_function(engine):
        class_acc = engine.state.metrics['class_acc']
        print("early stopping:", class_acc)
        if class_acc == 0:
            return 0
        else:
            return trainer.state.epoch
    return score_function
