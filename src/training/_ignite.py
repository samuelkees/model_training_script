from torch.nn import Module
import torch
import numpy as np

from typing import Callable, Tuple
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from _model import TripletAcc
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

from indexmodel import SSIndex, quantifier, confidence, min_dis
from _script_util import flatten


def inference_factory(model: Module, device: torch.device, 
                      index: SSIndex, k: int, gamma: float, 
                      prepair_batch_fun: Callable, loss_func=None, max_dis=None):
    def inner(engine, batch):
        result = {}
        # set model in eval mode
        model.eval() # <-- is this same as no_grad?
        model.to(device)
        # get data from batch
        *data, y = prepair_batch_fun(batch, device) 
        # inference 
        with torch.no_grad():
            embeddings = model(*data)
        result["embeddings"] = embeddings 
        result["y"] = y
        # similarity search 
        if index is not None:
            neighbors = index.search(embeddings.view(-1, index.index.d).cpu().numpy().astype(np.float32), k)
            result["y_pred"] = flatten(quantifier(neighbors, confidence(gamma, 1), max_dis))
            result["neighors"] = neighbors
        # calc loss
        if loss_func is not None:
            loss = loss_func(**embeddings)
            result["loss"] = loss
        
        return result
    return inner


def train_factory(model: Module, optimizer: Optimizer, loss_func: _Loss, device: torch.device, 
                      prepair_batch_fun: Callable):
    def inner(engine, batch):
        result = {}
        # set model in train mode
        model.train()
        model.to(device)
        optimizer.zero_grad() # <- only use gradients from "this" mini-batch
        # get data from batch
        *data, y = prepair_batch_fun(batch, device) 
        # inference 
        embeddings = model(*data)
        # calculated loss
        loss = loss_func(**embeddings)
        
        try:
            result["image_loss"] = loss_func.x_loss 
            result["geomertry_loss"] = loss_func.y_loss
        except Exception:
            pass

        # backprob loss
        loss.backward()
        # update gradients
        optimizer.step()
        
        result["embeddings"] = embeddings
        result["loss"] = loss
        result["y"] = y
        return result
    return inner


def get_big_triplets(differens, batch):
    return [{"anchor": a,
              "positive": p,
              "negative": n} for i, a, p, n in zip(differens, batch["anchor"]["image_id"], 
                                    batch["positive"]["image_id"], batch["negative"]["image_id"]) if not i]


def logo_prepair_batch(batch, device):
    image_data = batch["image"]
    image_data = image_data.to(device)
    geometry = batch["geometry"]
    geometry = geometry.to(device)
    y = batch["y"]

    return image_data, geometry, y


def logo_triplet_prepair_batch(batch, device):
    # anchor, positive, negative
    *anchor, anchor_y = logo_prepair_batch(batch['anchor'], device)
    *positive, positive_y = logo_prepair_batch(batch['positive'], device)
    *negative, negative_y = logo_prepair_batch(batch['negative'], device)
    y = {"anchor": anchor_y, "positive": positive_y, "negative": negative_y}
    
    return anchor, positive, negative, y


class TripletAccuracy(Metric):

    def __init__(self, margin, output_transform=lambda x: x, device="cpu"):
        TripletAccuracy.required_output_keys =  ("anchor", "positive", "negative")    
        self._pdist = TripletAcc(margin)
        self._sum_of_accuracy = None
        self._num_examples = None
        super(TripletAccuracy, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_of_accuracy = torch.tensor(0, device=self._device, dtype=torch.float32)
        self._num_examples = 0
        super(TripletAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        anchor, positive, negative = output[0].detach(), output[1].detach(), output[2].detach()
        
        t_acc = self._pdist(anchor, positive, negative)
        self._sum_of_accuracy += torch.sum(t_acc).to(self._device)
        self._num_examples += 1

    @sync_all_reduce("_num_examples", "_sum_of_accuracy:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._sum_of_accuracy.item() / self._num_examples
