import pickle
import torch
import os

from tqdm import tqdm
from torch import nn
from ml2rt import load_model, save_torch
from pathlib import Path
from torchvision import models
from torch.nn.functional import mse_loss

def save_data(data, name, save_dir):
    pickle.dump(data, open(f"{save_dir}/part_{name}.pickle", "bw"))
    print("save", name)


def run_model(model, loader, save_interval, save_dir, drive="cuda"):
    model.eval()
    feature_vectors = []
    model.to(drive)
    for i, data in enumerate(tqdm(loader)):
        with torch.no_grad():
            feature_vectors.append(model(data.to(drive)))
        if (i+1) % save_interval == 0:
            save_data(feature_vectors, i+1, save_dir)
    save_data(feature_vectors, "done", save_dir)


def export_model_to_script(model, path, example_input):
    # https://pytorch.org/tutorials/advanced/cpp_export.html
    model.eval()
    traced_script_module = torch.jit.trace(model, example_input)
    save_torch(traced_script_module, path)


def save_model(model, path:str, name: str):
    try:
        output_path = Path(path)
        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(),  output_path / name)
        print("model saved...")
    except Exception as e:
        print("model not saved...")
        print(e)


def load_model(path_state_dict: str, model: nn.Module, cpu=False):
    kwargs = {}
    if path_state_dict:
        if cpu:
            kwargs["map_location"] = "cpu"
        model.load_state_dict(torch.load(path_state_dict, **kwargs))
    return model

def get_resnet18(cut=None):
    model = models.resnet18(pretrained=True)
    if cut:
        model = cut_model(model, cut)
    return model


def cut_model(model, cut) -> torch.nn.modules.container.Sequential:
    return torch.nn.Sequential(*(list(model.children())[:-cut]))


class BottleModel(nn.Module):
    def __init__(self, resNet1, resNet2, resNet3):
        super(BottleModel, self).__init__()
        self.resNet1 = resNet1
        self.resNet2 = resNet2
        self.resNet3 = resNet3

        self.imageNet = nn.Sequential(
                                        nn.Linear(1536, 512),
                                        nn.PReLU(),
                                        nn.Linear(512, 255),
                                        nn.PReLU(),
                                        nn.Linear(255, 64)
                                        )
        
        self.combinNet = nn.Sequential(
                                        nn.Linear(64 + 1, 128),
                                        nn.PReLU(),
                                        nn.Linear(128, 64)
                                        )


    def forward(self, img_1, img_2, img_3, img_4, img_5, img_6, x_geometric):
        x1 = self.resNet1(torch.cat((img_1, img_2), dim=2)).view(-1, 512)
        x2 = self.resNet2(torch.cat((img_3, img_4), dim=2)).view(-1, 512)
        x3 = self.resNet3(torch.cat((img_5, img_6), dim=2)).view(-1, 512)

        x = self.imageNet(torch.cat((x1, x2, x3), dim=1))
        x_2 = self.combinNet(torch.cat((x, x_geometric), dim=1))
        return torch.add(x, x_2)


class LogoModel(torch.nn.Module):
    def __init__(self, resNet):
        super(LogoModel, self).__init__()
        self.resNet = resNet
        self.embeddingNet = torch.nn.Sequential(
                                            nn.Linear(512, 256),
                                            nn.BatchNorm1d(256),
                                            nn.PReLU(),
                                            nn.Linear(256, 128),
                                            nn.BatchNorm1d(128),
                                            nn.PReLU(),
                                            nn.Linear(128, 64)
                                        )
        self.combineNet = torch.nn.Sequential(
                                            nn.Linear(64 + 3, 128),
                                            nn.BatchNorm1d(128),
                                            nn.PReLU(),
                                            nn.Linear(128, 64)   
                                        )

    def forward(self, image, geometry):
        image = self.resNet(image)
        image = self.embeddingNet(image.view(-1, 512))
        geo_img = self.combineNet(torch.cat((image, geometry), dim=1))
        # L2 loss for combineNet?
        if self.training:
            return image, torch.add(image, geo_img)
        return torch.add(image, geo_img)
    

class TripletLogoModel(torch.nn.Module):
    def __init__(self, logo_model: nn.Module):
        super(TripletLogoModel, self).__init__()
        self.model = logo_model

    def forward(self, anchor, positive, negative):
            a_image, a_geometry = self.model(*anchor)
            p_image, p_geometry = self.model(*positive)
            n_image, n_geometry  = self.model(*negative)
            
            return {"x": {"anchor": a_image,
                                "positive": p_image,
                                "negative": n_image},
                    "y": {"anchor": a_geometry,
                                    "positive": p_geometry,
                                    "negative": n_geometry }} 


class TripletAcc(nn.Module):
    def __init__(self, margin):
        super(TripletAcc, self).__init__()
         # <- set eps up if you get division by zero errores.. ->
        self.pdist = nn.PairwiseDistance(p=2, eps=0) 
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        p_dis_to_a = self.pdist(anchor, positive)
        n_dis_to_a = self.pdist(anchor, negative)

        return torch.sum((p_dis_to_a + self.margin) < n_dis_to_a) / len(p_dis_to_a)


class ClampedTripledLoss(torch.nn.Module):
    def __init__(self, max_value, margin):
        super().__init__()
        self._loss = torch.nn.TripletMarginLoss(margin, reduction='none')
        self.UPPERVALUE = max_value

    def forward(self, anchor, positive, negative):
        loss = self._loss(anchor, positive, negative)
        clamp_loss = torch.clamp(loss, max=self.UPPERVALUE)
        big_loss = loss == clamp_loss
        return clamp_loss.mean(), big_loss


class DoubleTripledLoss(torch.nn.Module):
    def __init__(self, max_value, margin, weight):
        super().__init__()
        self._loss = ClampedTripledLoss(max_value, margin)
        self._weight = weight

    def forward(self, x, y):
        x_loss, _ = self._loss(**x)
        y_loss, _ = self._loss(**y)
        self.x_loss = x_loss
        self.y_loss = y_loss

        return (x_loss * self._weight) + (y_loss * (1-self._weight))


