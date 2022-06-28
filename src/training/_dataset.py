from torch.utils.data import Dataset, DataLoader
from torchvision import get_image_backend, transforms
# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
from torch.utils.data import IterableDataset, DataLoader
from indexmodel import SSIndex
import numpy as np
import torch
import random

from PIL import Image
from pathlib import Path


class SimpleDataset(Dataset):
    def __init__(self, paths, transfrom, geometry, label = None, root_dir="", image_mode=False):
        self.paths = paths 
        self.label = label
        self.geometry = geometry
        if self.label is None:
            self.label = [None for x in paths]
        self.transfrom = transfrom
        self.root_dir = root_dir
        self.image_mode = image_mode
        
    def __getitem__(self, index):
        
        if self.root_dir:
            path = Path(self.root_dir) / self.paths[index]
        else:
            path = self.paths[index]

        if self.image_mode:
            return {"image": Image.open(path), 
                    "geometry": self.geometry[index],
                    "y": self.label[index]}
        else:
            return {"image": self.transfrom(Image.open(path)), 
                    "geometry": torch.tensor(self.geometry[index], dtype=torch.float32),
                    "y": self.label[index]}

    
    def __len__(self):
        return len(self.paths)

trans = transforms.Compose([
   #         transforms.Resize(256),
   #         transforms.CenterCrop(224),
             transforms.ToTensor(),
           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def scaling_min_max(data):
    return (data-data.min()) / (data.max() - data.min())


class TripletIndexDataset(IterableDataset):
    def __init__(self, IndexModel: SSIndex, k: int, transform: transforms, stop: int, root_path:str = None, load=False):
        """Generates triplets from a SSIndex

        Parameters
        ----------
        IndexModel : SSIndex
            Expects "image_id", class_id", "geometry" and "path" in meta of SSIndex

        """
        super(TripletIndexDataset).__init__()
        self.model = IndexModel
        self.trans = transform
        self._k = k
        self._load = load
        self._root_path = root_path
        self.stop = stop
        self._special_classes = None
        self._p = 0.66
        self._label = "class_id"
        
    def _gen_search_df(self, d, p, columns):
        search_data = self.model.meta.iloc[p[0]][columns]
        search_data['dis'] = d[0] 
        return search_data
        
    def _draw_sample(self, item, df, is_neg):
        if is_neg:
            _df = df[df[self._label] != item[self._label].values[0]]
            weights= 1 - scaling_min_max(_df['dis'])
        else:
            _df = df[df[self._label] == item[self._label].values[0]]
            weights= scaling_min_max(_df['dis'])
        
        try:
            result = _df.sample(1, weights=weights)
        except ValueError:
            if _df.shape[0] != 0:
                result = _df.sample(1)
            else:
                raise Exception("can't draw a sample from population of 0")
         
        return result
    
    def _search(self, item_index):
        x = self.model.index.reconstruct(item_index)
        dis, pos = self.model.index.search(np.expand_dims(x, axis=0), k=self._k)
        return self._gen_search_df(dis, pos, self.model.meta.keys())
         
        
    def _get_image_and_label(self, df):
        x = str(Path(self._root_path) / Path(df["path"].iloc[0]))
        y = df["class_id"].iloc[0]
        if "dis" in df.keys():
            d = df["dis"].iloc[0]
            return x, y, d
        return x, y

    def _get_geometics(self, df):
        geometry = df["geometry"].iloc[0]
        return torch.tensor(geometry, dtype=torch.float32)

    
    def _get_image_id(self, df):
        return str(df["image_id"].iloc[0])


    def _draw_item(self):
        if self._special_classes:
            if random.random() < self._p:
                pool = self.model.meta[self.model.meta[self._label].isin(self._special_classes)]
                return pool.sample(1)
        return self.model.meta.sample(1)


    def __len__(self):
        return self.stop


    def __iter__(self):
        while True:
            # draw a random item from the index 
            item = self._draw_item()
            # seach in the index for the neighbors of that item by using the item_index
            neighbors = self._search(item.index.to_list()[0])

            # Bulid Triplet 
            # (1) ancker = Image
            ax, ay = self._get_image_and_label(item)
            ag = self._get_geometics(item) 

            # (2) negative = the closest image, witch comes from a different group then the group of the ancker image
            n_item = self._draw_sample(item, neighbors, True)
            ng = self._get_geometics(n_item)
            nx, ny, _nd = self._get_image_and_label(n_item)
            # (3) positive = the furthest away image, witch comes from the same group...
            p_item = self._draw_sample(item, neighbors, False)
            pg = self._get_geometics(p_item)
            px, py, _pd = self._get_image_and_label(p_item)

            # load images...
            if self._load:
                ax = self.trans(Image.open(ax))
                px = self.trans(Image.open(px))
                nx = self.trans(Image.open(nx))
            yield {"anchor":  {"image": ax,
                                "geometry": ag,
                                "y": ay,
                                "dis_to_a": 0,
                                "image_id": self._get_image_id(item)},
                    "positive": {"image": px,
                                 "geometry": pg,
                                 "y": py,
                                 "dis_to_anchor": _pd,
                                 "image_id": self._get_image_id(p_item)},
                    "negative": {"image": nx,
                                 "geometry": ng,
                                 "y": ny,
                                 "dis_to_anchor": _nd,
                                 "image_id": self._get_image_id(n_item)}}
         