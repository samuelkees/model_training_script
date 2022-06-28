from glob import glob
import pandas as pd 
import numpy as np
import pickle 
import torch
import faiss 
import os

from siuba import _, mutate, filter as _filter, group_by, summarize
from siuba.siu import symbolic_dispatch

from indexmodel import SSIndex

@symbolic_dispatch
def n(x):
    return len(x)

def flatten(x, depth=0):
    """Flattens a list for lists."""
    if depth == 0:
        return [item for subl in x for item in subl]
    return flatten([item for subl in x for item in subl], depth-1)


def load_dataset_meta(dir_path):
    meta_df = pd.read_pickle(glob(dir_path + "/*.pickle")[0])
    data_df = pd.concat(pd.read_pickle(x) for x in glob(dir_path + "/*/*.pickle"))
    
    return pd.merge(data_df, meta_df, on="class_id", how="left")


def load_index(root_dataset_path, vector_path, dim, columns):

    dataset_df = (load_dataset_meta(root_dataset_path)
                    >> mutate(image_path= _.class_id + "/" + _.image_id))
    # load tensors 
    dataset_tensor = torch.cat(pickle.load(open(vector_path, "rb"))).view(-1, dim)

    _index = faiss.IndexFlatL2(dim)
    _tensors = dataset_tensor.cpu().numpy()
    _index.add(_tensors)

    return SSIndex(_index, dataset_df, columns)


def sample_index_by_colum(_index, n=15, class_colum="class_id", dim=512):

    df = _index.meta.reset_index(drop=True)
    new_meta_df = (df 
        >> _filter(_.class_id.isin(_get_classes_up_to(df, n)))
        >> group_by(class_colum)).sample(n)
    
    new_vectors = np.asarray([_index.index.reconstruct(x) for x in new_meta_df.index])
    new_index = faiss.IndexFlatL2(dim)
    new_index.add(new_vectors)

    rest_meta_df = df[~df.index.isin(new_meta_df.index)]
    rest_vectors = np.asarray([_index.index.reconstruct(x) for x in rest_meta_df.index])
    rest_index = faiss.IndexFlatL2(dim)
    rest_index.add(rest_vectors)

    return (SSIndex(new_index, new_meta_df.reset_index(drop=True), class_colum),
            SSIndex(rest_index, rest_meta_df.reset_index(drop=True), class_colum))


def _get_classes_up_to(df, _n, class_col="class_id"):
    return ( df 
            >> group_by(class_col) 
            >> summarize(n=n(_.class_id))
            >> _filter(_.n > _n)
            )[class_col].values


def gen_ss_index(array, meta, mapping):
    _, d = array.shape
    ss_index = faiss.IndexFlatL2(d)
    ss_index.add(array)
    return SSIndex(ss_index, meta, mapping)


def save_ss_index(meta: pd.DataFrame, tensor: torch.Tensor, path):
    os.makedirs(path, exist_ok=True)
    meta.to_parquet(f"{path}/meta.parquet")
    torch.save(tensor.cpu(), f"{path}/tensor.pt")


def load_ss_index(path, mapping):
    meta = pd.read_parquet(glob(f"{path}/*.parquet")[0])
    tensor = torch.load(glob(f"{path}/*.pt")[0])
    _, d = tensor.shape
    index = faiss.IndexFlatL2(d)
    index.add(tensor.cpu().numpy())
    
    return SSIndex(index, meta, mapping)