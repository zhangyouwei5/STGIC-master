import os
import torch
import numpy as np
import pandas as pd
import random
import sys
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import subprocess
from STGIC.svg import *


seed = 42
n_domains = 20
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():  # GPU operation have separate seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


data_path = "../data/"
data_name = 'V1_Breast_Cancer_Block_A_Section_1'
save_path = "../Results"

deepen = run(save_path = save_path,
    task = "Identify_Domain", #### DeepST includes two tasks, one is "Identify_Domain" and the other is "Integration"
    pre_epochs = 800, ####  choose the number of training
    epochs = 1000, #### choose the number of training
    use_gpu = True)

###### Read in 10x Visium data, or user can read in themselves.
adata_ = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
###### Segment the Morphological Image
adata_ = deepen._get_image_crop(adata_, data_name=data_name)

###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
###### is only applicable to 10x visium and the remaining omics selects the other two.
###### "use_morphological" defines whether to use morphological images.
adata_ = deepen._get_augment(adata_, spatial_type="LinearRegress", use_morphological=True)

###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
graph_dict = deepen._get_graph(adata_.obsm["spatial"], distType = "BallTree")

###### Enhanced data preprocessing
data = deepen._data_process(adata_, pca_n_comps = 200)

###### Training models
deepst_embed = deepen._fit(
        data = data,
        graph_dict = graph_dict,)
###### DeepST outputs
adata_.obsm["DeepST_embed"] = deepst_embed

###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
adata = deepen._get_cluster_data(adata_, n_domains=n_domains, priori = True).copy()
###### Spatial localization map of the spatial domain
sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, save="domains.pdf")
sc.pp.neighbors(adata, use_rep='DeepST_embed')
sc.tl.umap(adata)
sc.pl.umap(adata, color=["DeepST_refine_domain"], save="umap.pdf")

# moran
def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata, Gene2Pattern, Gene1Pattern="ERCC"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)


moran = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)
moran.var_names_make_unique()
prefilter_genes(moran, min_cells=3)  # avoiding all genes are zeros
prefilter_specialgenes(moran, 'MT-')
sc.pp.normalize_per_cell(moran)
sc.pp.log1p(moran)
moran.obs["DeepST_refine_domain"] = adata.obs["DeepST_refine_domain"]

stgic_moran = screen_svg('DeepST_refine_domain', moran)
print('moran statistics mean and median', np.mean(stgic_moran),np.median(stgic_moran))
