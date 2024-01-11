import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import random
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle

def calculate_p(adj, l):
    adj_exp = np.exp(-1 * (adj ** 2) / (2 * (l ** 2)))
    return np.mean(np.sum(adj_exp, 1)) - 1

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


def prefilter_specialgenes(adata,Gene2Pattern, Gene1Pattern="ERCC"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)

def search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100):
    run = 0
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        print("l not found, try smaller start point.")
        return None
    elif p_high < p - tol:
        print("l not found, try bigger end point.")
        return None
    elif np.abs(p_low - p) <= tol:
        print("recommended l = ", str(start))
        return start
    elif np.abs(p_high - p) <= tol:
        print("recommended l = ", str(end))
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        print("Run " + str(run) + ": l [" + str(start) + ", " + str(end) + "], p [" + str(p_low) + ", " + str(
            p_high) + "]")
        if run > max_run:
            print("Exact l not found, closest values are:\n" + "l=" + str(start) + ": " + "p=" + str(
                p_low) + "\nl=" + str(end) + ": " + "p=" + str(p_high))
            return None
        mid = (start + end) / 2
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            print("recommended l = ", str(mid))
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid


def eu_np_dis(data_src,data_dst):
    src_num=data_src.shape[0]
    dst_num=data_dst.shape[0]
    data_src=np.expand_dims(data_src,1)
    data_src=np.tile(data_src,(1,dst_num,1))
    data_dst=np.expand_dims(data_dst,0)
    data_dst=np.tile(data_dst,(src_num,1,1))
    dist=np.sqrt(np.power(data_src-data_dst,2).sum(-1))
    return dist


def generate_input_visium(h5ad_file,n_components_agc=50,n_components=15,use_high_agc=False,use_high=True,Gene2Pattern='MT-'):
    adata = sc.read(h5ad_file)

    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    prefilter_specialgenes(adata,Gene2Pattern)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    coord_lattice=np.int32(adata.obs.loc[:,['array_row','array_col']].values)
    coord_pixel=adata.obsm['spatial']
    distance_2d=eu_np_dis(coord_pixel,coord_pixel)
    l=search_l(0.5,distance_2d)
    adj=np.exp(-1*(distance_2d**2)/(2*(l**2)))

    img_h,img_w=list(coord_lattice.max(0)+1)
    mean_vector=adata.X.A.mean(0)
    sc.pp.pca(adata,n_comps=n_components_agc,use_highly_variable=use_high_agc)
    feat_agc= adata.obsm['X_pca']
    sc.pp.pca(adata,n_comps=n_components,use_highly_variable=use_high)
    padding_vector=-mean_vector.dot(adata.varm['PCs'])


    feat = adata.obsm['X_pca']

    img=np.zeros([img_h,img_w,n_components])

    img[coord_lattice[:,0],coord_lattice[:,1]]=feat
    for j in zip(*[list(i) for i in np.where(img.sum(-1)==0)]):
        img[j[0],j[1],:]=padding_vector


    img=(img-img.min())/(img.max()-img.min())
    mask=np.zeros([img_h,img_w])
    mask[coord_lattice[:,0],coord_lattice[:,1]]=1

    return img,coord_lattice,mask,adj,feat_agc,adata




def generate_input_stereoseq(h5ad_file,n_components_agc=50,n_components=15,use_high_agc=False,use_high=True,Gene2Pattern='mt-',min_gene_num=20,min_cell_num=3):
    adata = sc.read(h5ad_file)
    

    adata.var_names_make_unique()
    #sc.pp.filter_genes(adata, min_cells=3)
    prefilter_genes(adata, min_cells=min_cell_num)  # avoiding all genes are zeros
    prefilter_specialgenes(adata,Gene2Pattern)
    sc.pp.filter_cells(adata, min_counts=min_gene_num)
    #Normalize and take log for UMI
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    #sc.pp.scale(adata)
    #n_clusters=len(adata.obs['annotation'].unique())

    coord_lattice=np.int32(adata.obsm['spatial'])
    coord_lattice-=coord_lattice.min(0)
    coord_pixel=np.int32(adata.obsm['spatial'])
    coord_pixel-=coord_pixel.min(0)
    # distance_2d=eu_dis(coord_pixel)
    distance_2d=eu_np_dis(coord_pixel,coord_pixel)
    l=search_l(0.5,distance_2d)
    adj=np.exp(-1*(distance_2d**2)/(2*(l**2)))

    img_h,img_w=list(coord_lattice.max(0)+1)
    #mean_vector=adata.X.A.mean(0)
    mean_vector=adata.X.mean(0)
    sc.pp.pca(adata,n_comps=n_components_agc,use_highly_variable=use_high_agc)
    feat_agc= adata.obsm['X_pca']
    sc.pp.pca(adata,n_comps=n_components,use_highly_variable=use_high)
    padding_vector=-mean_vector.dot(adata.varm['PCs'])

    feat = adata.obsm['X_pca']

    img=np.zeros([img_h,img_w,n_components])

    img[coord_lattice[:,0],coord_lattice[:,1]]=feat
    for j in zip(*[list(i) for i in np.where(img.sum(-1)==0)]):
        img[j[0],j[1],:]=padding_vector


    img=(img-img.min())/(img.max()-img.min())
    mask=np.zeros([img_h,img_w])
    mask[coord_lattice[:,0],coord_lattice[:,1]]=1

    return img,coord_lattice,mask,adj,feat_agc,adata
