import os,csv,re
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from loss import target_distribution, kl_loss
import torch.optim as optim
from torch.nn.parameter import Parameter
from anndata import AnnData
import torch
from sklearn.cluster import KMeans
from util import *
import torch.nn as nn
import argparse
from sklearn.decomposition import PCA
from models import *
from types import SimpleNamespace
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from STGIC.svg import *


dataset = "Human breast cancer"
opts = {
    "no-cuda": False,
    "lr": 0.001,
    "nhid1": 32,
    "n_cluster": 20,
    "max_epochs": 2000,
    "update_interval": 3,
    "seed": 45,
    "weight_decay": 0.001,
    "dataset": dataset,
    "sicle": "151673",
    "tol": 0.0001,
    "l": 1,
    "npca": 50,
    "n_neighbors": 10,
    "initcluster": "kmeans",
    "cuda": True,
    "device": "cuda"
}

opts = SimpleNamespace(**opts)
np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
if opts.cuda:
    torch.cuda.manual_seed_all(opts.seed)
    torch.cuda.manual_seed(opts.seed)

features_adata, features, labels = load_data(opts.dataset, opts.sicle, opts.npca)
adj1,adj2 = load_graph(opts.dataset,opts.sicle,opts.l)
model =STMGCN(nfeat=features.shape[1], nhid1=opts.nhid1, nclass=opts.n_cluster)
if opts.cuda:
    model.cuda()
    features = features.cuda()
    adj1 = adj1.cuda()
    adj2 = adj2.cuda()
optimizer = optim.Adam(model.parameters(),lr=opts.lr, weight_decay=opts.weight_decay)
emb = model.mgcn(features,adj1,adj2)

if opts.initcluster == "kmeans":
    print("Initializing cluster centers with kmeans, n_clusters known")
    n_clusters = opts.n_cluster
    kmeans = KMeans(n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(emb.detach().cpu().numpy())
elif opts.initcluster == "louvain":
    print("Initializing cluster centers with louvain,resolution=", opts.res)
    adata = sc.AnnData(emb.detach().cpu().numpy())
    sc.pp.neighbors(adata, n_neighbors=opts.n_neighbors)
    sc.tl.louvain(adata, resolution=opts.res)
    y_pred = adata.obs['louvain'].astype(int).to_numpy()
    n = len(np.unique(y_pred))

emb=pd.DataFrame(emb.detach().cpu().numpy(),index=np.arange(0,emb.shape[0]))
Group=pd.Series(y_pred,index=np.arange(0,emb.shape[0]),name="Group")
Mergefeature=pd.concat([emb,Group],axis=1)
cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())

y_pred_last = y_pred
with torch.no_grad():
    model.cluster_layer.copy_(torch.tensor(cluster_centers))

model.train()
for epoch in range(opts.max_epochs):

    if epoch % opts.update_interval == 0:
        _, tem_q = model(features, adj1, adj2)
        tem_q = tem_q.detach()
        p = target_distribution(tem_q)

        y_pred = torch.argmax(tem_q, dim=1).cpu().numpy()
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred
        if len(set(list(y_pred))) == n_domains and (
                True not in list(pd.Series(y_pred).value_counts().values / y_pred.shape[0] < cat_tol)):
            y_pred_last = y_pred
        else:
            break
        y = labels

        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)
        print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

        if epoch > 0 and delta_label < opts.tol:
            print('delta_label ', delta_label, '< tol ', opts.tol)
            print("Reach tolerance threshold. Stopping training.")
            break

    optimizer.zero_grad()
    x, q = model(features, adj1, adj2)
    loss = kl_loss(q.log(), p)
    loss.backward()
    optimizer.step()

#save emnddings
key_added = "STMGCN"
embeddings = pd.DataFrame(x.detach().cpu().numpy())
embeddings.index = features_adata.obs_names
features_adata.obsm[key_added] = embeddings.loc[features_adata.obs_names,].values
features_adata.obs["pred"] = [str(i) for i in y_pred_last]

#plot spatial
plt.rcParams["figure.figsize"] = (6, 3)
sc.pl.spatial(features_adata,color=["pred"], title=['STMGCN (ARI=%.3f)' % ari], save="domains.pdf")

sc.pp.neighbors(features_adata, use_rep='STMGCN')
sc.tl.umap(features_adata)
sc.pl.umap(features_adata, color=["pred"], save='umap.pdf')

stgic_moran = screen_svg('pred', features_adata)
print('moran statistics mean and median', np.mean(stgic_moran),np.median(stgic_moran))
