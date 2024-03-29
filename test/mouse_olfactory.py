import os
from STGIC.utils import set_seed
from STGIC.preprocess import *
from STGIC.AGC import *
from STGIC.DCF import *
import seaborn as sns


os.chdir('/home/zhangchen/stereoseq')
n_components_agc=50
n_components=15
use_high_agc=False
use_high=True
#agc_dims=30

nChannel=100
output_dim=100
agc_dims=26
pretrain_lr=0.005
lr=0.001

step_kl=0.78
step_con=0.62
step_con1=0.
step_ce=0.71
q_cut=0.5

mod3_ratio=0.9

seed=0
device=torch.device("cuda:0")

h5ad_file='stereo_seq_bin30.h5ad'
img,coord_lattice,mask,adj,feat_agc,adata=generate_input_stereoseq(h5ad_file,n_components_agc,n_components,use_high_agc,use_high)
n_clusters=8
label=None
adj_normalized = normalize_adj(adj)
adj_normalized = 1/3*torch.eye(adj_normalized.shape[0]).to(device)+2/3*adj_normalized
optimal_power,y_pred_init,kmeans_feature,pre_nmi,pre_ari=pre_cluster(adj_normalized,feat_agc,n_clusters,label,top_num=agc_dims,max_iter=30,platform='stereoseq')


adata.obs['y_pred_init']=[str(i) for i in y_pred_init]
adata.obsm['spatial']=np.stack([adata.obsm['spatial'][:,0],-adata.obsm['spatial'][:,1]],-1)
adata.obs['x']=adata.obsm['spatial'][:,0]
adata.obs['y']=adata.obsm['spatial'][:,1]
coord_x=torch.tensor(coord_lattice[:,0]).to(torch.long).to(device)
coord_y=torch.tensor(coord_lattice[:,1]).to(torch.long).to(device)
set_seed(0)
model3 = Dilate_MyNet5_stereoseq(img.shape[2],output_dim,nChannel,n_clusters,3,1).to(device)
model2 = Dilate_MyNet5_stereoseq(img.shape[2],output_dim,nChannel,n_clusters,3,2).to(device)
nmi,ari,y_pred_last,emb,final_epoch_idx=dilate2_train5_stereoseq(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,img,mask,y_pred_init,200,400,tol=1e-3,update_interval=4,q_cut=q_cut,mod3_ratio=mod3_ratio)
adata.obs['y_pred_last']=[str(i) for i in y_pred_last]
sc.pl.spatial(adata,color='y_pred_last',spot_size=25000/adata.shape[0])

adata.obsm['rep']=emb
sc.pp.neighbors(adata,use_rep='rep',random_state=0)
sc.tl.umap(adata,random_state=0)
sc.pl.umap(adata, color=['y_pred_last'])

sc.tl.rank_genes_groups(adata, 'y_pred_last', method='wilcoxon')


for i in ['6','1','7','0','5','2']:
    a=sc.get.rank_genes_groups_df(adata,i)
    b=a[(a.pvals_adj<0.01)&(a.logfoldchanges>0.7)]
    b.sort_values(by='scores',inplace=True,ascending=False)
    sc.pl.spatial(adata, img_key="hires", color=list(b.names)[:5],spot_size=25000/adata.shape[0])









h5ad_file='stereo_seq_bin15.h5ad'
img,coord_lattice,mask,adj,feat_agc,adata=generate_input_stereoseq(h5ad_file,n_components_agc,n_components,use_high_agc,use_high)
n_clusters=9
label=None
adj_normalized = normalize_adj(adj)
adj_normalized = 1/3*torch.eye(adj_normalized.shape[0]).to(device)+2/3*adj_normalized
optimal_power,y_pred_init,kmeans_feature,pre_nmi,pre_ari=pre_cluster(adj_normalized,feat_agc,n_clusters,label,top_num=agc_dims,max_iter=30,platform='stereoseq')


adata.obs['y_pred_init']=[str(i) for i in y_pred_init]
adata.obsm['spatial']=np.stack([adata.obsm['spatial'][:,0],-adata.obsm['spatial'][:,1]],-1)
adata.obs['x']=adata.obsm['spatial'][:,0]
adata.obs['y']=adata.obsm['spatial'][:,1]
coord_x=torch.tensor(coord_lattice[:,0]).to(torch.long).to(device)
coord_y=torch.tensor(coord_lattice[:,1]).to(torch.long).to(device)
set_seed(0)
model3 = Dilate_MyNet5_stereoseq(img.shape[2],output_dim,nChannel,n_clusters,3,1).to(device)
model2 = Dilate_MyNet5_stereoseq(img.shape[2],output_dim,nChannel,n_clusters,3,2).to(device)
nmi,ari,y_pred_last,emb,final_epoch_idx=dilate2_train5_stereoseq(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,img,mask,y_pred_init,200,400,tol=1e-3,update_interval=4,q_cut=q_cut,mod3_ratio=mod3_ratio)
adata.obs['y_pred_last']=[str(i) for i in y_pred_last]
sc.pl.spatial(adata,color='y_pred_last',spot_size=100000/adata.shape[0])

adata.obsm['rep']=emb
sc.pp.neighbors(adata,use_rep='rep',random_state=0)
sc.tl.umap(adata,random_state=0)
sc.pl.umap(adata, color=['y_pred_last'])

sc.tl.rank_genes_groups(adata, 'y_pred_last', method='wilcoxon')
for i in ['4','1','6','7','3','2','5']:
    a=sc.get.rank_genes_groups_df(adata,i)
    b=a[(a.pvals_adj<0.01)&(a.logfoldchanges>1)]
    b.sort_values(by='scores',inplace=True,ascending=False)
    sc.pl.spatial(adata, img_key="hires", color=list(b.names)[:7],spot_size=100000/adata.shape[0])


