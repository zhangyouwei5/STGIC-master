import os
from STGIC.utils import set_seed
from STGIC.preprocess import *
from STGIC.AGC import *
from STGIC.DCF import *
from STGIC.svg import *
import seaborn as sns

n_components_agc=50
n_components=15
use_high_agc=False
use_high=True
agc_dims=26

nChannel=100
output_dim=100
pretrain_lr=0.05
lr=0.01
step_kl=0.78
step_con=0.62
step_con1=0.58
step_ce=0.71
q_cut=0.5
mod3_ratio=0.7
seed=0
device=torch.device("cuda:0")

os.chdir('/home/zhangchen/validate')
pid='breast_cancer'
h5ad_file = '%s.h5ad' % pid

img,coord_lattice,mask,adj,feat_agc,adata=generate_input_visium(h5ad_file,n_components_agc,n_components,use_high_agc,use_high)
bcl=pd.read_table('/home/zhangchen/validate/metadata.tsv')
bcl.index=bcl.ID
obs1=pd.concat([adata.obs,bcl],1)
bll=obs1.fine_annot_type.values
bl=obs1.fine_annot_type.unique()
n_clusters=len(list(bl))
bl_dic={bl[i]:i for i in range(n_clusters)}
label=[]
for i in bll:
    label.append(bl_dic[i])

adj_normalized = normalize_adj(adj)
adj_normalized = 1/3*torch.eye(adj_normalized.shape[0]).to(device)+2/3*adj_normalized
optimal_power,y_pred_init,kmeans_feature,pre_nmi,pre_ari=pre_cluster(adj_normalized,feat_agc,n_clusters,label,top_num=agc_dims,max_iter=30)


coord_x=torch.tensor(coord_lattice[:,0]).to(torch.long).to(device)
coord_y=torch.tensor(coord_lattice[:,1]).to(torch.long).to(device)
set_seed(0)
model3 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,3,2).to(device)
model2 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,2,2).to(device)
nmi,ari,y_pred_last,emb,final_epoch_idx=dilate2_train5_visium(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,img,mask,y_pred_init,200,400,tol=1e-3,update_interval=4,q_cut=q_cut,mod3_ratio=mod3_ratio)


adata.obs['y_pred']=[str(int(i)) for i in y_pred_last]
adata.obs['y_pred_init']=[str(int(i)) for i in y_pred_init]

adata.obs['ground_label']=bll


adata.obs['x_pixel']=list(adata.obsm['spatial'][:,0])
adata.obs['y_pixel']=list(-adata.obsm['spatial'][:,1])
sc.pl.spatial(adata,color='y_pred')
sc.pl.spatial(adata,color='ground_label')
print('ari on breast cancer 20 clusters',ari)
print('pre-ari on breast cancer 20 clusters',pre_ari)





adata.obsm['rep']=emb
sc.pp.neighbors(adata,use_rep='rep',random_state=0)
sc.tl.umap(adata,random_state=0)
sc.pl.umap(adata, color=['y_pred'])




sc.tl.rank_genes_groups(adata, 'y_pred', method='wilcoxon')
for i in ['3','4','7','8','14','16']:
    a=sc.get.rank_genes_groups_df(adata,i)
    b=a[(a.pvals_adj<0.01)&(a.logfoldchanges>2)]
    b.sort_values(by='scores',inplace=True,ascending=False)
    sc.pl.spatial(adata, img_key="hires", color=list(b.names)[:17])



ground_moran=screen_svg('ground_label',adata)
stgic_moran=screen_svg('y_pred',adata)
print('ground truth moran statistics mean and median', np.mean(ground_moran),np.median(stgic_moran))
print('STGIC moran statistics mean and median', np.mean(stgic_moran),np.median(stgic_moran))