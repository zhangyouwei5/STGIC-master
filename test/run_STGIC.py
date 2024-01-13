import os
from STGIC.utils import set_seed
from STGIC.preprocess import *
from STGIC.AGC import *
from STGIC.DCF import *


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

pid_list=['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']


ari_list=[]

pre_ari_list=[]

for pid in pid_list:

    h5ad_file = f'/data/DLPFC/{pid}.h5ad'
    img,coord_lattice,mask,adj,feat_agc,adata=generate_input_visium(h5ad_file,n_components_agc,n_components,use_high_agc,use_high)

    label=[]
    ground_label=list(adata.obs.celltype)
    for i in ground_label:
        if i!='nan':
            label.append(i)
        else:
            label.append(-1)
    if -1 in label:
        n_clusters = len(set(label)) - 1
    else:
        n_clusters = len(set(label))


    adj_normalized = normalize_adj(adj)
    adj_normalized = 1/3*torch.eye(adj_normalized.shape[0]).to(device)+2/3*adj_normalized
    optimal_power,y_pred_init,kmeans_feature,pre_nmi,pre_ari=pre_cluster(adj_normalized,feat_agc,n_clusters,label,top_num=agc_dims,max_iter=30)
    pre_ari_list.append(pre_ari)

    coord_x=torch.tensor(coord_lattice[:,0]).to(torch.long).to(device)
    coord_y=torch.tensor(coord_lattice[:,1]).to(torch.long).to(device)
    set_seed(0)
    model3 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,3,2).to(device)
    model2 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,2,2).to(device)
    nmi,ari,y_pred_last,emb,final_epoch_idx=dilate2_train5_visium(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,img,mask,y_pred_init,200,400,tol=1e-3,update_interval=4,q_cut=q_cut,mod3_ratio=mod3_ratio)
    ari_list.append(ari)

    adata.obs['y_pred']=[str(int(i)) for i in y_pred_last]
    adata.obs['y_pred_init']=[str(int(i)) for i in y_pred_init]

    adata.obs['ground_label']=[i if i!=-1 else '' for i in label]


    adata.obs['x_pixel']=list(adata.obsm['spatial'][:,0])
    adata.obs['y_pixel']=list(-adata.obsm['spatial'][:,1])

    sc.pl.spatial(adata,color='y_pred_init', save=f"{pid}_y_pred_init_domains.pdf")
    sc.pl.spatial(adata,color='y_pred', save=f"{pid}_y_pred_domains.pdf")
    sc.pl.spatial(adata,color='ground_label', save=f"{pid}_y_ground_label_domains.pdf")



    print(pid,'\n\n')
print(np.array(ari_list).mean(),np.median(np.array(ari_list)))

print(np.array(pre_ari_list).mean(),np.median(np.array(pre_ari_list)))