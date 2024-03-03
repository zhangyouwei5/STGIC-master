import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
from scipy.sparse import issparse
import random
import numba


def count_nbr(target_cluster,cell_id, x, y, pred, radius):
    adj_2d=calculate_adj_matrix(x=x,y=y, histology=False)
    cluster_num = dict()
    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x=row["x"]
        y=row["y"]
        tmp_nbr=df[((df["x"]-x)**2+(df["y"]-y)**2)<=(radius**2)]
        num_nbr.append(tmp_nbr.shape[0])
    return np.mean(num_nbr)


@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(n):
			adj[i][j]=euclid_dist(X[i], X[j])
	return adj



def search_radius(target_cluster,cell_id, x, y, pred, start, end, num_min=8, num_max=15,  max_run=100):
    run=0
    num_low=count_nbr(target_cluster,cell_id, x, y, pred, start)
    num_high=count_nbr(target_cluster,cell_id, x, y, pred, end)
    if num_min<=num_low<=num_max:
        print("recommended radius = ", str(start))
        return start
    elif num_min<=num_high<=num_max:
        print("recommended radius = ", str(end))
        return end
    elif num_low>num_max:
        print("Try smaller start.")
        return None
    elif num_high<num_min:
        print("Try bigger end.")
        return None
    while (num_low<num_min) and (num_high>num_min):
        run+=1
        print("Run "+str(run)+": radius ["+str(start)+", "+str(end)+"], num_nbr ["+str(num_low)+", "+str(num_high)+"]")
        if run >max_run:
            print("Exact radius not found, closest values are:\n"+"radius="+str(start)+": "+"num_nbr="+str(num_low)+"\nradius="+str(end)+": "+"num_nbr="+str(num_high))
            return None
        mid=(start+end)/2
        num_mid=count_nbr(target_cluster,cell_id, x, y, pred, mid)
        if num_min<=num_mid<=num_max:
            print("recommended radius = ", str(mid), "num_nbr="+str(num_mid))
            return mid
        if num_mid<num_min:
            start=mid
            num_low=num_mid
        elif num_mid>num_max:
            end=mid
            num_high=num_mid





def find_neighbor_clusters(target_cluster,cell_id, x, y, pred,radius, ratio=1/2):
    cluster_num = dict()
    for i in pred:
        cluster_num[i] = cluster_num.get(i, 0) + 1
    df = {'cell_id': cell_id, 'x': x, "y":y, "pred":pred}
    df = pd.DataFrame(data=df)
    df.index=df['cell_id']
    target_df=df[df["pred"]==target_cluster]
    nbr_num={}
    row_index=0
    num_nbr=[]
    for index, row in target_df.iterrows():
        x=row["x"]
        y=row["y"]
        tmp_nbr=df[((df["x"]-x)**2+(df["y"]-y)**2)<=(radius**2)]
        #tmp_nbr=df[(df["x"]<x+radius) & (df["x"]>x-radius) & (df["y"]<y+radius) & (df["y"]>y-radius)]
        num_nbr.append(tmp_nbr.shape[0])
        for p in tmp_nbr["pred"]:
            nbr_num[p]=nbr_num.get(p,0)+1
    del nbr_num[target_cluster]
    nbr_num_back=nbr_num.copy() #Backup
    nbr_num=[(k, v)  for k, v in nbr_num.items() if v>(ratio*cluster_num[k])]
    nbr_num.sort(key=lambda x: -x[1])
    print("radius=", radius, "average number of neighbors for each spot is", np.mean(num_nbr))
    print(" Cluster",target_cluster, "has neighbors:")
    for t in nbr_num:
        print("Dmain ", t[0], ": ",t[1])
    ret=[t[0] for t in nbr_num]
    if len(ret)==0:
        nbr_num_back=[(k, v)  for k, v in nbr_num_back.items()]
        nbr_num_back.sort(key=lambda x: -x[1])
        ret=[nbr_num_back[0][0]]
        print("No neighbor domain found, only return one potential neighbor domain:",ret)
        print("Try bigger radius or smaller ratio.")
    return ret



def rank_genes_groups(input_adata, target_cluster,nbr_list, label_col, adj_nbr=True, log=False):
    if adj_nbr:
        nbr_list=nbr_list+[target_cluster]
        adata=input_adata[input_adata.obs[label_col].isin(nbr_list)]
    else:
        adata=input_adata.copy()
    adata.var_names_make_unique()
    adata.obs["target"]=((adata.obs[label_col]==target_cluster)*1).astype('category')
    sc.tl.rank_genes_groups(adata, groupby="target",reference="rest", n_genes=adata.shape[1],method='wilcoxon')
    pvals_adj=[i[0] for i in adata.uns['rank_genes_groups']["pvals_adj"]]
    genes=[i[1] for i in adata.uns['rank_genes_groups']["names"]]
    if issparse(adata.X):
        obs_tidy=pd.DataFrame(adata.X.A)
    else:
        obs_tidy=pd.DataFrame(adata.X)
    obs_tidy.index=adata.obs["target"].tolist()
    obs_tidy.columns=adata.var.index.tolist()
    obs_tidy=obs_tidy.loc[:,genes]
    # 1. compute mean value
    mean_obs = obs_tidy.groupby(level=0).mean()
    # 2. compute fraction of cells having value >0
    obs_bool = obs_tidy.astype(bool)
    fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()
    # compute fold change.
    if log: #The adata already logged
        fold_change=np.exp((mean_obs.loc[1] - mean_obs.loc[0]).values)
    else:
        fold_change = (mean_obs.loc[1] / (mean_obs.loc[0]+ 1e-9)).values
    df = {'genes': genes, 'in_group_fraction': fraction_obs.loc[1].tolist(), "out_group_fraction":fraction_obs.loc[0].tolist(),"in_out_group_ratio":(fraction_obs.loc[1]/fraction_obs.loc[0]).tolist(),"in_group_mean_exp": mean_obs.loc[1].tolist(), "out_group_mean_exp": mean_obs.loc[0].tolist(),"fold_change":fold_change.tolist(), "pvals_adj":pvals_adj}
    df = pd.DataFrame(data=df)
    return df


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
	#x,y,x_pixel, y_pixel are lists
	if histology:
		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
		print("Calculateing adj matrix using histology image...")
		#beta to control the range of neighbourhood when calculate grey vale for one spot
		#alpha to control the color scale
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
		c0, c1, c2=[], [], []
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
		c4=(c3-np.mean(c3))/np.std(c3)
		z_scale=np.max([np.std(x), np.std(y)])*alpha
		z=c4*z_scale
		z=z.tolist()
		print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
		X=np.array([x, y, z]).T.astype(np.float32)
	else:
		print("Calculateing adj matrix using xy only...")
		X=np.array([x, y]).T.astype(np.float32)
	return pairwise_distance(X)


#identify SVGs in reference to SpaGCN
def screen_svg(label_name,adata,gene_num=200,nei_num=3,min_in_group_fraction=0.8,min_in_out_group_ratio=1,min_fold_change=1.5,p_cut=0.05): 
    svg_pool=[]
    svg_dict={}
    unique_label=adata.obs[label_name].unique()
    for i in unique_label:
        target=str(i)
        x_array=adata.obs['array_col']
        y_array=adata.obs['array_row']
    #Search radius such that each spot in the target domain has approximately 10 neighbors on average
        adj_2d=calculate_adj_matrix(x=x_array, y=y_array, histology=False)
        start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
        r=search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs[label_name].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
    #Detect neighboring domains
        nbr_domians=find_neighbor_clusters(target_cluster=target,
                                   cell_id=adata.obs.index.tolist(),
                                   x=adata.obs["array_col"].tolist(),
                                   y=adata.obs["array_row"].tolist(),
                                   pred=adata.obs[label_name].tolist(),
                                   radius=r,
                                   ratio=1/2)
        nbr_domians=nbr_domians[0:nei_num]
        de_genes_info=rank_genes_groups(input_adata=adata,
                                target_cluster=target,
                                nbr_list=nbr_domians,
                                label_col=label_name,
                                adj_nbr=True,
                                log=True)

        de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<p_cut)]
        filtered_info=de_genes_info
        filtered_info=filtered_info[(filtered_info["pvals_adj"]<p_cut) &
                            (filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
                            (filtered_info["in_group_fraction"]>min_in_group_fraction) &
                            (filtered_info["fold_change"]>min_fold_change)]
        filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
        filtered_info["target_dmain"]=target
        filtered_info["neighbors"]=str(nbr_domians)
        svg_pool.append(filtered_info)
        
    svg_pd=pd.concat(svg_pool,axis=0)
    svg_pool_list=list((svg_pd.sort_values(by='fold_change',ascending=False).genes))
    svg_list=[]
    count=0
    for n in svg_pool_list:
        if n not in svg_list:
            svg_list.append(n)
            count+=1
        if count>=gene_num:
            break   
    sc.pp.neighbors(adata,use_rep="spatial")
    svg_idx=[list(adata.var_names).index(i) for i in svg_list]
    moran = sc.metrics.morans_i(adata.obsp["connectivities"], adata.X[:,svg_idx].T)
    print('mean and median moran I are', np.mean(moran),np.median(moran))
    return moran
