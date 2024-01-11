import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from scanpy import AnnData


os.chdir('/home/zhangchen/stereoseq')
adata=sc.read('filtered_feature_bc_matrix.h5ad')
coord_x,coord_y=adata.obsm['spatial'][:,0],adata.obsm['spatial'][:,1]
expr=adata.X
gene_names=adata.var_names
gene_num=len(gene_names)
df_pool=[]
for spot in range(expr.shape[0]):
    tmp_df=pd.DataFrame({'geneID':gene_names,'x':pd.Series(np.array([coord_x[spot]]*gene_num)),'y':pd.Series(np.array([coord_y[spot]]*gene_num)),'UMICount':pd.Series(expr[spot,:])})
    df_pool.append(tmp_df)

data=pd.concat(df_pool,0)
data=data[data.UMICount!=0]

x, y = data['x'].astype("int64"), data['y'].astype("int64")
binsize=30
x_rou = np.round(x/ binsize).astype('int32')
y_rou = np.round(y/ binsize).astype('int32')
x_rou = x_rou.astype(str)
y_rou = y_rou.astype(str)
data['barcode'] = x_rou.map(str) +"-" + y_rou

count = data.groupby(['barcode', 'geneID'])['UMICount'].sum()
count = count[count != 0]
count = count.to_frame()

exonic_index = count.index.values
cell_barcode_index = list(zip(*exonic_index))[0]
gene_index = list(zip(*exonic_index))[1]

count['barcode'] = cell_barcode_index
count['gene'] = gene_index

uniq_cell, uniq_gene = count.barcode.unique(), count.gene.unique()
uniq_cell, uniq_gene = list(uniq_cell), list(uniq_gene)
cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

count["csr_x_ind"] = count["barcode"].map(cell_dict)
count["csr_y_ind"] = count["gene"].map(gene_dict)


csr_mat = csr_matrix((count['UMICount'], (count["csr_x_ind"], count["csr_y_ind"])), shape=((len(uniq_cell), len(uniq_gene))))
var = pd.DataFrame({"gene_short_name": uniq_gene})
var.set_index("gene_short_name", inplace=True)

obs = pd.DataFrame({"cell_name": count['barcode'].unique().tolist()})
obs["array_row"] = pd.Series(obs['cell_name']).str.split('-', expand=True).iloc[:, 0]
obs["array_col"] = pd.Series(obs['cell_name']).str.split('-', expand=True).iloc[:, 1]
obsm = {"spatial": obs.loc[:, ["array_row", "array_col"]].values}
obs.set_index("cell_name", inplace=True)

adata_new = AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())
adata_new.layers['count'] = csr_mat.copy()

adata_new.obsm['spatial'] = adata_new.obsm['spatial'].astype("int64")
adata_new.write_h5ad('stereo_seq_bin%s.h5ad' %binsize)


