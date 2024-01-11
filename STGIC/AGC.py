import torch
import numpy as np
from sklearn.cluster import KMeans
from STGIC.utils import eval_perf

def normalize_adj(adj,device=torch.device('cuda:0')):
    adj_hat=torch.Tensor(adj).to(device)
    asum=adj_hat.sum(-1)
    D_neg05=torch.pow(asum+1e-10,-0.5)
    D_neg05=torch.diag_embed(D_neg05).squeeze(axis=0)
    norm_a=torch.matmul(D_neg05,adj_hat)
    norm_a=torch.matmul(norm_a,D_neg05)
    return norm_a

def to_onehot(prelabel):
    k = len(np.unique(prelabel))
    label = np.zeros([prelabel.shape[0], k])
    label[range(prelabel.shape[0]), prelabel] = 1
    label = label.T
    return label

def square_dist(prelabel, feature):
    feature = np.array(feature)
    onehot = to_onehot(prelabel)
    m, n = onehot.shape
    count = onehot.sum(1).reshape(m, 1)
    count[count==0] = 1
    mean = onehot.dot(feature)/count
    a2 = (onehot.dot(feature*feature)/count).sum(1)
    pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))
    intra_dist = pdist2.trace()
    inter_dist = pdist2.sum() - intra_dist
    intra_dist /= m
    inter_dist /= m * (m - 1)
    return intra_dist


def pre_cluster(adj_normalized,feature,n_clusters,label,top_num=10,rep=[0,42,20],max_iter=10,random_seed=True,device=torch.device('cuda:0'),platform='visium'):
    if platform=='visium':
        feature=torch.Tensor(feature).to(device)
    if platform=='stereoseq':
        feature=torch.Tensor(feature.copy()).to(device)    
    tt = 0
    intra_list = []
    intra_list.append(10000)
    u_pool=[0,1]
    pred_pool=[]

    while 1:
        tt = tt + 1
        intraD = np.zeros(len(rep))
        feature = torch.mm(adj_normalized,feature)
        u, s, v = torch.svd(feature,some=False)
        u=u[:,:top_num]
        if tt==1:
            u_pool[0]=u.data.cpu().numpy()

        elif tt==2:
            u_pool[1]=u.data.cpu().numpy()

        else:
            u_pool=[u_pool[1]]+[u.data.cpu().numpy()]

        tmp_pred_pool=[]

        for i in range(len(rep)):
            if random_seed:
                kmeans = KMeans(n_clusters=n_clusters,random_state=rep[i])
            else:
                kmeans = KMeans(n_clusters=n_clusters)
            predict_labels = kmeans.fit_predict(u.data.cpu().numpy())
            tmp_pred_pool.append(predict_labels)
            intraD[i] = square_dist(predict_labels, feature.data.cpu().numpy())

        intramean = np.mean(intraD)
        intra_list.append(intramean)
        pred_pool.append(tmp_pred_pool[np.argmin(intraD)])
        if intra_list[tt] > intra_list[tt - 1] or tt > max_iter:
            optimal_power=tt-1
            if label is not None:
                nmi,ari=eval_perf(pred_pool[-2],label)
                
            else:
                nmi,ari=None,None
            return optimal_power,pred_pool[-2],u_pool[0],nmi,ari        
                        



