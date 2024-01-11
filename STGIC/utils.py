import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score



def eval_perf(pred,ground):
    ct_rmnan=[]
    y_pred_clean=[]
    for j in range(len(ground)):
        if ground[j] != -1:
            ct_rmnan.append(ground[j])
            y_pred_clean.append(pred[j])
    ari = adjusted_rand_score(y_pred_clean,np.array(ct_rmnan))
    nmi=normalized_mutual_info_score(y_pred_clean,np.array(ct_rmnan))
    return nmi,ari

def set_seed(seed):
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