import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.init
from STGIC.utils import eval_perf

def compute_q(emb,center,alpha=0.2):
    q = 1.0 / ((1.0 + torch.sum((emb.unsqueeze(1) - center)**2, dim=2) / alpha) + 1e-8)
    q = q**(alpha+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

def compute_p(q):
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p

def compute_kl(p,q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
    loss = kld(p, q)
    return loss


class Dilate_MyNet5_stereoseq(nn.Module):
    def __init__(self,input_dim,output_dim,nChannel,n_clusters,kernel_size,dilate,modify=False):
        super(Dilate_MyNet5_stereoseq, self).__init__()
        if kernel_size==3:
            if dilate==1:
                padding=1
            if dilate==2:
                padding=2
        if kernel_size==2 and dilate==2:
            padding=1
        self.conv1 = nn.Conv2d(input_dim, nChannel,kernel_size=kernel_size, stride=1, padding=padding ,dilation=dilate)
        self.conv1_weight=nn.Parameter(torch.Tensor(nChannel,input_dim))
        nn.init.xavier_normal(self.conv1_weight)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.Conv2d(nChannel, nChannel, kernel_size=kernel_size, stride=1, padding=padding,dilation=dilate )
        self.conv2_weight=nn.Parameter(torch.Tensor(nChannel,nChannel))
        nn.init.xavier_normal(self.conv2_weight)
        self.bn2=nn.BatchNorm2d(nChannel)
        self.conv3 = nn.Conv2d(nChannel, output_dim, kernel_size=kernel_size, stride=1, padding=padding,dilation=dilate )
        self.conv3_weight=nn.Parameter(torch.Tensor(output_dim,nChannel))
        nn.init.xavier_normal(self.conv3_weight)
        self.bn3=nn.BatchNorm2d(output_dim)
        self.conv4 = nn.Conv2d(output_dim, n_clusters, kernel_size=1, stride=1, padding=0 )

        if dilate==1:
            self.conv1_weight_corner=nn.Parameter(torch.Tensor(nChannel,input_dim))
            self.conv1_weight_edgemid=nn.Parameter(torch.Tensor(nChannel,input_dim))
            nn.init.xavier_normal(self.conv1_weight_corner)
            nn.init.xavier_normal(self.conv1_weight_edgemid)
            self.conv2_weight_corner=nn.Parameter(torch.Tensor(nChannel,nChannel))
            self.conv2_weight_edgemid=nn.Parameter(torch.Tensor(nChannel,nChannel))
            nn.init.xavier_normal(self.conv2_weight_corner)
            nn.init.xavier_normal(self.conv2_weight_edgemid)
            self.conv3_weight_corner=nn.Parameter(torch.Tensor(output_dim,nChannel))
            self.conv3_weight_edgemid=nn.Parameter(torch.Tensor(output_dim,nChannel))
            nn.init.xavier_normal(self.conv3_weight_corner)
            nn.init.xavier_normal(self.conv3_weight_edgemid)
        self.kernel_size=kernel_size
        self.dilate=dilate
        self.modify=modify


    def modify_conv(self,conv_instance,conv_weight,conv_weight_corner=None):
        if self.kernel_size==3:
            if self.dilate==1:
                conv_instance.weight.data[:,:,0::2,0::2]=conv_weight_corner.data.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,2)

            elif self.dilate==2:
                conv_instance.weight.data[:,:,0::2,0::2]=0
                conv_instance.weight.data[:,:,1,1]=0
            conv_instance.weight.data[:,:,[0,1,1,2],[1,0,2,1]]=conv_weight.data.unsqueeze(-1).repeat(1,1,4)

        elif self.kernel_size==2:
            if self.dilate==2:
                conv_instance.weight.data=conv_weight.data.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,2)

        return conv_instance


    def forward(self, x):
        if self.modify and self.dilate==2:
            self.conv1=self.modify_conv(self.conv1,self.conv1_weight)
        elif self.modify and self.dilate==1:
            self.conv1=self.modify_conv(self.conv1,self.conv1_weight,self.conv1_weight_corner)
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)

        if self.modify and self.dilate==2:
            self.conv2=self.modify_conv(self.conv2,self.conv2_weight)
        elif self.modify and self.dilate==1:
            self.conv2=self.modify_conv(self.conv2,self.conv2_weight,self.conv2_weight_corner)
        x = self.conv2(x)
        x = F.relu( x )
        x = self.bn2(x)

        if self.modify and self.dilate==2:
            self.conv3=self.modify_conv(self.conv3,self.conv3_weight)
        elif self.modify and  self.dilate==1:
            self.conv3=self.modify_conv(self.conv3,self.conv3_weight,self.conv3_weight_corner)
        x = self.conv3(x)
        x = F.relu( x )
        x = self.bn3(x)
        output = self.conv4(x)
        return output


def dilate2_train5_stereoseq(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,image,mask,y_pred_init,max_pretrain_epoch,max_epoch=200,update_interval=3,tol=1e-3,q_cut=0.5,cat_tol=None,mod3_ratio=0.7,pretrain_lr=0.005,lr=0.001,device=torch.device('cuda:0')):
    mask_tensor = torch.from_numpy(mask).view(-1).to(device)
    data = image.transpose((2, 0, 1)) 
    data = data[np.newaxis, :, :, :] 
    data = torch.Tensor(data).to(device)
    pretrain_optimizer = optim.Adam(list(model3.parameters())+list(model2.parameters()), lr=pretrain_lr)

    loss_fn = torch.nn.CrossEntropyLoss()
    model3.train()
    model2.train()
    masky=mask[1:,:]*mask[0:-1,:]
    masky=torch.Tensor(masky).to(device)
    maskz=mask[:,1:]*mask[:,0:-1]
    maskz=torch.Tensor(maskz).to(device)
    maskyz=mask[1:,1:]*mask[0:-1,0:-1]
    maskyz=torch.Tensor(maskyz).to(device)
    loss_hpy = torch.nn.L1Loss(size_average = True)
    loss_hpz = torch.nn.L1Loss(size_average = True)
    loss_hpyz = torch.nn.L1Loss(size_average = True)
    pretrain_label=torch.tensor(y_pred_init).to(torch.long).to(device)
    if cat_tol is None:
        cat_tol=10**(-(len(str(n_clusters))+1))    
    nmi,ari,emb,final_epoch_idx=None,None,None,None     
    for pretrain_epoch in range(max_pretrain_epoch):
        pretrain_optimizer.zero_grad()
        output = mod3_ratio*model3( data )[0]+(1-mod3_ratio)*model2(data)[0]
        output = output.permute( 1, 2, 0 )
        output = output[coord_x,coord_y]
        pretrain_loss=loss_fn(output,pretrain_label)
        pretrain_loss.backward()
        pretrain_optimizer.step()

    #y_pred_last=y_pred_init
    y_pred_last=output.argmax(-1).data.cpu().numpy()
    output_np=output.data.cpu().numpy()
    features=pd.DataFrame(output_np,index=np.arange(0,output_np.shape[0]))
    Group=pd.Series(y_pred_last,index=np.arange(0,output_np.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    centroid=nn.Parameter(torch.zeros([n_clusters,n_clusters]).to(device))
    centroid.data.copy_(torch.Tensor(cluster_centers).to(device))


    optimizer=optim.Adam(list(model3.parameters())+list(model2.parameters())+[centroid],lr=lr)
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        output = mod3_ratio*model3( data )[0]+(1-mod3_ratio)*model2(data)[0]
        
        output = output.permute( 1, 2, 0 )
        outputHP=output
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        HPyz = outputHP[1:, 1:, :] - outputHP[0:-1, 0:-1, :]
        HPy = HPy*masky.unsqueeze(-1)
        HPz = HPz*maskz.unsqueeze(-1)
        HPyz = HPyz*maskyz.unsqueeze(-1)
        lhpy = loss_hpy(HPy,torch.zeros_like(HPy))
        lhpz = loss_hpz(HPz,torch.zeros_like(HPz))
        lhpyz = loss_hpyz(HPyz,torch.zeros_like(HPyz))
        output = output[coord_x,coord_y]
        q=compute_q(output,centroid)
       


        if epoch%update_interval==0:
            p=compute_p(q).data
        kl_loss=compute_kl(p,q)

        loss=step_kl*kl_loss+step_con*(lhpy+lhpz)+step_con1*lhpyz

        rep_argmax_pred=output.argmax(-1)
        max_prob,compute_q_pred=q.max(-1)
        pseudo_idx=max_prob>q_cut
        pseudo_emb=output[pseudo_idx]
        pseudo_lab=compute_q_pred[pseudo_idx]
        loss+=step_ce*nn.CrossEntropyLoss()(pseudo_emb,pseudo_lab)

        loss.backward()
        optimizer.step()
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / output_np.shape[0]
        real_n_clusters=len(np.unique(y_pred))


        if real_n_clusters == n_clusters and (True not in list(pd.Series(y_pred).value_counts().values/y_pred.size<cat_tol)):
            y_pred_last = y_pred
            emb=output.data.cpu().numpy()
            final_epoch_idx=epoch
            if label is not None:
                nmi,ari=eval_perf(y_pred,label)
         
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                
                break
        else:
            break
    return nmi,ari,y_pred_last,emb,final_epoch_idx




class Dilate_MyNet5_visium(nn.Module):
    def __init__(self,input_dim,output_dim,nChannel,n_clusters,kernel_size,dilate):
        super(Dilate_MyNet5_visium, self).__init__()
        if kernel_size==3:
            if dilate==1:
                padding=1
            if dilate==2:
                padding=2
        if kernel_size==2 and dilate==2:
            padding=1
        self.conv1 = nn.Conv2d(input_dim, nChannel,kernel_size=kernel_size, stride=1, padding=padding ,dilation=dilate)
        self.conv1_weight=nn.Parameter(torch.Tensor(nChannel,input_dim))
        nn.init.xavier_normal(self.conv1_weight)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.Conv2d(nChannel, nChannel, kernel_size=kernel_size, stride=1, padding=padding,dilation=dilate )
        self.conv2_weight=nn.Parameter(torch.Tensor(nChannel,nChannel))
        nn.init.xavier_normal(self.conv2_weight)
        self.bn2=nn.BatchNorm2d(nChannel)
        self.conv3 = nn.Conv2d(nChannel, output_dim, kernel_size=kernel_size, stride=1, padding=padding,dilation=dilate )
        self.conv3_weight=nn.Parameter(torch.Tensor(output_dim,nChannel))
        nn.init.xavier_normal(self.conv3_weight)
        self.bn3=nn.BatchNorm2d(output_dim)
        self.conv4 = nn.Conv2d(output_dim, n_clusters, kernel_size=1, stride=1, padding=0 )

        if dilate==1:
            self.conv1_weight_corner=nn.Parameter(torch.Tensor(nChannel,input_dim))
            nn.init.xavier_normal(self.conv1_weight_corner)
            self.conv2_weight_corner=nn.Parameter(torch.Tensor(nChannel,nChannel))
            nn.init.xavier_normal(self.conv2_weight_corner)
            self.conv3_weight_corner=nn.Parameter(torch.Tensor(output_dim,nChannel))
            nn.init.xavier_normal(self.conv3_weight_corner)
        self.kernel_size=kernel_size
        self.dilate=dilate


    def modify_conv(self,conv_instance,conv_weight,conv_weight_corner=None):
        if self.kernel_size==3:
            if self.dilate==1:
                conv_instance.weight.data[:,:,0::2,0::2]=conv_weight_corner.data.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,2)
            elif self.dilate==2:
                conv_instance.weight.data[:,:,0::2,0::2]=0
            conv_instance.weight.data[:,:,[0,1,1,2],[1,0,2,1]]=conv_weight.data.unsqueeze(-1).repeat(1,1,4)

        elif self.kernel_size==2:
            if self.dilate==2:
                conv_instance.weight.data=conv_weight.data.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,2)

        return conv_instance


    def forward(self, x):
        if self.dilate==2:
            self.conv1=self.modify_conv(self.conv1,self.conv1_weight)
        elif self.dilate==1:
            self.conv1=self.modify_conv(self.conv1,self.conv1_weight,self.conv1_weight_corner)
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)

        if self.dilate==2:
            self.conv2=self.modify_conv(self.conv2,self.conv2_weight)
        elif self.dilate==1:
            self.conv2=self.modify_conv(self.conv2,self.conv2_weight,self.conv2_weight_corner)
        x = self.conv2(x)
        x = F.relu( x )
        x = self.bn2(x)

        if self.dilate==2:
            self.conv3=self.modify_conv(self.conv3,self.conv3_weight)
        elif self.dilate==1:
            self.conv3=self.modify_conv(self.conv3,self.conv3_weight,self.conv3_weight_corner)
        x = self.conv3(x)
        x = F.relu( x )
        x = self.bn3(x)
        output = self.conv4(x)
        return output



def dilate2_train5_visium(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,image,mask,y_pred_init,max_pretrain_epoch,max_epoch=200,update_interval=3,tol=1e-3,q_cut=0.5,cat_tol=None,mod3_ratio=0.7,pretrain_lr=0.05,lr=0.01,device=torch.device('cuda:0')):
    mask_tensor = torch.from_numpy(mask).view(-1).to(device)
    data = image.transpose((2, 0, 1)) 
    data = data[np.newaxis, :, :, :]
    data = torch.Tensor(data).to(device)
    pretrain_optimizer = optim.Adam(list(model3.parameters())+list(model2.parameters()), lr=pretrain_lr)

    loss_fn = torch.nn.CrossEntropyLoss()
    model3.train()
    model2.train()
    masky=mask[2:,:]*mask[0:-2,:]
    masky=torch.Tensor(masky).to(device)
    maskz=mask[:,2:]*mask[:,0:-2]
    maskz=torch.Tensor(maskz).to(device)
    maskyz=mask[1:,1:]*mask[0:-1,0:-1]
    maskyz=torch.Tensor(maskyz).to(device)
    loss_hpy = torch.nn.L1Loss(size_average = True)
    loss_hpz = torch.nn.L1Loss(size_average = True)
    loss_hpyz = torch.nn.L1Loss(size_average = True)
    pretrain_label=torch.tensor(y_pred_init).to(torch.long).to(device)
    if cat_tol is None:
        cat_tol=10**(-(len(str(n_clusters))+1))
    nmi,ari,emb,final_epoch_idx=None,None,None,None    
    for pretrain_epoch in range(max_pretrain_epoch):
        pretrain_optimizer.zero_grad()
        output = mod3_ratio*model3( data )[0]+(1-mod3_ratio)*model2(data)[0]
        output = output.permute( 1, 2, 0 )
        output = output[coord_x,coord_y]
        pretrain_loss=loss_fn(output,pretrain_label)
        pretrain_loss.backward()
        pretrain_optimizer.step()


    y_pred_last=output.argmax(-1).data.cpu().numpy()
    output_np=output.data.cpu().numpy()
    features=pd.DataFrame(output_np,index=np.arange(0,output_np.shape[0]))
    Group=pd.Series(y_pred_last,index=np.arange(0,output_np.shape[0]),name="Group")
    Mergefeature=pd.concat([features,Group],axis=1)
    cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
    centroid=nn.Parameter(torch.zeros([n_clusters,n_clusters]).to(device))
    centroid.data.copy_(torch.Tensor(cluster_centers).to(device))


    optimizer=optim.Adam(list(model3.parameters())+list(model2.parameters())+[centroid],lr=lr)
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        output = mod3_ratio*model3( data )[0]+(1-mod3_ratio)*model2(data)[0]
        output = output.permute( 1, 2, 0 )
        outputHP=output
        HPy = outputHP[2:, :, :] - outputHP[0:-2, :, :]
        HPz = outputHP[:, 2:, :] - outputHP[:, 0:-2, :]
        HPyz = outputHP[1:, 1:, :] - outputHP[0:-1, 0:-1, :]
        HPy = HPy*masky.unsqueeze(-1)
        HPz = HPz*maskz.unsqueeze(-1)
        HPyz = HPyz*maskyz.unsqueeze(-1)
        lhpy = loss_hpy(HPy,torch.zeros_like(HPy))
        lhpz = loss_hpz(HPz,torch.zeros_like(HPz))
        lhpyz = loss_hpyz(HPyz,torch.zeros_like(HPyz))
        output = output[coord_x,coord_y]
        q=compute_q(output,centroid)



        if epoch%update_interval==0:
            p=compute_p(q).data
        kl_loss=compute_kl(p,q)

        loss=step_kl*kl_loss+step_con*(lhpy+lhpz)+step_con1*lhpyz

        rep_argmax_pred=output.argmax(-1)
        max_prob,compute_q_pred=q.max(-1)
        pseudo_idx=max_prob>q_cut
        pseudo_emb=output[pseudo_idx]
        pseudo_lab=compute_q_pred[pseudo_idx]
        loss+=step_ce*nn.CrossEntropyLoss()(pseudo_emb,pseudo_lab)

        loss.backward()
        optimizer.step()
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / output_np.shape[0]
        real_n_clusters=len(np.unique(y_pred))


        if real_n_clusters == n_clusters and (True not in list(pd.Series(y_pred).value_counts().values/y_pred.size<cat_tol)):
            y_pred_last = y_pred
            emb=output.data.cpu().numpy()
            final_epoch_idx=epoch
            if label is not None:
                nmi,ari=eval_perf(y_pred,label)
           

            print(epoch,ari)
            if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                break
        else:
            break
    return nmi,ari,y_pred_last,emb,final_epoch_idx
