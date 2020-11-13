# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:07:33 2019

@author: Yang Xu

MNIST
"""

import numpy as np
import pandas as pd

##neural net
import torch
import torch.nn.functional as F

import umap
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from Center_loss_pytorch import CenterLoss
from bi_tempered_loss_pytorch import bi_tempered_logistic_loss

import warnings
warnings.filterwarnings("ignore")

##-----------------------------------------------------------------------------
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        y = self.fc2(out)
        return out,y
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)  
        return out
    
class ConvNet_fashion(torch.nn.Module):
    def __init__(self):
        super(ConvNet_fashion, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(64))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(128))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 32, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(32))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(288, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 288), p=2, dim=1)
        y = self.fc2(out)
        return out,y
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 288), p=2, dim=1)
        
        return out
    
class ConvNet_USPS(torch.nn.Module):
    def __init__(self):
        super(ConvNet_USPS, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        y = self.fc2(out)
        return out,y
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        return out
    
##-----------------------------------------------------------------------------
def evaluation(y_pred,y,cluster_method="Kmeans"):
    
    if cluster_method=="Kmeans":
        embedding = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=10,
                              metric='correlation').fit_transform(y_pred)
    
        kmeans = KMeans(n_clusters=10, random_state=1).fit(embedding)
        centroid = kmeans.cluster_centers_.copy()
        y_label = kmeans.labels_.copy()
    elif cluster_method=="SC":
        embedding = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=10,
                              metric='correlation').fit_transform(y_pred)
        clustering = SpectralClustering(n_clusters=10,
                                        assign_labels="discretize",
                                        random_state=0).fit(embedding)
        y_label = clustering.labels_.copy()
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values
    else:
        embedding = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=10,
                              metric='correlation').fit_transform(y_pred)
        gmm = GaussianMixture(n_components=10).fit(embedding)
        y_label = gmm.predict(embedding)
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values
    
    cm = confusion_matrix(y, y_label)
    nmi = round(normalized_mutual_info_score(y,y_label),5)

    y_pseudo=np.zeros((y.shape[0],10))
    
    ##t-student distribution kernel soft-assignment,alpha=1
    #for j in range(10):
    #    y_pseudo[:,j]=(np.linalg.norm(embedding-centroid[j,:],axis=1)+1)**(-1)
    #y_pseudo = pd.DataFrame(y_pseudo)
    #y_pseudo2=np.zeros((y_pred.shape[0],10))
    #for j in range(10):
    #    y_pseudo2[:,j]=y_pseudo.iloc[:,j].values/np.sum(
    #        y_pseudo[y_pseudo.columns.difference([j])].values,axis=1)
    #y_pseudo = y_pseudo2
    
    for j in range(10):
        y_pseudo[:,j]=1/np.linalg.norm(embedding-centroid[j,:],axis=1)
    y_pseudo=softmax(y_pseudo,axis=1)
    
    ##auxiliary target distribution
    f = np.sum(np.square(y_pseudo)/np.sum(y_pseudo,axis=0),axis=1)
    y2 = np.square(y_pseudo)/np.sum(y_pseudo,axis=0)
    au_tar = (y2.T/f).T
    au_tar=au_tar[:,np.argmax(cm,axis=1).tolist()]
    
    return au_tar, nmi, y_label,embedding

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.kaiming_uniform(m.weight.data)

##-----------------------------------------------------------------------------
##MNIST handwritten data
mnist = pd.read_csv("mnist_test.csv",
                    header=0,index_col=False)

##Fashion data
#mnist = pd.read_csv("fashion-mnist_test.csv",
#                    header=0,index_col=False)
    
mnist= mnist.sample(frac=0.1)
ori_X = mnist.iloc[:,1:].values
y = mnist.iloc[:,0].values

new_X = ori_X.copy()
new_X = new_X/255.0
new_X = new_X.reshape((10000,1,28,28))

del ori_X
del mnist

num_sample = new_X.shape[0]
batch_size = 128

##-----------------------------------------------------------------------------
##USPS data
import h5py
with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
new_X = X_tr.reshape(len(y_tr),1,16,16)
y = y_tr
batch_size=128

##-----------------------------------------------------------------------------
##training
#use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ConvNet_fashion()
net.apply(weights_init)

#net.fc2.weight.requires_grad = False
#net.fc2.bias.requires_grad = False
dim_fea = 288
#t1, t2 = 1.0, 1.0
t1, t2 = 0.8, 1.2#with 0.01 center loss, I saw 0.70 NMI
#t1, t2 = 0.5, 1.5
#t1, t2 = 0.2, 1.8
#t1, t2 = 0.2, 4.0#it works
criterion_cent = CenterLoss(num_classes=10, feat_dim=dim_fea, use_gpu=True)
optimizer_model = torch.optim.Adam(net.parameters())
optimizer_centloss = torch.optim.Adam(criterion_cent.parameters())
#optimizer_model = torch.optim.SGD(net.parameters(),lr=0.01)
#optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(),lr=0.01)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_model, base_lr=0.01, 
#                                              max_lr=0.1)
#scheduler_center = torch.optim.lr_scheduler.CyclicLR(optimizer_centloss, 
#                                                     base_lr=0.01, max_lr=0.1)

num_epoch = 11
X_all_tensor = torch.tensor(new_X).float()
acc=[]
nmis = []
fea = []

y_pred = np.zeros((new_X.shape[0],dim_fea))
for j in range(new_X.shape[0]//batch_size+1):
    pred = net.forward_feature(X_all_tensor[j*batch_size:(j+1)*batch_size,:,:,:])
    pred = torch.Tensor.cpu(pred).detach().numpy()
    y_pred[j*batch_size:(j+1)*batch_size,:]=pred
fea.append(y_pred)
au_tar,nmi, y_label, embedding = evaluation(y_pred,y,cluster_method="GMM")
acc.append(nmi)

##update every epoch
for k in range(1,num_epoch):

    old_label=y_label.copy()
    net.to(device)

    X_train, X_test, y_train, y_test = train_test_split(new_X, au_tar, test_size=0.3)
    X_tensor=torch.tensor(X_train).float()
    y_tensor = torch.tensor(y_train).float()
    label_tensor = np.argmax(y_train,axis=1)
    label_tensor = torch.LongTensor(label_tensor).float()
    n = y_train.shape[0]
    
    for j in range(n//batch_size):
        
        inputs = X_tensor[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
        outputs = y_tensor[j*batch_size:(j+1)*batch_size,:].to(device)
        label = label_tensor[j*batch_size:(j+1)*batch_size].to(device)
        feas,output = net.forward(inputs)
        
        clf_loss = bi_tempered_logistic_loss(output, outputs,t1, t2)
        cen_loss = criterion_cent(feas, label)
        
        loss = clf_loss# + cen_loss*0.01
        optimizer_model.zero_grad()
        #optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        #for param in criterion_cent.parameters():
        #    param.grad.data *= (1. / 0.01)
        #optimizer_centloss.step()
        
        #scheduler.step()
        #scheduler_center.step()
    
    net.to(torch.device("cpu"))
    y_pred = np.zeros((new_X.shape[0],dim_fea))
    for j in range(new_X.shape[0]//batch_size+1):
        pred = net.forward_feature(X_all_tensor[j*batch_size:(j+1)*batch_size,:,:,:])
        pred = torch.Tensor.cpu(pred).detach().numpy()
        y_pred[j*batch_size:(j+1)*batch_size,:]=pred
    fea.append(y_pred)
    au_tar,nmi, y_label, embedding = evaluation(y_pred,y,cluster_method="GMM")
    print("Accuracy"+"("+str(k)+"/"+str(k-1)+"): "+str(nmi))
    acc.append(nmi)
    nmi = round(normalized_mutual_info_score(old_label, y_label),5)
    #print("NMI"+"("+str(k)+"/"+str(k-1)+"): "+str(nmi))
    nmis.append(nmi)
