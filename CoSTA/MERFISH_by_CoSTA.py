# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:07:33 2019

@author: Yang Xu

"""

import cv2
import numpy as np
import pandas as pd
import NaiveDE

##neural net
import torch
import torch.nn.functional as F

import umap
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score

from bi_tempered_loss_pytorch import bi_tempered_logistic_loss

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
##-----------------------------------------------------------------------------
class ConvNet_MERFISH(torch.nn.Module):
    def __init__(self):
        super(ConvNet_MERFISH, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=11,stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=7,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128*2*2, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*2*2), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*2*2), p=2, dim=1)
        return out

##-----------------------------------------------------------------------------
def evaluation(y_pred,cluster_method="Kmeans",num_cluster = 25,n_neighbors=20,min_dist=0.0):
    
    if cluster_method=="Kmeans":
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
    
        kmeans = KMeans(n_clusters=num_cluster, random_state=1).fit(embedding)
        centroid = kmeans.cluster_centers_.copy()
        y_label = kmeans.labels_.copy()
        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))
    elif cluster_method=="SC":
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
        clustering = SpectralClustering(n_clusters=num_cluster,
                                        assign_labels="discretize",
                                        random_state=0).fit(embedding)
        y_label = clustering.labels_.copy()
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values
        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))

    else:
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
        gmm = GaussianMixture(n_components=num_cluster).fit(embedding)
        y_label = gmm.predict(embedding)
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values

        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))
    
    ##t-student distribution kernel soft-assignment,alpha=1
    #for j in range(centroid.shape[0]):
    #    y_pseudo[:,j]=(np.linalg.norm(embedding-centroid[j,:],axis=1)+1)**(-1)
        ##cosine distance
        #y_pseudo[:,j]=((1-cosine_similarity(embedding,centroid[j,:].reshape(1,embedding.shape[1]))+1)**(-1))[:,0]
    #y_pseudo = pd.DataFrame(y_pseudo)
    #y_pseudo2=np.zeros((y_pred.shape[0],centroid.shape[0]))
    #for j in range(centroid.shape[0]):
    #    y_pseudo2[:,j]=y_pseudo.iloc[:,j].values/np.sum(
    #        y_pseudo[y_pseudo.columns.difference([j])].values,axis=1)
    #y_pseudo = y_pseudo2
    
    ##distance based soft-assignment
    for j in range(centroid.shape[0]):
        ##euclidean distance
        y_pseudo[:,j]=1/np.linalg.norm(embedding-centroid[j,:],axis=1)
        ##cosine similarity
        #y_pseudo[:,j]=1/(1-cosine_similarity(embedding,centroid[j,:].reshape(1,embedding.shape[1])))[:,0]
    y_pseudo=softmax(y_pseudo,axis=1)
    
    ##auxiliary target distribution
    f = np.sum(np.square(y_pseudo)/np.sum(y_pseudo,axis=0),axis=1)
    y2 = np.square(y_pseudo)/np.sum(y_pseudo,axis=0)
    au_tar = (y2.T/f).T
    
    return au_tar, y_label,embedding

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
def get_neighors(gene_list=None, embedding=None, target=["Vim"]):
    embedding = pd.DataFrame(embedding)
    embedding.index = gene_list
    gene_neighbors={}
    for i in target:
        distance = np.linalg.norm(embedding.values-embedding.loc[i,:].values,axis=1)
        distance = pd.DataFrame(distance)
        distance.index=gene_list
        distance = distance.sort_values(ascending=True,by=0)
        gene_neighbors[i]=distance.index.tolist()[1:51]
    return gene_neighbors

##-----------------------------------------------------------------------------
##Analyzing MERFISH data
gene = pd.read_csv("merfish_all_data.csv",
                    header=0,index_col=0)
n = gene.shape[0]
samples = gene.index.tolist()[-15:]
new_X = gene.values.copy().reshape((n,1,85,85))

##-----------------------------------------------------------------------------
##training
#use_cuda = torch.cuda.is_available()    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ConvNet_MERFISH()
net.apply(weights_init)

#t1, t2 = 0.5, 2.0
t1, t2 = 0.8, 1.2
num_epoch = 6
batch_size = 170
X_all_tensor = torch.tensor(new_X).float()

y_pred = net.forward_feature(X_all_tensor)
y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=5,min_dist=0.0,
                                        num_cluster=10,cluster_method='GMM') 
original = y_label.copy()
nmis=[]

##learning plan
#opt = torch.optim.SGD(net.parameters(),lr=0.01, momentum=0.9)
opt = torch.optim.Adam(net.parameters())

##for visualization
embedding = umap.UMAP(n_neighbors=5, min_dist=1, n_components=2,
                      metric='correlation').fit_transform(y_pred)

embedding = pd.DataFrame(embedding)
embedding.columns=['UMAP1','UMAP2']
embedding["Proton"]=original
f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Proton",
             fit_reg=False,legend=False,scatter_kws={'s':15})
for i in list(set(y_label)):
    plt.annotate(i, 
                 embedding.loc[embedding['Proton']==i,['UMAP1','UMAP2']].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold')

f.savefig("merfish_initial_umap.jpeg",dpi=450)
#f.savefig("merfish_trained_umap.jpeg",dpi=450)

for k in range(1,num_epoch):
    old_label=y_label.copy()
    net.to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(new_X, au_tar, test_size=0.001)
    X_tensor=torch.tensor(X_train).float()
    y_tensor = torch.tensor(y_train).float()
    n = y_train.shape[0]
    for j in range(n//batch_size):
        inputs = X_tensor[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
        outputs = y_tensor[j*batch_size:(j+1)*batch_size,:].to(device)
        opt.zero_grad()
        output = net.forward(inputs)
        #loss = Loss(output, outputs)
        loss = bi_tempered_logistic_loss(output, outputs,t1, t2)
        loss.backward()
        opt.step()
        
    #if k%5==0:
    net.to(torch.device("cpu"))
    y_pred = net.forward_feature(X_all_tensor)
    y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
    au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=5,min_dist=0.0,
                                            num_cluster=10,cluster_method='GMM') 
    cm = confusion_matrix(old_label, y_label)
    au_tar=au_tar[:,np.argmax(cm,axis=1).tolist()]
    nmi = round(normalized_mutual_info_score(old_label, y_label),5)
    print("NMI"+"("+str(k)+"/"+str(k-1)+"): "+str(nmi))
    nmis.append(nmi)
        
torch.save(net, "merfish_models")
net = torch.load("merfish_model")

neg = ['Blank_1','Blank_2','Blank_3','Blank_4','Blank_5']     

gene_list = get_neighors(gene_list=gene.index.tolist(),
                         embedding=y_pred,target=samples)
gene_neis = []
for key,values in gene_list.items():
    gene_neis = gene_neis+values[:5]    
gene_neis = list(set(gene_neis))
gene_neis=[i for i in gene_neis if i not in samples]
set(gene_neis).intersection(neg)

##permutation and null distribution
new_y_pred=pd.DataFrame(y_pred.copy())
new_y_pred.index = gene.index
net.to(torch.device("cpu"))
sub_X = new_X.copy().reshape(176,85*85)
sub_X = pd.DataFrame(sub_X)
sub_X.index = gene.index
SE_genes_hi = {}
SE_genes_low = {}
SE_genes_med = {}
for i in samples:
    SE_genes_hi[i]=[]
    SE_genes_low[i]=[]
    SE_genes_med[i]=[]
    if i in gene.index:
        for j in gene.index.tolist():
            if j not in samples:
                null = np.zeros((101,85*85))
                null[0,:]=sub_X.loc[j,:].values.copy()
                for k in range(1,101):
                    g = sub_X.loc[j,:].values.copy()
                    np.random.shuffle(np.transpose(g))
                    null[k,:]= g
                null = null.reshape(101,1,85,85)
                X_tensor = torch.tensor(null).float()
                y_pred = net.forward_feature(X_tensor)
                y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
                distance = np.linalg.norm(y_pred-new_y_pred.loc[i,:].values,axis=1)
                zscore=scipy.stats.zscore(distance)
                if zscore[0]<-1.645:
                    SE_genes_low[i].append(j)
                if zscore[0]<-2.325:
                    SE_genes_med[i].append(j)
                if zscore[0]<-3.1:
                    SE_genes_hi[i].append(j)
                    
##-----------------------------------------------------------------------------
##HC clustering
from sklearn.cluster import AgglomerativeClustering
n_clusters = 9
ward = AgglomerativeClustering(n_clusters=n_clusters,
                               affinity='euclidean', linkage='ward')
ward.fit(y_pred)
y_pred['Label']=ward.labels_    

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 75))
plt.title("Gene Dendograms")
dend = shc.dendrogram(shc.linkage(y_pred, method='ward'),
                      leaf_font_size=16, labels=gene.index,
                      leaf_rotation=0, orientation="left")
