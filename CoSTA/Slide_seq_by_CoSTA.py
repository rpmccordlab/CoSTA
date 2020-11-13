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

class ConvNet(torch.nn.Module):
    def __init__(self,out_dim=25):
        super(ConvNet, self).__init__()
        self.out_dim = out_dim
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=5,stride=2, padding=1),
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
            torch.nn.Tanh(),#torch.nn.ReLU(),#
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128, self.out_dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        return out

##-----------------------------------------------------------------------------
def evaluation(y_pred,cluster_method="Kmeans",num_cluster = 25,n_neighbors=20,min_dist=0.0):
    '''
    it supports Kmeans, Spectral clustering and GMM 3 clustering methods
    '''
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
    
    ##alternative approach to assigne soft-assignment through t-student distribution
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
    
    ##soft-assignment used in this study
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
##-----------------------------------------------------------------------------
##Slide-seq
gene = pd.read_csv("spatial_gene_profile_slide_seq_2w_raw.csv",
                    header=0,index_col=0)
#gene = pd.read_csv("spatial_gene_profile_slide_seq_3d_raw.csv",
#                    header=0,index_col=0)
    
n,m = gene.shape

a,b = gene['x'].values[1],gene['y'].values[1]
new_X = gene.iloc[:,:-2].values

new_X = new_X.reshape((n,a,b))
resize_X = []
for i in range(n):
    resize_X.append(cv2.resize(new_X[i,:,:], (48,48)))
    #resize_X.append(cv2.resize(new_X[i,:,:]/np.max(new_X[i,:,:]), (48,48)))
    
resize_X = np.asarray(resize_X)
new_X=resize_X.copy()
new_X = new_X.reshape((n,1,48,48))
del resize_X,a,b,i

##-----------------------------------------------------------------------------
n,_,a,b=new_X.shape
counts = pd.DataFrame(new_X.reshape(n,a*b)).T
counts.columns = gene.index

totals = np.sum(counts,axis=1)
bin1 = np.repeat(np.array([i for i in range(a)]), b)
bin2 = np.tile(np.array([i for i in range(b)]), a)
samples = pd.DataFrame({'x':bin1,'y':bin2,'total_counts':totals})

resid_expr = NaiveDE.regress_out(samples, counts.T, 'np.log(total_counts+1)').T
new_X = resid_expr.T.values.reshape((n,1,48,48))

##-----------------------------------------------------------------------------
##training
##this only will train one ConvNet. For ensemble learning, repeat this 5 times

#use_cuda = torch.cuda.is_available()
output_dim = 30##training with 30 clusters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ConvNet(out_dim=output_dim)
net.apply(weights_init)

num_epoch = 31
batch_size = 128

#t1, t2 = 1.0, 1.0
t1, t2 = 0.8, 1.2## parameters for bi-tempered logistic loss
#t1, t2 = 0.5, 1.5

#Loss = torch.nn.KLDivLoss()
X_all_tensor = torch.tensor(new_X).float()
y_pred = net.forward_feature(X_all_tensor)
y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=20,
                                        min_dist=0.0,num_cluster=output_dim,
                                        cluster_method="GMM") 
nmis=[0]
features = []
losses_list=[]
gene_neis=[]

##learning plan
opt = torch.optim.SGD(net.parameters(),lr=0.01, momentum=0.9)

for k in range(1,num_epoch):
    old_label=y_label.copy()
    net.to(device)
        
    X_train, X_test, y_train, y_test = train_test_split(new_X, au_tar, test_size=0.3)
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
    
    #if k%10==0:
    net.to(torch.device("cpu"))
    y_pred = net.forward_feature(X_all_tensor)
    y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
    
    au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=20,
                                            min_dist=0.0,num_cluster=output_dim,
                                            cluster_method="GMM")
    cm = confusion_matrix(old_label, y_label)
    au_tar=au_tar[:,np.argmax(cm,axis=1).tolist()]

    nmi = round(normalized_mutual_info_score(old_label, y_label),5)
    print("NMI"+"("+str(k)+"/"+str(k-1)+"): "+str(nmi))
    if nmi>=max(nmis):
        features.append(y_pred)
    nmis.append(nmi)
    #torch.save(net, "ConvNet_model_"+str(k))

##-----------------------------------------------------------------------------
##whole dataset stability test
import os
from itertools import combinations

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

##files that are learned features by ConvNet
files = os.listdir("Slide_results/features/")
features = []
for f in files:
    features.append(pd.read_csv("Slide_results/features/"+f,
                                header=None).values)
features = np.asarray(features)

##-----------------------------------------------------------------------------
##identify SE genes
gene_neis={}
for i in range(features.shape[0]):
    gene_neis[i]= get_neighors(gene_list=gene.index.tolist(),
                               embedding=features[i,:,:],
                               target=gene.index.tolist())

combs = list(combinations([i for i in range(features.shape[0])], 2))
neighbors = [5,10,15,20,25,30,40,50,100]
stability = np.zeros((gene.shape[0],len(neighbors),len(combs)))

for nei in range(len(neighbors)):
    for c in range(len(combs)):
        gene_neis1 = gene_neis[combs[c][0]]
        gene_neis2 = gene_neis[combs[c][1]]
        sim=[]
        for i in range(gene.shape[0]):
            list1 = gene_neis1[gene.index[i]][:neighbors[nei]]
            list2 = gene_neis2[gene.index[i]][:neighbors[nei]]
            sim.append(jaccard_similarity(list1, list2))
        stability[:,nei,c]=np.array(sim)

sta_mean = np.mean(stability,axis=2)
sta_mean = np.max(sta_mean,axis=1)

genes = gene[sta_mean>=0.2]
gene_list = genes.index.tolist()
gene_list = list(set(gene_list))

##-----------------------------------------------------------------------------
##identify correlated genes of genes of interest
full_list = []
for i in range(features.shape[0]):
    
    genes=gene.loc[gene_list,:]
    new_y_pred=features[i,:,:].copy()
    new_y_pred = pd.DataFrame(new_y_pred)
    new_y_pred.index = gene.index
    new_y_pred = new_y_pred.loc[gene_list,:]
    
    ##gene of interest
    gene_interest = ['Vim','Ctsd','Gfap']#['Sox4','Sox10']#['Id3','Lgals1']#

    correlated_genes = []
    for g in gene_interest:
        distance = np.linalg.norm(new_y_pred.values-new_y_pred.loc[g,:].values,axis=1)
        distance = pd.DataFrame(distance)
        distance.index=genes.index
        distance = distance.sort_values(ascending=True,by=0)
        distance['zscore']=scipy.stats.zscore(distance[0].values)
        #Vim = distance[distance['zscore'].values<-1.645].index.tolist()
        Vim = distance[distance['zscore'].values<-2.325].index.tolist()
        #Vim = distance[distance['zscore'].values<-3.1].index.tolist()
        correlated_genes = correlated_genes +Vim
    correlated_genes = list(set(correlated_genes))
    full_list+=correlated_genes
    
import collections
full_freq = collections.Counter(full_list)
full_list = []
for k,v in full_freq.items():
    if v>=3:
        full_list.append(k)
full_list.sort()

##-----------------------------------------------------------------------------
##for clustering
embedding = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2,
                      metric='cosine').fit_transform(new_y_pred)
kmeans = KMeans(n_clusters=6, random_state=1).fit(embedding)
y_label = kmeans.labels_.copy()

##for visualization
embedding = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2,
                      metric='cosine').fit_transform(new_y_pred)

embedding = pd.DataFrame(embedding)
embedding.columns=['UMAP1','UMAP2']
embedding["Proton"]=y_label
f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Proton",
             fit_reg=False,legend=False,scatter_kws={'s':5})
for i in list(set(y_label)):
    plt.annotate(i, 
                 embedding.loc[embedding['Proton']==i,['UMAP1','UMAP2']].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold')
