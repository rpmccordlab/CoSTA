# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:42:17 2020

@author: xuyan
"""
import cv2
import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE

import warnings
warnings.filterwarnings("ignore")

##-----------------------------------------------------------------------------
gene = pd.read_csv("spatial_gene_profile_slide_seq_2w_raw.csv",
                    header=0,index_col=0)
n,m = gene.shape

a,b = gene['x'].values[1],gene['y'].values[1]
new_X = gene.iloc[:,:-2].values

new_X = new_X.reshape((n,a,b))
resize_X = []
for i in range(n):
    resize_X.append(cv2.resize(new_X[i,:,:], (48,48)))
    
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

##-----------------------------------------------------------------------------
norm_expr = NaiveDE.stabilize(counts.T).T
resid_expr = NaiveDE.regress_out(samples, counts.T, 'np.log(total_counts+1)').T

X = samples[['x', 'y']]

results = SpatialDE.run(X, resid_expr)
#results.sort_values('qval').head(10)[['g', 'l', 'qval']]

##-----------------------------------------------------------------------------
sign_results = results.query('qval < 0.01')
#sign_results['l'].value_counts()
#np.mean(sign_results['l'].values)

histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr, 
                                                             sign_results, 
                                                             C=10, l=1.5, verbosity=1,
                                                             maxiter=30)
histology_results=histology_results.sort_values(by=['g'])