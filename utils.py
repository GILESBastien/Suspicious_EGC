import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import pandas as pd
def load_anomaly_detection_dataset(dataset, datadir,sample_density,sample_size):

	data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
	adj = data_mat['Network']
	feat = data_mat['Attributes'].astype(float)
	Label =pd.DataFrame(data_mat['Label'])
	normal=np.empty(0)
	for i in range(1,Label.size):
		normal=np.append(normal,i)
	suspicious = np.empty(0)
	for i in range(0,int(sample_density*sample_size)):
		suspicious_a=Label.index[Label[0]==1].tolist() 
		r=random.randrange(len(suspicious_a))
		suspicious=np.append(suspicious,suspicious_a[r])
		normal=np.delete(normal,np.argwhere(normal==suspicious_a[r]))
	for i in range(0,int(sample_size-sample_density*sample_size)):
		suspicious_n=Label.index[Label[0]==0].tolist() 
		r=random.randrange(len(suspicious_n))
		suspicious=np.append(suspicious,suspicious_n[r])
		normal=np.delete(normal,np.argwhere(normal==suspicious_n[r]))
	# suspicious = data_mat['Suspicious']
	# normal = data_mat['Normal']
	#truth= np.take_along_axis(truth,suspicious.astype(int),axis=0)
	# truth = truth.flatten()

	adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
	adj_norm = adj_norm.toarray()
	adj = adj + sp.eye(adj.shape[0])
	adj=adj
	# print(feat)
	#feat = feat/feat.max(axis=0)
	# print(feat)
	return adj_norm, feat, np.array(Label[0]), adj, normal, suspicious

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()