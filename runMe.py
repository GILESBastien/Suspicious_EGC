from Both import Both
from torch_geometric.data import Data
from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from datetime import datetime
import argparse
import gc
from model import Dominant
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import torch

import time
import math
from pygod.utils.utility import validate_device

from utils import load_anomaly_detection_dataset
import csv
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid 
from torch_geometric.datasets import Amazon 
import random
from pygod.metrics import eval_roc_auc

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='Cora', help='dataset name: Flickr/ACM/BlogCatalog')
	parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden embedding (default: 64)')
	parser.add_argument('--epoch', type=int, default=500, help='Training epoch')
	parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
	parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
	parser.add_argument('--alpha', type=float, default=0.5, help='balance parameter')
	parser.add_argument('--device', default=0, type=int, help='cuda/cpu')
	parser.add_argument('--hidden_dim2', type=int, default=32, help='dimension of hidden embedding (default: 64)')
	parser.add_argument('--epoch2', type=int, default=500, help='Training epoch')
	parser.add_argument('--lr2', type=float, default=5e-3, help='learning rate')
	parser.add_argument('--dropout2', type=float, default=0.5, help='Dropout rate')
	parser.add_argument('--s_dens', type=float, default=0.1, help='Suspected sample density')
	parser.add_argument('--error', type=float, default=0, help='Suspected sample density')
	# parser.add_argument('--trust', type=float, default=1., help='Trust')
	args = parser.parse_args()
	# os.environ["CUDA_VISIBLE_DEVICES"]=""
	
	F = open("ResultBothGCN10.csv","a")
	writer=csv.writer(F,delimiter=';')
	print(parser)
	
	
	
	if args.dataset=="Cora" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=6] = 0
		data.y[data.y==6]=1
	elif args.dataset=="Citeseer" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=0] = 10
		data.y[data.y==0]=1
		data.y[data.y==10]=0
	elif  args.dataset=="PubMed" :
		data = Planetoid('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=0] = 10
		data.y[data.y==0]=1
		data.y[data.y==10]=0
	elif  args.dataset=="Computers" :
		data = Amazon('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=9] = 0
		data.y[data.y==9]=1
	elif  args.dataset=="Photo" :
		data = Amazon('./data/'+args.dataset, args.dataset, transform=T.NormalizeFeatures())[0]
		data.y[data.y!=7] = 0
		data.y[data.y==7]=1
	rauc=np.empty(0)
	rn=np.empty(0)
	rs=np.empty(0)
	rnsubrs=np.empty(0)
	rndivrs=np.empty(0)
	d=np.empty(0)
	kuma=np.empty(0)
	write=np.empty(0)
	data.y=data.y.bool()
	write=np.append(write,args.dataset)
	write=np.append(write,args.s_dens)
	write=np.append(write,args.epoch)
	write=np.append(write,args.epoch2)
	write=np.append(write,args.dropout)
	write=np.append(write,args.dropout2)
	write=np.append(write,args.hidden_dim)
	write=np.append(write,args.hidden_dim2)
	write=np.append(write,args.alpha)
	path="./"+str(args.dataset)+"/"+str(args.s_dens)+"/"+str(args.epoch)+"/"+str(args.epoch2)+"/"+str(args.lr)+'/'+str(args.lr2)
	isExist = os.path.exists(path)
	if not isExist:
		os.makedirs(path)
	device = validate_device(args.device)
	
	for j in range(0,10):		
		print("-------------------------------------------------------"+str(j))
		indexes=np.arange(int(len(data.y)))  
		normal=np.empty(0)
		suspicious=np.empty(0)
		index_labeled = np.empty(0)
		
		for i in range(1,int(len(data.y)*args.s_dens)):
			index_labeled=np.append(index_labeled,i)				
		#---------------------------------------------------------------------new block------------------------------------------------
		x = pd.DataFrame(data.x.numpy())
		y = pd.DataFrame(data.y.numpy())
		node_data=x
		feature_names = ["w_{}".format(ii) for ii in range(x.shape[1])]
		column_names =	feature_names + ["subject"]
		node_data.columns=feature_names
		node_data["label"]=y
		
		unlabeled_data, labeled_data, index_unlabeled, index_labeled = train_test_split(node_data, indexes, test_size=0.1, stratify=node_data['label'], random_state=j)
		
		index_labeled=index_labeled.astype(int)
		
	
		edge_list= pd.DataFrame(data.edge_index.numpy().T)
		edge_list.columns=['target','source']
		num_node, num_feature = node_data.shape[0], node_data.shape[1]-1
		node_index = np.array(node_data.index)
		index_unlabeled = np.delete(indexes,index_labeled.astype(int))
		unlabeled_data = np.take(node_data,index_unlabeled,axis=0)
		# labeled_data = np.take(node_data,index_labeled,axis=0)
		train_data, val_data, index_train, index_val = train_test_split(labeled_data, index_labeled, test_size=0.2, stratify=labeled_data['label'], random_state=j)
		index_normal_train = []
		index_anomaly_train = []
		error=100-args.error
		for i in index_train:
			r=random.randrange(100)
			if node_data.iloc[i.astype(int)]['label'] == 0:
				if(r<error):
					index_normal_train.append(i)
				else:
					index_anomaly_train.append(i)
			elif node_data.iloc[i]['label'] == 1:
				if(r<error):
					index_anomaly_train.append(i)
				else:
					index_normal_train.append(i)	
		index_normal_val = []
		index_anomaly_val = []
		for i in index_val:
			r=random.randrange(100)
			if node_data.iloc[i]['label'] == 0:
				if(r<error):
					index_normal_val.append(i)
				else:
					index_anomaly_val.append(i)
			elif node_data.iloc[i]['label'] == 1:
				if(r<error):
					index_anomaly_val.append(i)
				else:
					index_normal_val.append(i)     
		#-----------------------------------------------------------------------------------------------------------------------------
		
		sample_s =torch.LongTensor(index_anomaly_train)
		sample_n =torch.LongTensor(index_normal_train)
		sample_sv =torch.LongTensor(index_anomaly_val)
		sample_nv =torch.LongTensor(index_normal_val)
		
		model = Both(Sample_n=sample_n,Sample_vn=sample_nv, Sample_s= sample_s,Sample_vs=sample_sv,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
		model.fit(data)
		Rn = model.decision_function(data)

		model = Both(Sample_n=sample_s,Sample_vn=sample_sv, Sample_s= sample_n,Sample_vs=sample_nv,epoch=args.epoch,hid_dim=args.hidden_dim,gpu=args.device,dropout=args.dropout,alpha=args.alpha)
		model.fit(data)
		Rs = model.decision_function(data)
			
		Rn=(Rn-Rn.min())/(Rn.max()-Rn.min())
		Rs=(Rs-Rs.min())/(Rs.max()-Rs.min())
		Rs=Rs+0.0001
		
		Rs=Rs+0.001
		Rn=np.nan_to_num(Rn)
		Rs=np.nan_to_num(Rs)
		sample=np.append(normal,suspicious)
		indexes=np.delete(indexes,sample.astype(int) )
		
		auc_score = eval_roc_auc(data.y.numpy()[indexes], Rn[indexes])
		rn=np.append(rn,auc_score)
		print('AUC Score Norm:', auc_score)
		auc_score = eval_roc_auc(data.y.numpy()[indexes],-Rs[indexes])
		print('AUC Score Susp:', auc_score)
		rs=np.append(rs,auc_score)
		auc_score = eval_roc_auc(data.y.numpy()[index_unlabeled], Rn[index_unlabeled]-Rs[index_unlabeled])
		rnsubrs=np.append(rnsubrs,auc_score)
		print('AUC Score Norm-Susp:', auc_score)
		auc_score = eval_roc_auc(data.y.numpy()[indexes], Rn[indexes]/Rs[indexes])
		rndivrs=np.append(rndivrs,auc_score)
		print('AUC Score Norm/Susp:', auc_score)
		t=100
		print(auc_score)
		np.savetxt(path+"Vs"+str(j)+".csv",np.append(suspicious,normal))
		np.savetxt(path+"Rn"+str(j)+".csv",Rn)
		np.savetxt(path+"Rs"+str(j)+".csv",Rs)
		
	write=np.append(write,np.mean(rn))
	write=np.append(write,np.std(rn))
	write=np.append(write,np.mean(rs))
	write=np.append(write,np.std(rs))
	write=np.append(write,np.mean(rnsubrs))
	write=np.append(write,np.std(rnsubrs))
	write=np.append(write,np.mean(rndivrs))
	write=np.append(write,np.std(rndivrs))

	writer=csv.writer(F,dialect='excel',delimiter=';',lineterminator = '\n')
	writer.writerow(write)	