'''
Compute Linear SVM accuracy score for a given model name
'''
import numpy as np 
import torch
import pickle
import argparse

# ignore convergence warnings from sklearn
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

from . import misc
from .loader import ModelLoader


# defining the argument parser
parser = argparse.ArgumentParser(description='Extract and save pretrained model features')
parser.add_argument('-n','--modelname', type=str, help='model name', required = True)


args = parser.parse_args()


# model loader instance
model_loader = ModelLoader()

# load model
model = model_loader.load(args.modelname)

# transfer model to device
model = model.to(misc.device)

# read cadieu dataset
data_path = 'data/PLoSCB2014_data_20141216'
with open('data/PLoSCB2014_data_20141216/NeuralData_IT_multiunits.pkl','rb') as f:
    data = pickle.load(f)

# read input images
X = misc.read_images(image_paths=data['image_paths'],data_path=data_path)

# extract model features
model_features = misc.extract_features(X,model)

# neural features
neural_features = data['features']

# compute lsvm score
lsvm_acc_mean,lsvm_acc_std = misc.linear_svm_score_v2(model_features,data['image_ctg'],neural_features,
                                            			num_val_splits=10,val_ratio=0.2)

# print results
print ("\nModel name: {}\t lsvm_acc_mean: {:.4f}\t lsvm_acc_std: {:.4f}".format(args.modelname,lsvm_acc_mean,lsvm_acc_std))