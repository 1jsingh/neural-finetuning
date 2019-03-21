'''
Extract and save pretrained model features
'''

# numpy
import numpy as np
# pickle
import pickle
# argparse
import argparse
import os

#model loader
from .loader import ModelLoader

# misc utility functions
from . import misc

# defining the argument parser
parser = argparse.ArgumentParser(description='Extract and save pretrained model features')
parser.add_argument('-n','--modelnames', type=str, help='list of model names', required = False)


args = parser.parse_args()

# get model names
model_names = args.modelnames

# model loader instance
model_loader = ModelLoader()


if model_names is None:
	# use all model names if no model names were given
	model_names = model_loader.get_model_byname.keys()
else:
	# convert model names to list of strings
	model_names = model_names.split()

# print progress
print ("extracting and saving features for the following models:\n{}\n".format(model_names))

# read data
with open('data/PLoSCB2014_data_20141216/NeuralData_IT_multiunits.pkl','rb') as f:
	data = pickle.load(f)

# read input images
X = misc.read_images(image_paths=data['image_paths'])

# loop through all model_names
for model_name in model_names:

	# get model
	model = model_loader.load(model_name).to(misc.device)

	# extract features
	model_features = misc.extract_features(X,model)

	# print progress
	print ("model: {}\t feature_shape: {}".format(model_name,model_features.shape))

	# save model features
	model_feat_filename = 'models/model_features/'+model_name+'_feat.pkl'
	with open(model_feat_filename,'wb') as f:
		pickle.dump(model_features,f)

	# print progress
	print ("saved {} model features to {}\n".format(model_name,model_feat_filename))