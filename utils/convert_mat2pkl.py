'''
Convert and save matlab data files to organized python pickle data
'''

# numpy
import numpy as np
# pandas
import pandas as pd
# read matlab data files
from scipy.io import loadmat
# pickle
import pickle
import argparse

# defining the argument parser
parser = argparse.ArgumentParser(description='Convert and save matlab data files to organized python pickle data')
parser.add_argument('-f','--filename', type=str ,help='matlab data file',required = True)



args = parser.parse_args()

# read data
print ("reading {}".format(args.filename))
data = loadmat(args.filename)

# coverting meta data into organized format
meta_data = data['meta']

# image paths
image_paths = []

# image_categories
image_ctg_name = []

# 10 splits with ratio 80:20 
image_splits = []

for image_num in range(len(meta_data)):
    image_data = meta_data[image_num].split()
    image_paths.append(image_data[0])
    image_ctg_name.append(image_data[1])
    image_splits.append(list(map(int,image_data[2:])))

    
image_splits = np.array(image_splits)

# get category name to int mapping
ctg_list = sorted(list(set(image_ctg_name)))
mapping = dict(zip(ctg_list,range(len(ctg_list))))
inv_mapping =  dict(zip(range(len(ctg_list)),ctg_list))

# convert category names to int data
image_ctg = np.array([mapping[_] for _ in image_ctg_name])

# neural feature
features = data['features']

# put neural data in organized format
neural_data = {}

neural_data['image_paths'] = image_paths
neural_data['image_ctg'] = image_ctg
neural_data['image_splits'] = image_splits
neural_data['features'] = features
neural_data['categ_name_map'] = inv_mapping

# get base filename
base_filename = args.filename.split('.mat')[0]

# save the neural data dict
print ("saving neural data to {}".format(base_filename+'.pkl'))
with open(base_filename+'.pkl','wb') as f:
    pickle.dump(neural_data,f)