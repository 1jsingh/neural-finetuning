# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

# os, scipy
import os
import scipy

# sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# ignore convergence warnings from sklearn
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


# pickle
import pickle

# PIL
from PIL import Image

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_images(image_paths,data_path='data/PLoSCB2014_data_20141216'):
    '''
    Read and preprocess cadieu dataset images as tensors
    '''

    # define normalize transform to be used while feeding images to the pretrained CNN
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # combine of transforms in a composition
    transform = transforms.Compose([
                transforms.Resize(size=(224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    # preprocessed input images list
    X = []

    for img_path in image_paths:
        img = transform(Image.open(os.path.join(data_path,img_path)))
        X.append(img)

    # convert the list into a tensor
    X = torch.stack(X)

    return X


def get_train_val_split(total_num_imgs=1960,val_ratio=0.2):
    '''
    Inputs:
        total_num_imgs: number of images in the dataset
        val_ratio: fraction of total images to be used in the validation set

    Outputs:
        train_mask: (total_num_imgs,) numpy array, with train_mask[index]=1 for training images and 0 otherwise
        val_mask: (total_num_imgs,) numpy array, with val_mask[index]=1 for validation images and 0 otherwise
    '''
    # number of objects
    num_obj = 49
    
    # number of images per object
    num_imgs_per_obj = int(total_num_imgs/num_obj)
    
    # number of validation images for each object according to val_ratio
    num_val_imgs_per_obj = int(num_imgs_per_obj*val_ratio)
    
    # compute validation mask s.t. val_mask = 1 for validation images and o/w 0
    val_mask = np.zeros(total_num_imgs,dtype=int)
    
    for obj_count in range(num_obj):
        choose = np.random.choice(range(num_imgs_per_obj),num_val_imgs_per_obj,replace=False)
        choose += num_imgs_per_obj*obj_count
        val_mask[choose] = 1
    
    # compute train mask as inverse of val_mask
    train_mask = (val_mask==0).astype(int)
    
    return train_mask,val_mask


def get_train_val_split_indices(total_num_imgs=1960,val_ratio=0.2):
    '''
    Inputs:
        total_num_imgs: number of images in the dataset
        val_ratio: fraction of total images to be used in the validation set

    Outputs:
        train_mask: (total_num_imgs*(1-val_ratio),) numpy array, with indices for training data
        val_mask: (total_num_imgs*val_ratio,) numpy array, with indices for validation data
    '''
    # number of objects
    num_obj = 49
    
    # number of images per object
    num_imgs_per_obj = int(total_num_imgs/num_obj)
    
    # number of validation images for each object according to val_ratio
    num_val_imgs_per_obj = int(num_imgs_per_obj*val_ratio)
    
    # validation mask with indices for validation data
    val_mask = []
    
    for obj_count in range(num_obj):
        choose = np.random.choice(range(num_imgs_per_obj),num_val_imgs_per_obj,replace=False)
        choose += num_imgs_per_obj*obj_count
        val_mask.append(choose)
    
    # concatenate validation indices for each object
    val_mask = np.concatenate(val_mask)
    
    # compute train mask
    train_mask = np.array([x for x in range(total_num_imgs) if x not in val_mask])

    return train_mask,val_mask


def extract_features(X,model,batch_size=8,mode='eval'):
    '''
    Inputs:
        X: tensor images of shape (N,C=3,H=224,W=224)
        model: pytorch model used for extracting features

    Outputs:
        model_features: tensor of shape (N,D)
    '''

    # total number of images
    total_num_images = X.shape[0]

    # check if batch size is appropriate
    assert total_num_images%batch_size == 0

    # number of batches
    num_batches = total_num_images//batch_size

    # model features
    model_features = []

    # set model to eval mode
    model.eval()

    for i in range(num_batches):
        # sample batch and put to device
        x_batch = X[i*batch_size:(i+1)*batch_size].float().to(device)
        
        # extract features for the batch
        out = model(x_batch)
        
        # detach output and convert to numpy
        out = out.detach().cpu().numpy()
        
        # store model features for the current batch
        model_features.append(out)

    # concatenate the model features for all batches
    model_features = np.concatenate(model_features).squeeze()

    return model_features
    

def get_corr_matrix(X):
    '''
    Input:
        X: num_obj,D tensor
    
    Output:
        rdm: num_obj,num_obj tensor
    '''
    # make X zero mean
    X_ = X - torch.mean(X,dim=1).unsqueeze(1)
    
    # compute Covariance matrix
    cov_matrix = torch.matmul(X_,X_.transpose(0,1))/X.shape[1]
    
    # get standard deviations for each of the dimensions
    std_devs = torch.std(X,dim=1,unbiased=False).unsqueeze(dim=1)
    
    # get normalizing standard deviation product matrix
    std_matrix = torch.matmul(std_devs,std_devs.transpose(0,1))
    
    # compute correlation matrix
    corr_matrix = cov_matrix/std_matrix
    
    return corr_matrix


def get_rdm(features):
    '''
    Input:
        features: N,D numpy array
    
    Output:
        rdm: num_obj,num_obj numpy array
    '''

    # number of objects
    num_obj = 49
    
    # number of images per object
    num_imgs_per_obj = int(features.shape[0]/num_obj)
    
    # compute avg features per object
    avg_features = np.zeros((num_obj,features.shape[1]))
    for i in range(num_obj):
        avg_features[i] = np.mean(features[i*num_imgs_per_obj:(i+1)*num_imgs_per_obj],axis=0)
    
    # compute correlation matrix
    correlation_matrix = np.corrcoef(avg_features)
    
    # compute rdm matrix
    rdm = 1 - correlation_matrix
    
    return rdm


def get_rdm_tensor(features):
    '''
    Input:
        features: N,D tensor
    
    Output:
        rdm: num_obj,num_obj tensor
    '''
    
    # number of objects
    num_obj = 49
    
    # number of images per object
    num_imgs_per_obj = int(features.shape[0]/num_obj)
    
    # compute avg features per object
    avg_features = torch.zeros((num_obj,features.shape[1])).float().to(device)
    for i in range(num_obj):
        avg_features[i] = torch.mean(features[i*num_imgs_per_obj:(i+1)*num_imgs_per_obj],dim=0)
    
    # compute correlation matrix
    correlation_matrix = get_corr_matrix(avg_features)
    
    # compute rdm matrix
    rdm = 1 - correlation_matrix
    
    return rdm


def noise_correction(model_features,neural_features):
    '''
    Inputs:
        model_features: (N,D) model features numpy array
        neural_features: (N,Dn) neural features numpy array

    Outputs:
        model_features: (N,D) model features numpy array after noise correction
    '''

    # noise variance estimation parameters for multiunit
    a = 0.14
    b = 0.92
    
    # number of trials
    T = 47
    
    # estimated noise variance
    noise_var = np.mean((a*neural_features+b)**2)/T
    
    # total variance of signal + noise in the neural features
    sig_noise_var = np.var(neural_features)
    
    # expected signal variance for model representations
    expected_sig_var = sig_noise_var - noise_var
    
    # scaling model representations to match expected signal variance
    model_features = np.sqrt(expected_sig_var/np.var(model_features))*model_features
    
    # adding noise to model representations
    noise = np.random.randn(model_features.shape[0],model_features.shape[1])
    noise = (a*model_features+b)*noise/np.sqrt(T)
    model_features += noise
    
    return model_features


def sit_score(model_features,neural_features,num_val_splits=100,val_ratio=0.2):
    # store sit scores for different validation splits
    sit_scores = []

    for i in range(num_val_splits):
        # get validation split
        _,val_mask = get_train_val_split_indices(total_num_imgs=model_features.shape[0],val_ratio=val_ratio)
        
        # get model and neural features for validation images
        val_model_features = model_features[val_mask]
        val_neural_features = neural_features[val_mask]
        
        # apply noise correction using validation model and neural features
        val_model_features = noise_correction(val_model_features,val_neural_features)
        
        # compute RDM matrices for neural and model representations
        rdm_neural = get_rdm(val_neural_features)
        rdm_model = get_rdm(val_model_features)
        
        # get upper triangular matrix values for model and neural rdm
        iu1 = np.triu_indices(49,k=1)
        rdm_neural_triu = rdm_neural[iu1]
        rdm_model_triu = rdm_model[iu1]
        
        # compute sit score and store the result
        sit_score = scipy.stats.spearmanr(rdm_model_triu,rdm_neural_triu).correlation
        sit_scores.append(sit_score)

    return np.mean(sit_scores),np.std(sit_scores)

def linear_svm_score(features,labels,neural_features=None,num_subsampled_feat=168,num_val_splits=10,num_feat_samples=10):
    '''
    Inputs:
        features: (N,D) numpy features array
        labels: (N,) numpy array with int class labels
        neural_features: if not None, is used to perform noise correction to features
        num_subsampled_feat: number of feature dimensions to be sampled for determining accuracy
        num_val_splits: number of train/val/test splits for evaluating svm accuracy score
        num_feat_samples: number of 'num_subsampled_feat' subsamplings of feature space

    Outputs:
        acc_mean: mean accuracy over different train/val/test splits and feature subsampling
        acc_std: std of accuracy over different train/val/test splits and feature subsampling
    '''

    # check if using all feature dimensions
    if num_subsampled_feat == -1:
        num_subsampled_feat = features.shape[1]
        num_feat_samples = 1

    # accuracy scores
    acc_scores = []
    
    # apply noise correction
    if neural_features is not None:
        features = noise_correction(features,neural_features)
        
    for _ in range(num_val_splits):
        # get train:test split with ratio 80:20
        train_mask,test_mask = get_train_val_split_indices(features.shape[0],val_ratio=0.2)

        # get training and test datasets
        X_train,y_train = features[train_mask],labels[train_mask]
        X_test,y_test = features[test_mask],labels[test_mask]
        
        # get train:val split with ratio 80:20
        train_mask,val_mask = get_train_val_split_indices(X_train.shape[0],val_ratio=0.2)

        # get training and val datasets
        X_val,y_val = X_train[val_mask],y_train[val_mask]
        X_train,y_train = X_train[train_mask],y_train[train_mask]
        
        
        for i in range(num_feat_samples):
            # get a subsample
            feat_subsample = np.random.choice(range(features.shape[1]),num_subsampled_feat,replace=False)
            
            # get subsampled train,validation and test datasets
            X_train_subsample = X_train[:,feat_subsample]
            X_val_subsample = X_val[:,feat_subsample]
            X_test_subsample = X_test[:,feat_subsample]
            
            # range to sample regularization parameter C
            C_range = [1e-3,1e-2,1e-1,1e0,1e1,1e2]
            
            # store val acc scores to choose the best C
            val_acc_scores = []
            
            for C in C_range:
                # linear SVM classifier
                clf = LinearSVC(C=C,max_iter=1000)

                # fit training data
                clf.fit(X_train_subsample,y_train)

                # get mean accuracy on validation data
                val_acc = clf.score(X_val_subsample,y_val)
                
                # store val_acc
                val_acc_scores.append(val_acc)
            
            # choose best C
            best_C = C_range[np.argmax(val_acc_scores)]
            
            # linear SVM classifier for best C
            clf = LinearSVC(C=best_C,max_iter=5000)

            # fit training data
            clf.fit(X_train_subsample,y_train)
            
            # get mean accuracy on test data
            test_acc = clf.score(X_test_subsample,y_test)
            
            # store accuracy
            acc_scores.append(test_acc)
    
    return np.mean(acc_scores),np.std(acc_scores)


def linear_svm_score_v2(features,labels,neural_features=None,num_subsampled_feat=168,num_feat_samples=10,num_val_splits=10,val_ratio=0.2):
    '''
    Inputs:
        features: (N,D) numpy features array
        labels: (N,) numpy array with int class labels
        neural_features: if not None, is used to perform noise correction to features
        num_subsampled_feat: number of feature dimensions to be sampled for determining accuracy
        num_val_splits: number of train/val/test splits for evaluating svm accuracy score
        num_feat_samples: number of 'num_subsampled_feat' subsamplings of feature space
        val_ratio: ratio of train and validation sets while performing cross validation

    Outputs:
        acc_mean: mean accuracy over different train/val/test splits and feature subsampling
        acc_std: std of accuracy over different train/val/test splits and feature subsampling
    '''

    # check if using all feature dimensions
    if num_subsampled_feat == -1:
        num_subsampled_feat = features.shape[1]
        num_feat_samples = 1

    # accuracy scores
    acc_scores = []
    
    # apply noise correction
    if neural_features is not None:
        features = noise_correction(features,neural_features)
    
    # get cross validation indices
    cv = []
    for i in range(num_val_splits):
        train_indices,test_indices = get_train_val_split_indices(total_num_imgs=features.shape[0],val_ratio=val_ratio)
        cv.append((train_indices,test_indices))

    # range to sample regularization parameter C
    parameters = {'C':[1e-3,1e-2,1e-1,1e0,1e1,1e2]}
    svc = LinearSVC()
    clf = GridSearchCV(svc, parameters, cv=cv,n_jobs=-1)

    # perform cross validation
    clf.fit(features, labels)
    
    # get acc mean and standard deviation for best choice of hyperparameters
    acc_mean = clf.best_score_
    acc_std = clf.cv_results_['std_test_score'][clf.best_index_]

    return acc_mean,acc_std