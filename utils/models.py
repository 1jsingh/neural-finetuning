# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import utils.resnet as resnet

# Fire class for squeezenet models
from torchvision.models.squeezenet import Fire

# Identity function for pytorch
def Identity():
    return nn.Sequential()

# Alexnet
class Alexnet_partial(nn.Module):
    def __init__(self):
        super(Alexnet_partial, self).__init__()
        self.model = torchvision.models.alexnet(pretrained=True)
        modules = list(self.model.classifier.children())[:-2]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x


# VGG 11,13,16,19 models with and without batch normalization
class vgg11_partial(nn.Module):
    def __init__(self):
        super(vgg11_partial, self).__init__()
        self.model = torchvision.models.vgg11(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg11_bn_partial(nn.Module):
    def __init__(self):
        super(vgg11_bn_partial, self).__init__()
        self.model = torchvision.models.vgg11_bn(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg13_partial(nn.Module):
    def __init__(self):
        super(vgg13_partial, self).__init__()
        self.model = torchvision.models.vgg13(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg13_bn_partial(nn.Module):
    def __init__(self):
        super(vgg13_bn_partial, self).__init__()
        self.model = torchvision.models.vgg13_bn(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg16_partial(nn.Module):
    def __init__(self):
        super(vgg16_partial, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg16_bn_partial(nn.Module):
    def __init__(self):
        super(vgg16_bn_partial, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x
        
class vgg19_partial(nn.Module):
    def __init__(self):
        super(vgg19_partial, self).__init__()
        self.model = torchvision.models.vgg19(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x

class vgg19_bn_partial(nn.Module):
    def __init__(self):
        super(vgg19_bn_partial, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=True)
        modules = list(self.model.classifier.children())[:-3]
        new_classifier = nn.Sequential(*modules)
        self.model.classifier = new_classifier
        
    def forward(self, x):
        x = self.model(x)
        return x


# Resnet 18,34,50,101,152 models
class resnet18_partial(nn.Module):
    def __init__(self):
        super(resnet18_partial, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        #self.model = resnet.resnet18(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet34_partial(nn.Module):
    def __init__(self):
        super(resnet34_partial, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        #self.model = resnet.resnet34(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet50_partial(nn.Module):
    def __init__(self):
        super(resnet50_partial, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        #self.model = resnet.resnet50(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet101_partial(nn.Module):
    def __init__(self):
        super(resnet101_partial, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        #self.model = resnet.resnet101(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

class resnet152_partial(nn.Module):
    def __init__(self):
        super(resnet152_partial, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

# Densenet 121,161,169,201 models
class densenet121_partial(nn.Module):
    def __init__(self):
        super(densenet121_partial, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class densenet161_partial(nn.Module):
    def __init__(self):
        super(densenet161_partial, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class densenet169_partial(nn.Module):
    def __init__(self):
        super(densenet169_partial, self).__init__()
        self.model = torchvision.models.densenet169(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

class densenet201_partial(nn.Module):
    def __init__(self):
        super(densenet201_partial, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x

# Inception_v3
class Inceptionv3_partial(nn.Module):
    def __init__(self):
        super(Inceptionv3_partial, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=True)
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model(x)
        return x

# Squeezenet v0,v1 models
class Squeezenetv0_partial(nn.Module):
    def __init__(self):
        super(Squeezenetv0_partial, self).__init__()
        self.model = torchvision.models.squeezenet1_0(pretrained=True)
        # modules = list(self.model.features.children())

        # # recreate last block without relu
        # modules[-1].expand1x1_activation =  Identity()
        # modules[-1].expand3x3_activation =  Identity()

        # self.model.features = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.model.features(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=13).view(x.size(0), -1)
        return x

class Squeezenetv1_partial(nn.Module):
    def __init__(self):
        super(Squeezenetv1_partial, self).__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        
    def forward(self, x):
        x = self.model.features(x)
        #x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=13).view(x.size(0), -1)
        return x