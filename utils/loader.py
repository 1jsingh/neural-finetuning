'''
Load truncated pretrained models (till penultimate layer)
'''

from . import models



class ModelLoader:
	def __init__(self):
		self.get_model_byname = {
		'alexnet': models.alexnet_partial,
		'vgg11': models.vgg11_partial,
		'vgg13': models.vgg13_partial,
		'vgg16': models.vgg16_partial,
		'vgg19': models.vgg19_partial,
		'vgg11_bn': models.vgg11_bn_partial,
		'vgg13_bn': models.vgg13_bn_partial,
		'vgg16_bn': models.vgg16_bn_partial,
		'vgg19_bn': models.vgg19_bn_partial,
		'resnet18':models.resnet18_partial,
		'resnet34':models.resnet34_partial,
		'resnet50':models.resnet50_partial,
		'resnet101':models.resnet101_partial,
		'resnet152':models.resnet152_partial,
		'squeezenet_v0': models.squeezenet_v0_partial,
		'squeezenet_v1': models.squeezenet_v1_partial,
		'densenet121': models.densenet121_partial,
		'densenet161': models.densenet161_partial,
		'densenet169': models.densenet169_partial,
		'densenet201': models.densenet201_partial#,
		#'inception_v3': models.inception_v3_partial
		}



	def load(self,model_name):
		'''
		Load model by model_name

		Inputs:
			model_name: string describing model name
		Outputs:
			model: pytorch model
		'''

		# check if model name is valid
		assert model_name in self.get_model_byname.keys()

		# get model
		model = self.get_model_byname[model_name]()

		return model