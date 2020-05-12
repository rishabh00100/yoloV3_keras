from __future__ import division
import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, LeakyReLU, UpSampling2D, Layer, Input
from keras.layers.merge import add, concatenate
import tensorflow as tf
import numpy as np
from pprint import pprint

def parse_cfg(cfgfile):
	"""
	Takes a configuration file
	
	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list
	
	"""

	file = open(cfgfile, 'r')
	lines = file.read().split('\n')                        # store the lines in a list
	lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
	lines = [x for x in lines if x[0] != '#']              # get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces

	block = {}
	blocks = []

	for line in lines:
		if line[0] == "[":               # This marks the start of a new block
			if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
				blocks.append(block)     # add it the blocks list
				block = {}               # re-init the block
			block["type"] = line[1:-1].rstrip()     
		else:
			key,value = line.split("=") 
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks

class EmptyLayer(Layer):
	def __init__(self):
		super(EmptyLayer, self).__init__()

class DetectionLayer(Layer):
	def __init__(self, anchors):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors

def create_modules(blocks):
	net_info = blocks[0]
	module_list = []
	layer_list = []
	prev_filters = 3
	output_filters = []
	inp = Input(shape=(None, None, 3))
	for index, x in enumerate(blocks[1:]):
		print("index", index)
		print("X", x)
		# module = Sequential()
		X = inp
		try:
			if x.get("type") == "convolutional":
				activation = x.get("activation")
				batch_normalize = int(x.get("batch_normalize", "0"))
				if batch_normalize == 1:
					bias = False
				else:
					bias = True
				filters = int(x.get("filters"))
				stride = int(x.get("stride"))
				padding = int(x.get("pad"))
				kernel_size = int(x.get("size"))

				if padding == 1:
					pad = "valid"
				else:
					pad = "same"

				# Add the convolution layer
				X = Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), padding=pad, use_bias=bias, name="conv_{}".format(str(index)))(X)
				# module.add(conv)

				if batch_normalize == 1:
					X = BatchNormalization(name="bnorm_{}".format(str(index)))(X)
					# module.add(bn)

				if activation == "leaky":
					X = LeakyReLU(name="leaky_{}".format(str(index)))(X)
				elif activation == "linear":
					X = Activation(activation,  name="linear_{}".format(str(index)))(X)
				layer_list.append(X)
				# module.add(X)

			elif x.get("type") == "upsample":
				stride = int(x.get("stride"))
				X = UpSampling2D(size = stride, interpolation = "bilinear")(X)
				layer_list.append(X)
				# module.add(upsample)

			elif x.get("type") == "route":
				x["layers"] = x["layers"].split(',')
				#Start  of a route
				start = int(x["layers"][0])
				#end, if there exists one.
				try:
					end = int(x["layers"][1])
				except:
					end = 0
				#Positive anotation
				if start > 0: 
					start = start - index
				if end > 0:
					end = end - index
				# route = EmptyLayer()
				# module.add(route)
				if end < 0:
					filters = output_filters[index + start] + output_filters[index + end]
				else:
					filters= output_filters[index + start]
				print("start", start)
				print("end", end)
				X = concatenate([X, layer_list[end]])

			elif x.get("type") == "shortcut":
				shortcut = add([layer_list[int(x.get("from"))], layer_list[-1]])
				layer_list.append(shortcut)
				# module.add(shortcut)

			elif x.get("type") == "yolo":
				mask = x["mask"].split(",")
				mask = [int(x) for x in mask]

				anchors = x["anchors"].split(",")
				anchors = [int(a) for a in anchors]
				anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
				anchors = [anchors[i] for i in mask]

				detection = DetectionLayer(anchors)
				layer_list.append(detection)
				# module.add(detection)

			# module_list.append(module)
			prev_filters = filters
			output_filters.append(filters)
		except Exception as e:
			pprint(index)
			pprint(x)
			raise(e)

	return (net_info, module_list)

class Darknet():
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
	    modules = self.blocks[1:]
	    outputs = {}   #We cache the outputs for the route layer


if __name__ == "__main__":
	cfgfile = "cfg/yolov3.cfg"
	blocks = parse_cfg(cfgfile)
	pprint(create_modules(blocks))