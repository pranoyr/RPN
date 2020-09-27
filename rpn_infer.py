from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from PIL import Image
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from torch.jit.annotations import List, Tuple, Dict, Optional
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
import time
import pdb
from torchvision.models.resnet import resnet101
import torchvision
import math
import cv2



def resize_boxes(boxes, original_size, new_size):
	ratios = [
		torch.tensor(s, dtype=torch.float32, device=boxes.device) /
		torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
		for s, s_orig in zip(new_size, original_size)
	]
	ratio_height, ratio_width = ratios
	xmin, ymin, xmax, ymax = boxes.unbind(1)


	xmin = xmin /ratio_width
	xmax = xmax / ratio_width
	ymin = ymin / ratio_height
	ymax = ymax / ratio_height
	return torch.stack((xmin, ymin, xmax, ymax), dim=1)



class RPN(nn.Module):
	def __init__(self):
		super(RPN, self).__init__()
		# Define FPN
		self.fpn = resnet_fpn_backbone(backbone_name='resnet101', pretrained=True)
		anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
		aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
		# Generate anchor boxes
		anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
		# Define RPN Head
		# rpn_head = RPNHead(256, 9)
		rpn_head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
		# RPN parameters,
		rpn_pre_nms_top_n_train = 2000
		rpn_pre_nms_top_n_test = 1000
		rpn_post_nms_top_n_train = 2000
		rpn_post_nms_top_n_test = 1000
		rpn_nms_thresh = 0.7
		# rpn_nms_thresh = 0.45
		rpn_fg_iou_thresh = 0.7
		rpn_bg_iou_thresh = 0.3
		rpn_batch_size_per_image = 256
		rpn_positive_fraction = 0.5

		# transform parameters
		min_size = 800
		max_size = 1333
		image_mean = [0.485, 0.456, 0.406]
		image_std = [0.229, 0.224, 0.225]
		self.transform = GeneralizedRCNNTransform(
			min_size, max_size, image_mean, image_std)

		rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
								 testing=rpn_pre_nms_top_n_test)
		rpn_post_nms_top_n = dict(
			training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

		# Create RPN
		self.rpn = RegionProposalNetwork(
			anchor_generator, rpn_head,
			rpn_fg_iou_thresh, rpn_bg_iou_thresh,
			rpn_batch_size_per_image, rpn_positive_fraction,
			rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

	def forward(self, images, targets=None):
		# l = torch.FloatTensor([[1,2,3,4],[1,2,3,4]])
		# targets = [{"boxes":l},{"boxes":l}]
		# targets = [{i: index for i, index in enumerate(l)}]
		original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
		for img in images:
			val = img.shape[-2:]
			assert len(val) == 2
			original_image_sizes.append((val[0], val[1]))

		images, targets = self.transform(images, targets)
		fpn_feature_maps = self.fpn(images.tensors)

	
		# fpn_feature_maps = OrderedDict(
		#     {i: index for i, index in enumerate(fpn_feature_maps)})
		
		# fpn_feature_maps = OrderedDict([('0', fpn_feature_maps)])

		if self.training:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
			for org_size, tar_size in zip(original_image_sizes, images.image_sizes):
				boxes = resize_boxes(boxes[0], org_size, tar_size)
		return boxes, losses


rpn = RPN()


# load pretrained weights
# checkpoint = torch.load('./snapshots/faster_rcnn_custom.pth', map_location='cpu')
checkpoint = torch.load('/Users/pranoyr/Downloads/faster_rcnn_custom.pth', map_location='cpu')
rpn.load_state_dict(checkpoint['state_dict'], strict=False)
print("Model Restored")

rpn.eval()


im = Image.open('/Users/pranoyr/Downloads/err.jpg')
img = np.array(im)
draw = img.copy()
# draw = cv2.resize(draw,(1344,768))
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.type(torch.float32)

boxes, losses = rpn([img])

print(boxes.shape)
boxes = boxes.type(torch.int)
for box in boxes:
	cv2.rectangle(draw, (box[0].item(),  box[1].item())  ,(box[2].item(),  box[3].item()), (255, 255, 0), 4)
cv2.imwrite('./results/rpn_sample.jpg', draw)


