from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from dataset import VOCDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
import os

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
from fpn import resnet101


dataset_train = VOCDataset(root='/home/neuroplex/data/VOCdevkit/VOC2007')
# dataset_train = VOCDataset(root='/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007')
dataloader = DataLoader(
	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
		images, targets = self.transform(images, targets)
		fpn_feature_maps = self.fpn(images.tensors.to(DEVICE))
		# fpn_feature_maps = OrderedDict(
		#     {i: index for i, index in enumerate(fpn_feature_maps)})
		
		# fpn_feature_maps = OrderedDict([('0', fpn_feature_maps)])

		if self.training:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
		return boxes, losses


rpn = RPN().to(DEVICE)
optimizer = optim.Adam(rpn.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, patience=3, verbose=True)
# x = torch.Tensor(2, 3, 224, 224)
# boxes, losses = rpn(x)
# print(boxes)
# print(losses)
n_epochs = 100
rpn.train()

for epoch in range(1, n_epochs+1):
	loss = []
	for i, data in enumerate(dataloader):
		images, annotations = data
		boxes, losses = rpn(images, annotations)
		final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
		loss.append(final_loss.item())

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()
		print(f'loss : {final_loss.item()},\n\
				cls_loss : {losses["loss_objectness"].item()},\n\
				reg_loss : {losses["loss_rpn_box_reg"].item()}')

	loss = torch.tensor(loss, dtype=torch.float32)
	print(f'loss : {torch.mean(loss)}')
	# scheduler.step(torch.mean(loss))


	state = {'state_dict': rpn.state_dict()}
	torch.save(state, os.path.join('./snapshots', f'rpn.pth'))
	print("model saved")


