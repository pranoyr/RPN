from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from dataset import VOCDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor, RoIHeads 
import os
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
from fpn import resnet101


# class FPN(nn.Module):
#     """ FPN """

#     def __init__(self):
#         super(FPN, self).__init__()
#         resnet = resnet101(pretrained=True)

#         # if self.pretrained == True:
#         #     print("Loading pretrained weights from %s" %(self.model_path))
#         #     state_dict = torch.load(self.model_path)
#         #     resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

#         self.RCNN_layer0 = nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
#         self.RCNN_layer1 = nn.Sequential(resnet.layer1)
#         self.RCNN_layer2 = nn.Sequential(resnet.layer2)
#         self.RCNN_layer3 = nn.Sequential(resnet.layer3)
#         self.RCNN_layer4 = nn.Sequential(resnet.layer4)
#         self.maxpool2d = nn.MaxPool2d(1, stride=2)

#         # Top layer
#         self.RCNN_toplayer = nn.Conv2d(
#             2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

#         # Smooth layers
#         self.RCNN_smooth1 = nn.Conv2d(
#             256, 256, kernel_size=3, stride=1, padding=1)
#         self.RCNN_smooth2 = nn.Conv2d(
#             256, 256, kernel_size=3, stride=1, padding=1)
#         self.RCNN_smooth3 = nn.Conv2d(
#             256, 256, kernel_size=3, stride=1, padding=1)

#         # Lateral layers
#         self.RCNN_latlayer1 = nn.Conv2d(
#             1024, 256, kernel_size=1, stride=1, padding=0)
#         self.RCNN_latlayer2 = nn.Conv2d(
#             512, 256, kernel_size=1, stride=1, padding=0)
#         self.RCNN_latlayer3 = nn.Conv2d(
#             256, 256, kernel_size=1, stride=1, padding=0)

#     def _init_weights(self):
#         def normal_init(m, mean, stddev, truncated=False):
#             """
#             weight initalizer: truncated normal and random normal.
#             """
#             # x is a parameter
#             if truncated:
#                 m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
#                     mean)  # not a perfect approximation
#             else:
#                 m.weight.data.normal_(mean, stddev)
#                 m.bias.data.zero_()

#         normal_init(self.RCNN_toplayer, 0, 0.01, False)
#         normal_init(self.RCNN_smooth1, 0, 0.01, False)
#         normal_init(self.RCNN_smooth2, 0, 0.01, False)
#         normal_init(self.RCNN_smooth3, 0, 0.01, False)
#         normal_init(self.RCNN_latlayer1, 0, 0.01, False)
#         normal_init(self.RCNN_latlayer2, 0, 0.01, False)
#         normal_init(self.RCNN_latlayer3, 0, 0.01, False)

#     def create_architecture(self):
#         self._init_modules()
#         self._init_weights()

#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
#         _, _, H, W = y.size()
#         return F.upsample(x, size=(H, W), mode='bilinear') + y

#     def forward(self, im_data):
#         # feed image data to base model to obtain base feature map
#         # Bottom-up
#         c1 = self.RCNN_layer0(im_data)
#         c2 = self.RCNN_layer1(c1)
#         c3 = self.RCNN_layer2(c2)
#         c4 = self.RCNN_layer3(c3)
#         c5 = self.RCNN_layer4(c4)
#         # Top-down
#         p5 = self.RCNN_toplayer(c5)
#         p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
#         p4 = self.RCNN_smooth1(p4)
#         p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
#         p3 = self.RCNN_smooth2(p3)
#         p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
#         p2 = self.RCNN_smooth3(p2)

#         p6 = self.maxpool2d(p5)

#         rpn_feature_maps = [p2, p3, p4, p5, p6]
#         return rpn_feature_maps
		# mrcnn_feature_maps = [p2, p3, p4, p5]


dataset_train = VOCDataset(root='/home/neuroplex/data/VOCdevkit/VOC2007')
# dataset_train = VOCDataset(root='/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007')
dataloader = DataLoader(
	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)


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
		
		# Box parameters
		box_roi_pool=None
		box_head=None
		box_predictor=None
		box_score_thresh=0.05
		box_nms_thresh=0.5
		box_detections_per_img=100
		box_fg_iou_thresh=0.5
		box_bg_iou_thresh=0.5
		box_batch_size_per_image=512
		box_positive_fraction=0.25
		bbox_reg_weights=None
		num_classes=21

		if box_roi_pool is None:
			box_roi_pool = MultiScaleRoIAlign(
				featmap_names=['0', '1', '2', '3'],
				output_size=7,
				sampling_ratio=2)

		if box_head is None:
			resolution = box_roi_pool.output_size[0]
			representation_size = 1024
			box_head = TwoMLPHead(
				256 * resolution ** 2,
				representation_size)

		if box_predictor is None:
			representation_size = 1024
			box_predictor = FastRCNNPredictor(
				representation_size,
				num_classes)

		self.roi_heads = RoIHeads(
			# Box
			box_roi_pool, box_head, box_predictor,
			box_fg_iou_thresh, box_bg_iou_thresh,
			box_batch_size_per_image, box_positive_fraction,
			bbox_reg_weights,
			box_score_thresh, box_nms_thresh, box_detections_per_img)

	def forward(self, images, targets=None):
			
		original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
		for img in images:
			val = img.shape[-2:]
			assert len(val) == 2
			original_image_sizes.append((val[0], val[1]))
		

		# images, targets = self.transform(images, targets)
		fpn_feature_maps = self.fpn(images.tensors.cuda())
		fpn_feature_maps = self.fpn(images.tensors)
		
		

		if self.training:
			proposals, proposal_losses = self.rpn(images, fpn_feature_maps, targets)
			detections, detector_losses = self.roi_heads(fpn_feature_maps, proposals, images.image_sizes, targets)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
			losses = {}
			losses.update(detector_losses)
			losses.update(proposal_losses)
		else:
			detections, losses = self.rpn(images, fpn_feature_maps)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
		return detections, losses


rpn = RPN().cuda()
# rpn = RPN()
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
		print(losses)
		final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"] + \
		losses['loss_box_reg'] + losses['loss_classifier']
		loss.append(final_loss.item())

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()
		print(f'RCNN_Loss    : {final_loss.item()},\n\
				rpn_cls_loss : {losses["loss_objectness"].item()},\n\
				rpn_reg_loss : {losses["loss_rpn_box_reg"].item()}\n\
				box_loss 	 : {losses["loss_box_reg"]}\n\
				cls_loss     : {losses["loss_classifier"]}'
				)

	loss = torch.tensor(loss, dtype=torch.float32)
	print(f'Total Loss : {torch.mean(loss)}')
	# scheduler.step(torch.mean(loss))


	state = {'state_dict': rpn.state_dict()}
	torch.save(state, os.path.join('./snapshots', f'rpn.pth'))
	print("model saved")


