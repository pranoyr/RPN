from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from datasets.pascal_voc import VOCDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor, RoIHeads 
import os
from torch.jit.annotations import List, Tuple, Dict, Optional
from PIL import Image
import cv2

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


class_names = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
								'bottle', 'bus', 'car', 'cat', 'chair',
								'cow', 'diningtable', 'dog', 'horse',
								'motorbike', 'person', 'pottedplant',
								'sheep', 'sofa', 'train', 'tvmonitor')


class FasterRCNN(nn.Module):
	def __init__(self):
		super(FasterRCNN, self).__init__()
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
		box_score_thresh=0.5
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
		

		images, targets = self.transform(images, targets)
		# fpn_feature_maps = self.fpn(images.tensors.cuda())
		fpn_feature_maps = self.fpn(images.tensors)
		
		

		if self.training:
			proposals, proposal_losses = self.rpn(images, fpn_feature_maps, targets)
			detections, detector_losses = self.roi_heads(fpn_feature_maps, proposals, images.image_sizes, targets)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
			losses = {}
			losses.update(detector_losses)
			losses.update(proposal_losses)
		else:
			proposals, losses = self.rpn(images, fpn_feature_maps)
			detections, detector_losses = self.roi_heads(fpn_feature_maps, proposals, images.image_sizes)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
		return detections, losses


faster_rcnn = FasterRCNN()

# load pretrained weights
# checkpoint = torch.load('./snapshots/faster_rcnn_custom.pth', map_location='cpu')
checkpoint = torch.load('/Users/pranoyr/Downloads/faster_rcnn_custom.pth', map_location='cpu')
faster_rcnn.load_state_dict(checkpoint['state_dict'])
print("Model Restored")

faster_rcnn.eval()


im = Image.open('/Users/pranoyr/Downloads/err.jpg')
img = np.array(im)
draw = img.copy()
# draw = cv2.resize(draw,(1344,768))
img = torch.from_numpy(img)
img = img.permute(2,0,1)
img = img.type(torch.float32)

detections, losses = faster_rcnn([img])
boxes = detections[0]['boxes']
scores = detections[0]['scores']
labels =  detections[0]['labels']
print(scores.shape)

print(boxes.shape)
for i in range(boxes.size(0)):
	box = boxes[i]
	score = scores[i]
	label = f"{class_names[labels[i]]}: {scores[i]:.2f}"
	cv2.rectangle(draw, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
	label = f"""{class_names[labels[i]]}: {scores[i]:.2f}"""
	label = f"{class_names[labels[i]]}: {scores[i]:.2f}"
	cv2.putText(draw, label,
				(box[0] + 20, box[1] + 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,  # font scale
				(255, 0, 255),
				2)  # line type
path = "./results/faster_rcnn_sample.jpg"
cv2.imwrite(path, draw)

# boxes = boxes.type(torch.int)
# for box in boxes:
# 	cv2.rectangle(draw, (box[0].item(),  box[1].item())  ,(box[2].item(),  box[3].item()), (255, 255, 0), 4)
# cv2.imwrite('a.jpg', draw)


