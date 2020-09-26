

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from dataset import VOCDataset, collater
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.models.detection._utils as  det_utils
from torchvision.ops import boxes as box_ops
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


# dataset_train = VOCDataset(root='/home/neuroplex/data/VOCdevkit/VOC2007')
dataset_train = VOCDataset(root='/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007')
dataloader = DataLoader(
	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)




class RoIHeads(torch.nn.Module):
	__annotations__ = {
		'box_coder': det_utils.BoxCoder,
		'proposal_matcher': det_utils.Matcher,
		'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
	}

	def __init__(self,
				 box_roi_pool,
				 box_head,
				 box_predictor,
				 # Faster R-CNN training
				 fg_iou_thresh, bg_iou_thresh,
				 batch_size_per_image, positive_fraction,
				 bbox_reg_weights,
				 # Faster R-CNN inference
				 score_thresh,
				 nms_thresh,
				 detections_per_img,
				 # Mask
				 mask_roi_pool=None,
				 mask_head=None,
				 mask_predictor=None,
				 keypoint_roi_pool=None,
				 keypoint_head=None,
				 keypoint_predictor=None,
				 ):
		super(RoIHeads, self).__init__()

		self.box_similarity = box_ops.box_iou
		# assign ground-truth boxes for each proposal
		self.proposal_matcher = det_utils.Matcher(
			fg_iou_thresh,
			bg_iou_thresh,
			allow_low_quality_matches=False)

		self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
			batch_size_per_image,
			positive_fraction)

		if bbox_reg_weights is None:
			bbox_reg_weights = (10., 10., 5., 5.)
		self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

		self.box_roi_pool = box_roi_pool
		self.box_head = box_head
		self.box_predictor = box_predictor

		self.score_thresh = score_thresh
		self.nms_thresh = nms_thresh
		self.detections_per_img = detections_per_img

		self.mask_roi_pool = mask_roi_pool
		self.mask_head = mask_head
		self.mask_predictor = mask_predictor

		self.keypoint_roi_pool = keypoint_roi_pool
		self.keypoint_head = keypoint_head
		self.keypoint_predictor = keypoint_predictor

	def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		matched_idxs = []
		labels = []
		for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

			if gt_boxes_in_image.numel() == 0:
				# Background image
				device = proposals_in_image.device
				clamped_matched_idxs_in_image = torch.zeros(
					(proposals_in_image.shape[0],), dtype=torch.int64, device=device
				)
				labels_in_image = torch.zeros(
					(proposals_in_image.shape[0],), dtype=torch.int64, device=device
				)
			else:
				#  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
				match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
				matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

				clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

				labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
				labels_in_image = labels_in_image.to(dtype=torch.int64)

				# Label background (below the low threshold)
				bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
				labels_in_image[bg_inds] = 0

				# Label ignore proposals (between low and high thresholds)
				ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
				labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

			matched_idxs.append(clamped_matched_idxs_in_image)
			labels.append(labels_in_image)
		return matched_idxs, labels

	def subsample(self, labels):
		# type: (List[Tensor]) -> List[Tensor]
		sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
		sampled_inds = []
		for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
			zip(sampled_pos_inds, sampled_neg_inds)
		):
			img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
			sampled_inds.append(img_sampled_inds)
		return sampled_inds

	def add_gt_proposals(self, proposals, gt_boxes):
		# type: (List[Tensor], List[Tensor]) -> List[Tensor]
		proposals = [
			torch.cat((proposal, gt_box))
			for proposal, gt_box in zip(proposals, gt_boxes)
		]

		return proposals

	def check_targets(self, targets):
		# type: (Optional[List[Dict[str, Tensor]]]) -> None
		assert targets is not None
		assert all(["boxes" in t for t in targets])
		assert all(["labels" in t for t in targets])

	def select_training_samples(self,
								proposals,  # type: List[Tensor]
								targets     # type: Optional[List[Dict[str, Tensor]]]
								):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
		self.check_targets(targets)
		assert targets is not None
		dtype = proposals[0].dtype
		device = proposals[0].device

		gt_boxes = [t["boxes"].to(dtype) for t in targets]
		gt_labels = [t["labels"] for t in targets]

		# append ground-truth bboxes to propos
		proposals = self.add_gt_proposals(proposals, gt_boxes)

		# get matching gt indices for each proposal
		matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
		# sample a fixed proportion of positive-negative proposals
		sampled_inds = self.subsample(labels)
		matched_gt_boxes = []
		num_images = len(proposals)
		for img_id in range(num_images):
			img_sampled_inds = sampled_inds[img_id]
			proposals[img_id] = proposals[img_id][img_sampled_inds]
			labels[img_id] = labels[img_id][img_sampled_inds]
			matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

			gt_boxes_in_image = gt_boxes[img_id]
			if gt_boxes_in_image.numel() == 0:
				gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
			matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

		regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
		return proposals, matched_idxs, labels, regression_targets

	def postprocess_detections(self,
							   class_logits,    # type: Tensor
							   box_regression,  # type: Tensor
							   proposals,       # type: List[Tensor]
							   image_shapes     # type: List[Tuple[int, int]]
							   ):
		# type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
		device = class_logits.device
		num_classes = class_logits.shape[-1]

		boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
		pred_boxes = self.box_coder.decode(box_regression, proposals)

		pred_scores = F.softmax(class_logits, -1)

		pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
		pred_scores_list = pred_scores.split(boxes_per_image, 0)

		all_boxes = []
		all_scores = []
		all_labels = []
		for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
			boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

			# create labels for each prediction
			labels = torch.arange(num_classes, device=device)
			labels = labels.view(1, -1).expand_as(scores)

			# remove predictions with the background label
			boxes = boxes[:, 1:]
			scores = scores[:, 1:]
			labels = labels[:, 1:]

			# batch everything, by making every class prediction be a separate instance
			boxes = boxes.reshape(-1, 4)
			scores = scores.reshape(-1)
			labels = labels.reshape(-1)

			# remove low scoring boxes
			inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
			boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

			# remove empty boxes
			keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
			boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

			# non-maximum suppression, independently done per class
			keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
			# keep only topk scoring predictions
			keep = keep[:self.detections_per_img]
			boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

			all_boxes.append(boxes)
			all_scores.append(scores)
			all_labels.append(labels)

		return all_boxes, all_scores, all_labels

	def forward(self,
				features,      # type: Dict[str, Tensor]
				proposals,     # type: List[Tensor]
				image_shapes,  # type: List[Tuple[int, int]]
				targets=None   # type: Optional[List[Dict[str, Tensor]]]
				):
		# type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
		"""
		Arguments:
			features (List[Tensor])
			proposals (List[Tensor[N, 4]])
			image_shapes (List[Tuple[H, W]])
			targets (List[Dict])
		"""
		if targets is not None:
			for t in targets:
				# TODO: https://github.com/pytorch/pytorch/issues/26731
				floating_point_types = (torch.float, torch.double, torch.half)
				assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
				assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
		if self.training:
			proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
			print(labels)
		else:
			labels = None
			regression_targets = None
			matched_idxs = None

		# box_features = self.box_roi_pool(features, proposals, image_shapes)
		# box_features = self.box_head(box_features)
		# class_logits, box_regression = self.box_predictor(box_features)

		# result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
		# losses = {}
		# if self.training:
		#     assert labels is not None and regression_targets is not None
		#     loss_classifier, loss_box_reg = fastrcnn_loss(
		#         class_logits, box_regression, labels, regression_targets)
		#     losses = {
		#         "loss_classifier": loss_classifier,
		#         "loss_box_reg": loss_box_reg
		#     }
		# else:
		#     boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
		#     num_images = len(boxes)
		#     for i in range(num_images):
		#         result.append(
		#             {
		#                 "boxes": boxes[i],
		#                 "labels": labels[i],
		#                 "scores": scores[i],
		#             }
		#         )
		   
		# return result, losses



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
		# fpn_feature_maps = self.fpn(images.tensors.cuda())
		fpn_feature_maps = self.fpn(images.tensors)
		# fpn_feature_maps = OrderedDict(
		#     {i: index for i, index in enumerate(fpn_feature_maps)})
		
		# fpn_feature_maps = OrderedDict([('0', fpn_feature_maps)])

		if self.training:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
		return boxes, losses, fpn_feature_maps, images.image_sizes


# rpn = RPN().cuda()
# Box parameters
box_roi_pool=None
box_head=None
box_predictor=None
box_score_thresh=0.05
box_nms_thresh=0.5
box_detections_per_img=100,
box_fg_iou_thresh=0.5
box_bg_iou_thresh=0.5
box_batch_size_per_image=512
box_positive_fraction=0.25
bbox_reg_weights=None

rpn = RPN()
roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

# optimizer = optim.Adam(rpn.parameters(), lr=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# 	optimizer, patience=3, verbose=True)
# # x = torch.Tensor(2, 3, 224, 224)
# # boxes, losses = rpn(x)
# # print(boxes)
# print(losses)
n_epochs = 100



rpn.train()
roi_heads.train()

for epoch in range(1, n_epochs+1):
	loss = []
	for i, data in enumerate(dataloader):
		images, annotations = data
		proposals, losses, features, image_shapes = rpn(images, annotations)
		roi_heads(features,     
				proposals,     
				image_shapes, 
				annotations)
		break
	break





	# 	final_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
	# 	loss.append(final_loss.item())

	# 	optimizer.zero_grad()
	# 	final_loss.backward()
	# 	optimizer.step()
	# 	print(f'loss : {final_loss.item()},\n\
	# 			cls_loss : {losses["loss_objectness"].item()},\n\
	# 			reg_loss : {losses["loss_rpn_box_reg"].item()}')

	# loss = torch.tensor(loss, dtype=torch.float32)
	# print(f'loss : {torch.mean(loss)}')
	# # scheduler.step(torch.mean(loss))


	# state = {'state_dict': rpn.state_dict()}
	# torch.save(state, os.path.join('./snapshots', f'rpn.pth'))
	# print("model saved")

