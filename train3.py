

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead, AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
# from datasets.pascal_voc import VOCDataset, collater
from datasets.vrd import VRDDataset, collater
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

from torch.jit.annotations import Optional, List, Dict, Tuple
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
	# type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
	"""
	Computes the loss for Faster R-CNN.

	Arguments:
		class_logits (Tensor)
		box_regression (Tensor)
		labels (list[BoxList])
		regression_targets (Tensor)

	Returns:
		classification_loss (Tensor)
		box_loss (Tensor)
	"""

	labels = torch.cat(labels, dim=0)
	regression_targets = torch.cat(regression_targets, dim=0)

	classification_loss = F.cross_entropy(class_logits, labels)

	# get indices that correspond to the regression targets for
	# the corresponding ground truth labels, to be used with
	# advanced indexing
	sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
	labels_pos = labels[sampled_pos_inds_subset]
	N, num_classes = class_logits.shape
	box_regression = box_regression.reshape(N, -1, 4)

	box_loss = det_utils.smooth_l1_loss(
		box_regression[sampled_pos_inds_subset, labels_pos],
		regression_targets[sampled_pos_inds_subset],
		beta=1 / 9,
		size_average=False,
	)
	box_loss = box_loss / labels.numel()

	return classification_loss, box_loss


# dataset_train = VOCDataset(root='/home/neuroplex/data/VOCdevkit/VOC2007')
# dataset_train = VOCDataset(root='/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VOCdevkit2007/VOC2007')
# dataloader = DataLoader(
# 	dataset_train, num_workers=0, collate_fn=collater, batch_size=1)


# dataset_train = VRDDataset('/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VRD', 'train')
dataset_train = VRDDataset('/home/neuroplex/code/faster-rcnn/data/VRD', 'train')
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
	
	def assign_targets_to_relation_proposals(self, proposals, gt_boxes, gt_labels):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		# proposals size -> 64 * 64 
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

	def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
		# type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
		matched_idxs = []
		labels = []
		# if assign_to == "subject":
		# 	slice_index = 0
		# else:
		# 	slice_index = 1
		for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
			
			# gt_boxes_in_image = gt_boxes_in_image[:,slice_index,:]
			# gt_labels_in_image = gt_labels_in_image[:,slice_index]

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
			torch.cat((proposal, gt_box.view(-1,4)))
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

		gt_boxes = [t["boxes"].to(dtype).view(-1,4) for t in targets]
		gt_labels = [t["labels"].view(-1) for t in targets]
		# gt_preds = [t["preds"] for t in targets]

		# append ground-truth bboxes to propos
		proposals = self.add_gt_proposals(proposals, gt_boxes)


		# # get matching gt indices for each proposal
		# sub_matched_idxs, sub_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="subject")
		# sampled_inds = self.subsample(sub_labels)   			#	size 64 --> 32 pos, 32 neg
		# sub_proposals = proposals.copy()
		# sub_matched_gt_boxes = []
		# num_images = len(proposals)
		# for img_id in range(num_images):
		# 	img_sampled_inds = sampled_inds[img_id]
		# 	sub_proposals[img_id] = sub_proposals[img_id][img_sampled_inds]
		# 	sub_labels[img_id] = sub_labels[img_id][img_sampled_inds]
		# 	sub_matched_idxs[img_id] = sub_matched_idxs[img_id][img_sampled_inds]

		# 	gt_boxes_in_image = gt_boxes[img_id][:,0,:]
		# 	if gt_boxes_in_image.numel() == 0:
		# 		gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
		# 	sub_matched_gt_boxes.append(gt_boxes_in_image[sub_matched_idxs[img_id]])

		# sub_regression_targets = self.box_coder.encode(sub_matched_gt_boxes, sub_proposals)




		# # get matching gt indices for each proposal
		# obj_matched_idxs, obj_labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, assign_to="objects")
		# sampled_inds = self.subsample(obj_labels)   				#size 64 --> 32 pos, 32 neg
		# obj_proposals = proposals.copy()
		# obj_matched_gt_boxes = []
		# num_images = len(proposals)
		# for img_id in range(num_images):
		# 	img_sampled_inds = sampled_inds[img_id]
		# 	obj_proposals[img_id] = obj_proposals[img_id][img_sampled_inds]
		# 	obj_labels[img_id] = obj_labels[img_id][img_sampled_inds]
		# 	obj_matched_idxs[img_id] = obj_matched_idxs[img_id][img_sampled_inds]

		# 	gt_boxes_in_image = gt_boxes[img_id][:,1,:]
		# 	if gt_boxes_in_image.numel() == 0:
		# 		gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
		# 	obj_matched_gt_boxes.append(gt_boxes_in_image[obj_matched_idxs[img_id]])

		# obj_regression_targets = self.box_coder.encode(obj_matched_gt_boxes, obj_proposals)






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

		# labels = []
		# regression_targets = []
		# matched_idxs = []
		# proposals = []
		# for i in range(len(sub_proposals)):
		# 	proposals.append(torch.cat([sub_proposals[i], obj_proposals[i]]))
		# 	matched_idxs.append(torch.cat([sub_matched_idxs[i], obj_matched_idxs[i]]))
		# 	labels.append(torch.cat([sub_labels[i],obj_labels[i]]))
		# 	regression_targets.append(torch.cat([sub_regression_targets[i],obj_regression_targets[i]]))
			


		# print(proposals[0].shape)
		# print(matched_idxs[0].shape)
		# print(labels[0].shape)
		# print(regression_targets[0].shape)

		# print(sub_proposals[0].shape)            # --> 64,4
		# print(sub_matched_idxs[0].shape)	   	 #  size --> 64
		# print(sub_labels[0].shape)               #size --> 64
		# print(sub_regression_targets[0].shape)   #size --> 64,4



		# print(obj_proposals[0].shape)            size --> 64,4
		# print(obj_matched_idxs[0].shape)	   	   size --> 64
		# print(obj_labels[0].shape)               size --> 64
		# print(obj_regression_targets[0].shape)   size --> 64,4



		# prepare relation candidates
		# relation_proposals = create_relation_proposals(sub_proposals, obj_proposals)   size --> 64 * 64 = 4096 relation proposals
		# assign predicate to subjects
		# matched_idxs, sub_labels = self.assign_targets_to_relation_proposals(relation_proposals, gt_boxes, gt_labels)
		# assign predicate to objects
		# matched_idxs, obj_labels = self.assign_targets_to_proposals(obj_proposals, obj_regression_targets, predicates)
		# compare sub_labels == obj_labels 
		# gt_predicates =    of size   List[tensor[] of size 512 * 512 ]
		# return gt_predicates, sub_obj_proposals

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
			# print(labels[0].shape)
		else:
			labels = None
			regression_targets = None
			matched_idxs = None


		# unioned_relation_box_proposals = 
		# sub_box_proposals = sub_obj_proposals[:,0,:]
		# obj_box_proposals = sub_obj_proposals[:,1,:]

		# sub_box_features = self.box_roi_pool(features, sub_box_proposals, image_shapes)
		# obj_box_features = self.box_roi_pool(features, obj_box_proposals, image_shapes)
		# rel_box_features = self.box_roi_pool(features, unioned_relation_box_proposals, image_shapes)

		# concat relation_box_candidates, box_features
		# class_logits = self.relation_box_predictor(box_features)

		box_features = self.box_roi_pool(features, proposals, image_shapes)
		box_features = self.box_head(box_features)

		class_logits, box_regression = self.box_predictor(box_features)

		result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
		losses = {}
		if self.training:
			assert labels is not None and regression_targets is not None
			loss_classifier, loss_box_reg = fastrcnn_loss(
				class_logits, box_regression, labels, regression_targets)
			losses = {
				"loss_classifier": loss_classifier,
				"loss_box_reg": loss_box_reg
			}
		else:
			boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
			num_images = len(boxes)
			for i in range(num_images):
				result.append(
					{
						"boxes": boxes[i],
						"labels": labels[i],
						"scores": scores[i],
					}
				)
		   
		return result, losses


class RPN(nn.Module):
	def __init__(self):
		super(RPN, self).__init__()
		# Define FPN
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

	def prepare_gt_for_rpn(self, targets):
		gth_list = []
		for target in targets:
			gt = {}
			gt["boxes"] = target["boxes"].view(-1,4)
			gt["labels"] = target["labels"].view(-1)
			gth_list.append(gt)
		return gth_list
	
	def forward(self, images, fpn_feature_maps, targets=None):
		# l = torch.FloatTensor([[1,2,3,4],[1,2,3,4]])
		# targets = [{"boxes":l},{"boxes":l}]
		# targets = [{i: index for i, index in enumerate(l)}]

		targets = self.prepare_gt_for_rpn(targets)
		fpn_feature_maps = self.fpn(images.tensors.to(DEVICE))
		
		if self.training:
			boxes, losses = self.rpn(images, fpn_feature_maps, targets)
		else:
			boxes, losses = self.rpn(images, fpn_feature_maps)
		return boxes, losses, fpn_feature_maps




class FasterRCNN(nn.Module):
	def __init__(self):
		super(FasterRCNN, self).__init__()
		# Define FPN
		self.fpn = resnet_fpn_backbone(backbone_name='resnet101', pretrained=True)
		self.rpn = RPN()
	

		# transform parameters
		min_size = 800
		max_size = 1333
		image_mean = [0.485, 0.456, 0.406]
		image_std = [0.229, 0.224, 0.225]
		self.transform = GeneralizedRCNNTransform(
			min_size, max_size, image_mean, image_std)

		
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
		num_classes=101

		if box_roi_pool is None:
			box_roi_pool = MultiScaleRoIAlign(
				featmap_names=['0', '1', '2', '3', '4'],
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
		fpn_feature_maps = self.fpn(images.tensors.to(DEVICE))
		
		if self.training:
			proposals, rpn_losses, fpn_feature_maps = self.rpn(images, fpn_feature_maps, targets)
			detections, detector_losses = self.roi_heads(fpn_feature_maps, proposals, images.image_sizes, targets)
			# detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
			losses = {}
			losses.update(detector_losses)
			losses.update(rpn_losses)
		else:
			losses = {}
			proposals, rpn_losses, fpn_feature_maps = self.rpn(images, fpn_feature_maps)
			detections, detector_losses = self.roi_heads(fpn_feature_maps, proposals, images.image_sizes)
			detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
		return detections, losses


			

faster_rcnn = FasterRCNN().to(DEVICE)

optimizer = optim.Adam(faster_rcnn.parameters(), lr=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# 	optimizer, patience=3, verbose=True)
# # x = torch.Tensor(2, 3, 224, 224)
# # boxes, losses = rpn(x)
# # print(boxes)
# print(losses)
n_epochs = 100

faster_rcnn.train()

for epoch in range(1, n_epochs+1):
	loss = []
	for i, data in enumerate(dataloader):
		images, targets = data
		result, rpn_losses, head_losses = faster_rcnn(images, targets)
		
	# 	break
	# break
	
		final_loss = rpn_losses["loss_objectness"] + rpn_losses["loss_rpn_box_reg"] + head_losses["loss_classifier"] + head_losses["loss_box_reg"]
		loss.append(final_loss.item())

		optimizer.zero_grad()
		final_loss.backward()
		optimizer.step()
		print(f'RCNN_Loss    : {final_loss.item()},\n\
				rpn_cls_loss : {rpn_losses["loss_objectness"].item()},\n\
				rpn_reg_loss : {rpn_losses["loss_rpn_box_reg"].item()}\n\
				box_loss 	 : {head_losses["loss_box_reg"]}\n\
				cls_loss     : {head_losses["loss_classifier"]}')

	loss = torch.tensor(loss, dtype=torch.float32)
	print(f'loss : {torch.mean(loss)}')
	# scheduler.step(torch.mean(loss))


	state = {'state_dict': faster_rcnn.state_dict()}
	torch.save(state, os.path.join('./snapshots', f'faster_rcnn.pth'))
	print("model saved")

