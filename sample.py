# import torch
# loss = [1,2,3]
# loss = torch.tensor(loss, dtype=torch.float32)
# print(f'loss : {loss.mean()}')
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
fpn = resnet_fpn_backbone(backbone_name='resnet101', pretrained=True)



x = torch.Tensor(2,3,224,224)
x = fpn(x)
print(x['1'].shape)


for s, s_orig in zip([1,2], [1,2]):
	s=1




def resize_boxes(boxes):
	xmin, ymin, xmax, ymax = boxes.unbind(1)
	return torch.stack((xmin, ymin, xmax, ymax), dim=1)


boxes = torch.tensor([[1,2,3,4],[1,2,3,4]])
print(boxes.shape)
boxes = resize_boxes(boxes)
print(boxes.shape)


a = [1,2,3,7]
b = [1,2,3,4]

assert len(a) == len(b), "labels and boxes should be of equal length"


a = torch.tensor(  [ [[1,2,3,4],[5,6,7,8]], 
					[[9,10,11,12],[13,14,15,16]]  ])

b = torch.tensor(  [ [1,2], 
					[3,4] ]  )




a = a.view(-1,4)
b = b.view(-1)

print(a.view(-1,2,4))
print(b.view(-1,2))



input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output.shape)


import numpy as np


def rois_union(rois1, rois2):
    assert (rois1[:, 0] == rois2[:, 0]).all()
    xmin = np.minimum(rois1[:, 1], rois2[:, 1])
    ymin = np.minimum(rois1[:, 2], rois2[:, 2])
    xmax = np.maximum(rois1[:, 3], rois2[:, 3])
    ymax = np.maximum(rois1[:, 4], rois2[:, 4])
    return np.vstack((rois1[:, 0], xmin, ymin, xmax, ymax)).transpose()



a = np.array([[1,2,3,3],
			[1,2,3,4]])

b = np.array([[1,2,3,3],
			[2,2,3,4]])


# print(rois_union(a,b))

# c = []
# c[0]=1

print(torch.Tensor((1,2)).device)
# import json
# import os
# with open(os.path.join('/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VRD', 'json_dataset', 'objects.json'), 'r') as f:
# 	objects = json.load(f)

# classes = ['__background__']
# classes.extend(objects)
# num_classes = len(classes)
# # self._classes.extend(self.predicates)
# _class_to_ind = dict(zip(range(num_classes), classes))
# print(_class_to_ind)

x = torch.tensor([0,0,0,1,0,2])

y = torch.tensor([0,0,0,1,0,2])

sorted, indices = torch.sort(x, 0, True)

print(sorted)

print(y[indices])

