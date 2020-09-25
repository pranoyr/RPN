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