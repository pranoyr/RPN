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