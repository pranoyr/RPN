import torch
loss = [1,2,3]
loss = torch.tensor(loss, dtype=torch.float32)
print(f'loss : {loss.mean()}')
