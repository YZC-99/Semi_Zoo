import torch
import torch.nn.functional as F

label1 = torch.tensor([[0,1,1,1]])
label2 = torch.tensor([[0,1,1,0]])
print(label2.shape)
label = torch.cat([label1,label2],dim=0)
print(label.shape)
onehot = F.one_hot(label).permute(2,0,1)
print(onehot.shape)
print(onehot)