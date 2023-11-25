import torch

input1 = torch.tensor([1,2,3])
input2 = torch.tensor([[1,2,3],[1,2,3]])
soft1 = torch.sigmoid(input1)
soft2 = torch.sigmoid(input2)
print("soft1")
print(soft1)
print("soft2")
print(soft2)