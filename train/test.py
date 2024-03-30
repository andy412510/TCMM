import torch

# Create a random tensor of size 3x3
a = torch.rand(512,1,384)
b = torch.rand(512,128,384)

c = torch.einsum("bxd,byd->bxy",[a,b])