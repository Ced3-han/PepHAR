import torch

a = torch.randn(10,3)
print(a)
print(torch.cuda.is_available())